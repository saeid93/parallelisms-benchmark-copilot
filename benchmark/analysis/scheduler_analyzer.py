"""Batch scheduler analyzer for LLM serving systems.

Inspired by:
* Orca: A Distributed Serving System for Transformer-Based Models (Yu et al. 2022)
* SARATHI: Efficient LLM Inference with Piggybacked Decodes (Agrawal et al. 2024)
* vLLM continuous batching & preemption logic
* Splitwise: Efficient Generative LLM Inference with Flow Splitting

Analyses scheduling decisions, batch composition over time,
preemption patterns, request priority fairness (Jain's fairness
index), and waiting time distributions.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SchedulingPolicy(str, Enum):
    """Scheduling policies for request batching."""

    FCFS = "fcfs"
    SHORTEST_JOB_FIRST = "sjf"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"
    CHUNKED_PREFILL = "chunked_prefill"
    CONTINUOUS_BATCHING = "continuous_batching"


class PreemptionReason(str, Enum):
    """Why a request was preempted."""

    KV_CACHE_FULL = "kv_cache_full"
    HIGHER_PRIORITY = "higher_priority"
    MAX_BATCH_TOKENS = "max_batch_tokens"
    MAX_BATCH_SEQS = "max_batch_seqs"
    TIMEOUT = "timeout"
    RESOURCE_CONTENTION = "resource_contention"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class SchedulingDecision:
    """A single scheduling step record."""

    step_index: int
    timestamp_us: float
    batch_size: int = 0
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    total_tokens_in_batch: int = 0
    max_seq_len_in_batch: int = 0
    gpu_memory_used_pct: float = 0.0
    preempted_request_ids: List[str] = field(default_factory=list)
    preemption_reasons: List[PreemptionReason] = field(default_factory=list)
    newly_admitted: int = 0
    queue_depth: int = 0
    scheduling_latency_us: float = 0.0
    chunked_prefill_count: int = 0


@dataclass
class RequestSchedulingInfo:
    """Per-request scheduling metrics."""

    request_id: str
    arrival_us: float = 0.0
    first_scheduled_us: float = 0.0
    completion_us: float = 0.0
    total_wait_us: float = 0.0
    num_preemptions: int = 0
    total_preempt_us: float = 0.0
    priority: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    batches_participated: int = 0

    @property
    def scheduling_delay_us(self) -> float:
        return self.first_scheduled_us - self.arrival_us if self.first_scheduled_us > 0 else 0.0

    @property
    def service_time_us(self) -> float:
        return self.completion_us - self.first_scheduled_us if self.completion_us > 0 else 0.0


@dataclass
class FairnessMetrics:
    """Fairness analysis of scheduling."""

    jains_fairness_index: float = 0.0
    max_min_ratio: float = 0.0
    cv_wait_time: float = 0.0  # coefficient of variation
    gini_coefficient: float = 0.0
    starvation_count: int = 0  # requests waiting > 2x mean
    starvation_threshold_us: float = 0.0


@dataclass
class SchedulerStats:
    """Aggregated scheduler analysis."""

    total_steps: int = 0
    mean_batch_size: float = 0.0
    max_batch_size: int = 0
    mean_tokens_per_batch: float = 0.0
    prefill_decode_ratio: float = 0.0
    total_preemptions: int = 0
    preemption_rate: float = 0.0
    preemption_reasons: Dict[str, int] = field(default_factory=dict)
    mean_queue_depth: float = 0.0
    max_queue_depth: int = 0
    mean_scheduling_latency_us: float = 0.0
    mean_wait_us: float = 0.0
    p99_wait_us: float = 0.0
    fairness: FairnessMetrics = field(default_factory=FairnessMetrics)
    batch_utilization_pct: float = 0.0  # tokens / max_tokens capacity
    chunked_prefill_ratio: float = 0.0

    def summary(self) -> str:
        lines = [
            "Scheduler Analysis",
            f"  Total steps           : {self.total_steps}",
            f"  Mean batch size       : {self.mean_batch_size:.1f}",
            f"  Max batch size        : {self.max_batch_size}",
            f"  Mean tokens/batch     : {self.mean_tokens_per_batch:.0f}",
            f"  Prefill/decode ratio  : {self.prefill_decode_ratio:.2f}",
            f"  Total preemptions     : {self.total_preemptions}",
            f"  Preemption rate       : {self.preemption_rate:.2%}",
            f"  Mean queue depth      : {self.mean_queue_depth:.1f}",
            f"  Max queue depth       : {self.max_queue_depth}",
            f"  Mean scheduling lat.  : {self.mean_scheduling_latency_us:.1f} us",
            f"  Mean wait time        : {self.mean_wait_us / 1e3:.1f} ms",
            f"  P99 wait time         : {self.p99_wait_us / 1e3:.1f} ms",
            f"  Batch utilization     : {self.batch_utilization_pct:.1f}%",
            f"  Chunked prefill ratio : {self.chunked_prefill_ratio:.1%}",
            f"  Jain's fairness       : {self.fairness.jains_fairness_index:.4f}",
            f"  Gini coefficient      : {self.fairness.gini_coefficient:.4f}",
            f"  Starvation count      : {self.fairness.starvation_count}",
        ]
        if self.preemption_reasons:
            lines.append("  Preemption reasons:")
            for reason, count in sorted(self.preemption_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _jains_fairness(values: List[float]) -> float:
    """Compute Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2)."""
    if not values:
        return 1.0
    n = len(values)
    s = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return 1.0
    return (s * s) / (n * s2)


def _gini(values: List[float]) -> float:
    """Compute Gini coefficient."""
    if not values or len(values) < 2:
        return 0.0
    s = sorted(values)
    n = len(s)
    total = sum(s)
    if total == 0:
        return 0.0
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(s):
        cumsum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


class SchedulerAnalyzer:
    """Analyses batch scheduling decisions and fairness.

    Usage::

        sa = SchedulerAnalyzer(max_batch_tokens=4096)
        sa.record_decision(SchedulingDecision(...))
        sa.record_request(RequestSchedulingInfo(...))
        stats = sa.analyse()
        print(stats.summary())
    """

    def __init__(self, max_batch_tokens: int = 4096, max_batch_seqs: int = 256) -> None:
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_seqs = max_batch_seqs
        self._decisions: List[SchedulingDecision] = []
        self._requests: List[RequestSchedulingInfo] = []

    def record_decision(self, decision: SchedulingDecision) -> None:
        self._decisions.append(decision)

    def record_request(self, info: RequestSchedulingInfo) -> None:
        self._requests.append(info)

    def analyse(self) -> SchedulerStats:
        if not self._decisions:
            return SchedulerStats()

        batch_sizes = [d.batch_size for d in self._decisions]
        tokens = [d.total_tokens_in_batch for d in self._decisions]
        queues = [d.queue_depth for d in self._decisions]
        sched_lats = [d.scheduling_latency_us for d in self._decisions]
        total_prefill = sum(d.num_prefill_requests for d in self._decisions)
        total_decode = sum(d.num_decode_requests for d in self._decisions)
        total_preempted = sum(len(d.preempted_request_ids) for d in self._decisions)
        chunked = sum(d.chunked_prefill_count for d in self._decisions)
        total_steps_with_prefill = sum(1 for d in self._decisions if d.num_prefill_requests > 0)

        # Preemption reasons
        reasons: Dict[str, int] = {}
        for d in self._decisions:
            for r in d.preemption_reasons:
                reasons[r.value] = reasons.get(r.value, 0) + 1

        # Batch utilization
        utils = []
        for d in self._decisions:
            if self._max_batch_tokens > 0:
                utils.append(100.0 * d.total_tokens_in_batch / self._max_batch_tokens)

        # Fairness analysis
        wait_times = [r.total_wait_us for r in self._requests]
        fairness = FairnessMetrics()
        if wait_times:
            mean_wait = statistics.mean(wait_times)
            fairness.jains_fairness_index = _jains_fairness(wait_times)
            fairness.gini_coefficient = _gini(wait_times)
            fairness.cv_wait_time = (
                statistics.stdev(wait_times) / mean_wait
                if mean_wait > 0 and len(wait_times) > 1
                else 0.0
            )
            if wait_times:
                fairness.max_min_ratio = (
                    max(wait_times) / min(wait_times)
                    if min(wait_times) > 0
                    else float("inf")
                )
            starvation_threshold = 2.0 * mean_wait if mean_wait > 0 else float("inf")
            fairness.starvation_count = sum(1 for w in wait_times if w > starvation_threshold)
            fairness.starvation_threshold_us = starvation_threshold

        pd_ratio = total_prefill / total_decode if total_decode > 0 else 0.0

        return SchedulerStats(
            total_steps=len(self._decisions),
            mean_batch_size=statistics.mean(batch_sizes) if batch_sizes else 0.0,
            max_batch_size=max(batch_sizes) if batch_sizes else 0,
            mean_tokens_per_batch=statistics.mean(tokens) if tokens else 0.0,
            prefill_decode_ratio=pd_ratio,
            total_preemptions=total_preempted,
            preemption_rate=(
                total_preempted / len(self._decisions) if self._decisions else 0.0
            ),
            preemption_reasons=reasons,
            mean_queue_depth=statistics.mean(queues) if queues else 0.0,
            max_queue_depth=max(queues) if queues else 0,
            mean_scheduling_latency_us=(
                statistics.mean(sched_lats) if sched_lats else 0.0
            ),
            mean_wait_us=statistics.mean(wait_times) if wait_times else 0.0,
            p99_wait_us=_percentile(wait_times, 99),
            fairness=fairness,
            batch_utilization_pct=statistics.mean(utils) if utils else 0.0,
            chunked_prefill_ratio=(
                chunked / total_steps_with_prefill if total_steps_with_prefill > 0 else 0.0
            ),
        )

    def export_batch_timeline(self) -> List[Dict]:
        """Export batch composition over time."""
        return [
            {
                "step": d.step_index,
                "timestamp_us": d.timestamp_us,
                "batch_size": d.batch_size,
                "prefill_reqs": d.num_prefill_requests,
                "decode_reqs": d.num_decode_requests,
                "total_tokens": d.total_tokens_in_batch,
                "queue_depth": d.queue_depth,
                "preemptions": len(d.preempted_request_ids),
                "mem_used_pct": d.gpu_memory_used_pct,
            }
            for d in self._decisions
        ]

    def reset(self) -> None:
        self._decisions.clear()
        self._requests.clear()
