"""End-to-end request lifecycle tracer with Gantt chart support.

Inspired by:
* Google Dapper distributed tracing
* Jaeger / OpenTelemetry tracing models
* vLLM request scheduling internals
* "Orca: A Distributed Serving System for Transformer-Based Models"

Provides fine-grained request lifecycle tracking from arrival
through completion, with Gantt chart export, queue time breakdown,
scheduling delay analysis, inter-token latency jitter, and
tail latency root cause analysis.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RequestPhase(str, Enum):
    """Lifecycle phases of a serving request."""

    ARRIVAL = "arrival"
    QUEUE_WAIT = "queue_wait"
    TOKENIZATION = "tokenization"
    SCHEDULING = "scheduling"
    PREFILL = "prefill"
    DECODE = "decode"
    KV_TRANSFER = "kv_transfer"
    DETOKENIZATION = "detokenization"
    RESPONSE = "response"
    PREEMPTED = "preempted"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"


class TailLatencyCause(str, Enum):
    """Root causes of tail latency."""

    QUEUE_DELAY = "queue_delay"
    LONG_PREFILL = "long_prefill"
    PREEMPTION = "preemption"
    KV_CACHE_MISS = "kv_cache_miss"
    BATCH_WAIT = "batch_wait"
    DECODE_JITTER = "decode_jitter"
    SCHEDULING_DELAY = "scheduling_delay"
    NETWORK_LATENCY = "network_latency"
    GC_PAUSE = "gc_pause"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimestamp:
    """Start/end of a request phase."""

    phase: RequestPhase
    start_us: float
    end_us: float
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


@dataclass
class InterTokenTiming:
    """Per-token decode timing for jitter analysis."""

    token_index: int
    timestamp_us: float
    duration_us: float  # time since previous token


@dataclass
class RequestLifecycle:
    """Complete lifecycle of a single request."""

    request_id: str
    arrival_us: float = 0.0
    completion_us: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    phases: List[PhaseTimestamp] = field(default_factory=list)
    inter_token_timings: List[InterTokenTiming] = field(default_factory=list)
    preemption_count: int = 0
    batch_position: int = 0
    scheduled_batch_size: int = 0
    priority: float = 0.0
    tail_cause: Optional[TailLatencyCause] = None

    @property
    def e2e_latency_us(self) -> float:
        return self.completion_us - self.arrival_us if self.completion_us > 0 else 0.0

    @property
    def queue_time_us(self) -> float:
        return sum(p.duration_us for p in self.phases if p.phase == RequestPhase.QUEUE_WAIT)

    @property
    def prefill_time_us(self) -> float:
        return sum(p.duration_us for p in self.phases if p.phase == RequestPhase.PREFILL)

    @property
    def decode_time_us(self) -> float:
        return sum(p.duration_us for p in self.phases if p.phase == RequestPhase.DECODE)

    @property
    def ttft_us(self) -> float:
        """Time to first token: arrival -> first decode output."""
        for p in self.phases:
            if p.phase == RequestPhase.DECODE:
                return p.start_us - self.arrival_us
        # fallback: arrival -> prefill end
        for p in self.phases:
            if p.phase == RequestPhase.PREFILL:
                return p.end_us - self.arrival_us
        return 0.0

    @property
    def mean_itl_us(self) -> float:
        """Mean inter-token latency."""
        if not self.inter_token_timings:
            return 0.0
        return statistics.mean([t.duration_us for t in self.inter_token_timings])

    @property
    def itl_jitter(self) -> float:
        """Coefficient of variation of inter-token latency."""
        if len(self.inter_token_timings) < 2:
            return 0.0
        durations = [t.duration_us for t in self.inter_token_timings]
        mean = statistics.mean(durations)
        if mean <= 0:
            return 0.0
        return statistics.stdev(durations) / mean

    def phase_breakdown(self) -> Dict[str, float]:
        """Return time spent in each phase."""
        result: Dict[str, float] = {}
        for p in self.phases:
            result[p.phase.value] = result.get(p.phase.value, 0.0) + p.duration_us
        return result

    def to_gantt_events(self) -> List[Dict]:
        """Export phases as Gantt chart data."""
        return [
            {
                "request_id": self.request_id,
                "phase": p.phase.value,
                "start_us": p.start_us,
                "end_us": p.end_us,
                "duration_us": p.duration_us,
                "metadata": p.metadata,
            }
            for p in self.phases
        ]


@dataclass
class LifecycleStats:
    """Aggregated request lifecycle statistics."""

    total_requests: int = 0
    mean_e2e_us: float = 0.0
    p50_e2e_us: float = 0.0
    p90_e2e_us: float = 0.0
    p99_e2e_us: float = 0.0
    mean_queue_us: float = 0.0
    p99_queue_us: float = 0.0
    mean_prefill_us: float = 0.0
    mean_decode_us: float = 0.0
    mean_ttft_us: float = 0.0
    p99_ttft_us: float = 0.0
    mean_itl_us: float = 0.0
    mean_itl_jitter: float = 0.0
    preemption_rate: float = 0.0
    tail_cause_distribution: Dict[str, int] = field(default_factory=dict)
    phase_time_pcts: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Request Lifecycle Stats",
            f"  Requests        : {self.total_requests}",
            f"  E2E mean        : {self.mean_e2e_us / 1e3:.1f} ms",
            f"  E2E p50/p90/p99 : {self.p50_e2e_us / 1e3:.1f}/{self.p90_e2e_us / 1e3:.1f}/{self.p99_e2e_us / 1e3:.1f} ms",
            f"  Queue mean      : {self.mean_queue_us / 1e3:.1f} ms (p99: {self.p99_queue_us / 1e3:.1f} ms)",
            f"  TTFT mean       : {self.mean_ttft_us / 1e3:.1f} ms (p99: {self.p99_ttft_us / 1e3:.1f} ms)",
            f"  ITL mean        : {self.mean_itl_us:.1f} us (jitter CV: {self.mean_itl_jitter:.3f})",
            f"  Preemption rate : {self.preemption_rate:.1%}",
        ]
        if self.tail_cause_distribution:
            lines.append("  Tail causes:")
            for cause, count in sorted(self.tail_cause_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"    {cause}: {count}")
        if self.phase_time_pcts:
            lines.append("  Phase breakdown:")
            for phase, pct in sorted(self.phase_time_pcts.items(), key=lambda x: -x[1]):
                lines.append(f"    {phase}: {pct:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lifecycle Tracer
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


class RequestLifecycleTracer:
    """Tracks complete request lifecycles and provides analysis.

    Usage::

        tracer = RequestLifecycleTracer()
        lc = tracer.new_lifecycle("req-1")
        lc.phases.append(PhaseTimestamp(RequestPhase.QUEUE_WAIT, t0, t1))
        lc.phases.append(PhaseTimestamp(RequestPhase.PREFILL, t1, t2))
        ...
        tracer.record(lc)
        stats = tracer.compute_stats()
        print(stats.summary())
    """

    def __init__(self) -> None:
        self._lifecycles: List[RequestLifecycle] = []

    def new_lifecycle(
        self, request_id: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> RequestLifecycle:
        """Create a new lifecycle object (not yet recorded)."""
        return RequestLifecycle(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def record(self, lifecycle: RequestLifecycle) -> None:
        """Record a completed request lifecycle."""
        self._lifecycles.append(lifecycle)

    def record_batch(self, lifecycles: List[RequestLifecycle]) -> None:
        self._lifecycles.extend(lifecycles)

    def _classify_tail_cause(self, lc: RequestLifecycle, p99_threshold: float) -> Optional[TailLatencyCause]:
        """Classify root cause for tail-latency requests."""
        if lc.e2e_latency_us < p99_threshold:
            return None

        breakdown = lc.phase_breakdown()
        total = lc.e2e_latency_us
        if total <= 0:
            return TailLatencyCause.UNKNOWN

        queue_frac = breakdown.get(RequestPhase.QUEUE_WAIT.value, 0) / total
        prefill_frac = breakdown.get(RequestPhase.PREFILL.value, 0) / total

        if lc.preemption_count > 0:
            return TailLatencyCause.PREEMPTION
        if queue_frac > 0.5:
            return TailLatencyCause.QUEUE_DELAY
        if prefill_frac > 0.6:
            return TailLatencyCause.LONG_PREFILL
        if lc.itl_jitter > 0.5:
            return TailLatencyCause.DECODE_JITTER
        return TailLatencyCause.UNKNOWN

    def compute_stats(self) -> LifecycleStats:
        """Compute aggregate lifecycle statistics."""
        if not self._lifecycles:
            return LifecycleStats()

        e2e = [lc.e2e_latency_us for lc in self._lifecycles if lc.e2e_latency_us > 0]
        queues = [lc.queue_time_us for lc in self._lifecycles]
        prefills = [lc.prefill_time_us for lc in self._lifecycles]
        decodes = [lc.decode_time_us for lc in self._lifecycles]
        ttfts = [lc.ttft_us for lc in self._lifecycles if lc.ttft_us > 0]
        itls = [lc.mean_itl_us for lc in self._lifecycles if lc.mean_itl_us > 0]
        jitters = [lc.itl_jitter for lc in self._lifecycles if lc.itl_jitter > 0]
        preempted = sum(1 for lc in self._lifecycles if lc.preemption_count > 0)

        # Tail cause analysis
        p99_e2e = _percentile(e2e, 99)
        tail_causes: Dict[str, int] = {}
        for lc in self._lifecycles:
            cause = self._classify_tail_cause(lc, p99_e2e)
            if cause:
                lc.tail_cause = cause
                tail_causes[cause.value] = tail_causes.get(cause.value, 0) + 1

        # Phase breakdown (aggregate)
        total_phase_time: Dict[str, float] = {}
        for lc in self._lifecycles:
            for phase, dur in lc.phase_breakdown().items():
                total_phase_time[phase] = total_phase_time.get(phase, 0.0) + dur
        grand_total = sum(total_phase_time.values())
        phase_pcts = {
            phase: 100.0 * dur / grand_total
            for phase, dur in total_phase_time.items()
        } if grand_total > 0 else {}

        return LifecycleStats(
            total_requests=len(self._lifecycles),
            mean_e2e_us=statistics.mean(e2e) if e2e else 0.0,
            p50_e2e_us=_percentile(e2e, 50),
            p90_e2e_us=_percentile(e2e, 90),
            p99_e2e_us=p99_e2e,
            mean_queue_us=statistics.mean(queues) if queues else 0.0,
            p99_queue_us=_percentile(queues, 99),
            mean_prefill_us=statistics.mean(prefills) if prefills else 0.0,
            mean_decode_us=statistics.mean(decodes) if decodes else 0.0,
            mean_ttft_us=statistics.mean(ttfts) if ttfts else 0.0,
            p99_ttft_us=_percentile(ttfts, 99),
            mean_itl_us=statistics.mean(itls) if itls else 0.0,
            mean_itl_jitter=statistics.mean(jitters) if jitters else 0.0,
            preemption_rate=preempted / len(self._lifecycles) if self._lifecycles else 0.0,
            tail_cause_distribution=tail_causes,
            phase_time_pcts=phase_pcts,
        )

    def export_gantt(self) -> List[Dict]:
        """Export all requests as Gantt chart data."""
        result: List[Dict] = []
        for lc in self._lifecycles:
            result.extend(lc.to_gantt_events())
        return result

    def export_itl_timeseries(self) -> List[Dict]:
        """Export inter-token latency timeseries for jitter analysis."""
        result: List[Dict] = []
        for lc in self._lifecycles:
            for t in lc.inter_token_timings:
                result.append({
                    "request_id": lc.request_id,
                    "token_index": t.token_index,
                    "timestamp_us": t.timestamp_us,
                    "itl_us": t.duration_us,
                })
        return result

    def tail_latency_requests(self, percentile: float = 99.0) -> List[RequestLifecycle]:
        """Return requests above the given latency percentile."""
        e2e = [lc.e2e_latency_us for lc in self._lifecycles if lc.e2e_latency_us > 0]
        threshold = _percentile(e2e, percentile)
        return [lc for lc in self._lifecycles if lc.e2e_latency_us >= threshold]

    def reset(self) -> None:
        self._lifecycles.clear()

    def __len__(self) -> int:
        return len(self._lifecycles)
