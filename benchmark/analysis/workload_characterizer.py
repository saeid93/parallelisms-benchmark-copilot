"""Workload characterizer for LLM serving benchmarks.

Inspired by:
* SPEC CPU / Cloud benchmarking suites
* CloudSuite workload characterization
* "Characterizing Deep Learning Training Workloads" (Mattson et al.)
* vLLM request pattern analysis

Analyses request distribution characteristics, classifies workload
phases, clusters similar requests, and generates synthetic workload
profiles that match observed patterns.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class WorkloadPhase(str, Enum):
    """Identified workload phases."""

    RAMP_UP = "ramp_up"
    STEADY_STATE = "steady_state"
    BURST = "burst"
    COOL_DOWN = "cool_down"
    IDLE = "idle"


class IntensityClass(str, Enum):
    """Workload intensity classification."""

    COMPUTE_INTENSIVE = "compute_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    BALANCED = "balanced"


class RequestCategory(str, Enum):
    """Request type categories."""

    SHORT_PROMPT_SHORT_GEN = "short_prompt_short_gen"
    SHORT_PROMPT_LONG_GEN = "short_prompt_long_gen"
    LONG_PROMPT_SHORT_GEN = "long_prompt_short_gen"
    LONG_PROMPT_LONG_GEN = "long_prompt_long_gen"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class RequestCharacteristics:
    """Characteristics of a single request."""

    request_id: str
    arrival_time_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    e2e_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    category: Optional[RequestCategory] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def io_ratio(self) -> float:
        """Input/output ratio."""
        return self.input_tokens / self.output_tokens if self.output_tokens > 0 else 0.0


@dataclass
class DistributionStats:
    """Statistics for a distribution."""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    skewness: float = 0.0
    cv: float = 0.0  # coefficient of variation


@dataclass
class PhaseInfo:
    """Information about a detected workload phase."""

    phase: WorkloadPhase
    start_s: float = 0.0
    end_s: float = 0.0
    request_count: int = 0
    mean_rate_rps: float = 0.0
    description: str = ""


@dataclass
class RequestCluster:
    """A cluster of similar requests."""

    cluster_id: int
    size: int = 0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    category: Optional[RequestCategory] = None
    fraction: float = 0.0


@dataclass
class SyntheticProfile:
    """Parameters to recreate a similar workload synthetically."""

    arrival_process: str = "poisson"
    mean_rate_rps: float = 1.0
    input_token_distribution: str = "lognormal"
    input_token_mean: float = 256.0
    input_token_std: float = 128.0
    output_token_distribution: str = "lognormal"
    output_token_mean: float = 128.0
    output_token_std: float = 64.0
    burst_probability: float = 0.0
    burst_multiplier: float = 3.0
    category_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkloadProfile:
    """Full workload characterization."""

    total_requests: int = 0
    duration_s: float = 0.0
    mean_arrival_rate_rps: float = 0.0
    peak_arrival_rate_rps: float = 0.0
    input_token_stats: Optional[DistributionStats] = None
    output_token_stats: Optional[DistributionStats] = None
    latency_stats: Optional[DistributionStats] = None
    intensity_class: IntensityClass = IntensityClass.BALANCED
    category_distribution: Dict[str, float] = field(default_factory=dict)
    phases: List[PhaseInfo] = field(default_factory=list)
    clusters: List[RequestCluster] = field(default_factory=list)
    burstiness_index: float = 0.0  # 0=uniform, higher=bursty
    inter_arrival_cv: float = 0.0
    synthetic_profile: Optional[SyntheticProfile] = None
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Workload Characterization",
            f"  Total requests    : {self.total_requests}",
            f"  Duration          : {self.duration_s:.1f} s",
            f"  Mean rate         : {self.mean_arrival_rate_rps:.2f} RPS",
            f"  Peak rate         : {self.peak_arrival_rate_rps:.2f} RPS",
            f"  Intensity class   : {self.intensity_class.value}",
            f"  Burstiness index  : {self.burstiness_index:.3f}",
            f"  Inter-arrival CV  : {self.inter_arrival_cv:.3f}",
        ]
        if self.input_token_stats:
            its = self.input_token_stats
            lines.append(
                f"  Input tokens      : mean={its.mean:.0f}, median={its.median:.0f}, "
                f"std={its.std:.0f}, range=[{its.min_val:.0f}, {its.max_val:.0f}]"
            )
        if self.output_token_stats:
            ots = self.output_token_stats
            lines.append(
                f"  Output tokens     : mean={ots.mean:.0f}, median={ots.median:.0f}, "
                f"std={ots.std:.0f}, range=[{ots.min_val:.0f}, {ots.max_val:.0f}]"
            )
        if self.category_distribution:
            lines.append("  Request categories:")
            for cat, pct in sorted(self.category_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat}: {pct:.1f}%")
        if self.phases:
            lines.append("  Detected phases:")
            for p in self.phases:
                lines.append(f"    {p.phase.value}: {p.start_s:.1f}-{p.end_s:.1f}s, {p.mean_rate_rps:.1f} RPS")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Characterizer
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


def _compute_dist_stats(values: List[float]) -> DistributionStats:
    if not values:
        return DistributionStats()
    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0

    # Skewness
    if std > 0 and n > 2:
        skew = sum((v - mean) ** 3 for v in values) / (n * std**3)
    else:
        skew = 0.0

    return DistributionStats(
        count=n,
        mean=mean,
        median=median,
        std=std,
        min_val=min(values),
        max_val=max(values),
        p25=_percentile(values, 25),
        p75=_percentile(values, 75),
        p95=_percentile(values, 95),
        p99=_percentile(values, 99),
        skewness=skew,
        cv=cv,
    )


def _classify_request(
    input_tokens: int, output_tokens: int, input_median: float, output_median: float
) -> RequestCategory:
    short_in = input_tokens <= input_median
    short_out = output_tokens <= output_median
    if short_in and short_out:
        return RequestCategory.SHORT_PROMPT_SHORT_GEN
    if short_in and not short_out:
        return RequestCategory.SHORT_PROMPT_LONG_GEN
    if not short_in and short_out:
        return RequestCategory.LONG_PROMPT_SHORT_GEN
    return RequestCategory.LONG_PROMPT_LONG_GEN


class WorkloadCharacterizer:
    """Characterizes workload patterns from request data.

    Usage::

        wc = WorkloadCharacterizer()
        wc.add_request(RequestCharacteristics(...))
        profile = wc.analyse()
        print(profile.summary())
    """

    def __init__(self) -> None:
        self._requests: List[RequestCharacteristics] = []

    def add_request(self, req: RequestCharacteristics) -> None:
        self._requests.append(req)

    def add_requests(self, reqs: List[RequestCharacteristics]) -> None:
        self._requests.extend(reqs)

    def _detect_phases(self, sorted_reqs: List[RequestCharacteristics]) -> List[PhaseInfo]:
        """Detect workload phases using windowed arrival rate analysis."""
        if len(sorted_reqs) < 5:
            return []

        window_s = max(1.0, (sorted_reqs[-1].arrival_time_s - sorted_reqs[0].arrival_time_s) / 20)
        phases: List[PhaseInfo] = []
        t_start = sorted_reqs[0].arrival_time_s
        t_end = sorted_reqs[-1].arrival_time_s

        # Compute windowed rates
        t = t_start
        rates: List[Tuple[float, float]] = []
        while t < t_end:
            count = sum(1 for r in sorted_reqs if t <= r.arrival_time_s < t + window_s)
            rate = count / window_s
            rates.append((t, rate))
            t += window_s

        if not rates:
            return []

        mean_rate = statistics.mean([r for _, r in rates])

        # Classify windows into phases
        for ts, rate in rates:
            if rate < mean_rate * 0.3:
                phase = WorkloadPhase.IDLE
            elif rate > mean_rate * 2.0:
                phase = WorkloadPhase.BURST
            else:
                phase = WorkloadPhase.STEADY_STATE

            if not phases or phases[-1].phase != phase:
                count = sum(1 for r in sorted_reqs if ts <= r.arrival_time_s < ts + window_s)
                phases.append(
                    PhaseInfo(
                        phase=phase,
                        start_s=ts,
                        end_s=ts + window_s,
                        request_count=count,
                        mean_rate_rps=rate,
                    )
                )
            else:
                phases[-1].end_s = ts + window_s
                count = sum(1 for r in sorted_reqs if ts <= r.arrival_time_s < ts + window_s)
                phases[-1].request_count += count

        return phases

    def _generate_synthetic(
        self,
        input_stats: DistributionStats,
        output_stats: DistributionStats,
        mean_rate: float,
        category_dist: Dict[str, float],
        phases: List[PhaseInfo],
    ) -> SyntheticProfile:
        burst_count = sum(1 for p in phases if p.phase == WorkloadPhase.BURST)
        total_phases = len(phases) if phases else 1

        return SyntheticProfile(
            arrival_process="poisson",
            mean_rate_rps=mean_rate,
            input_token_distribution="lognormal" if input_stats.skewness > 0.5 else "normal",
            input_token_mean=input_stats.mean,
            input_token_std=input_stats.std,
            output_token_distribution="lognormal" if output_stats.skewness > 0.5 else "normal",
            output_token_mean=output_stats.mean,
            output_token_std=output_stats.std,
            burst_probability=burst_count / total_phases,
            category_weights=category_dist,
        )

    def analyse(self) -> WorkloadProfile:
        if not self._requests:
            return WorkloadProfile()

        sorted_reqs = sorted(self._requests, key=lambda r: r.arrival_time_s)

        # Basic metrics
        input_tokens = [r.input_tokens for r in self._requests if r.input_tokens > 0]
        output_tokens = [r.output_tokens for r in self._requests if r.output_tokens > 0]
        latencies = [r.e2e_latency_ms for r in self._requests if r.e2e_latency_ms > 0]

        input_stats = _compute_dist_stats(input_tokens)
        output_stats = _compute_dist_stats(output_tokens)
        latency_stats = _compute_dist_stats(latencies) if latencies else None

        # Duration and rates
        duration = sorted_reqs[-1].arrival_time_s - sorted_reqs[0].arrival_time_s if len(sorted_reqs) > 1 else 0.0
        mean_rate = len(self._requests) / duration if duration > 0 else 0.0

        # Inter-arrival times
        inter_arrivals: List[float] = []
        for i in range(1, len(sorted_reqs)):
            ia = sorted_reqs[i].arrival_time_s - sorted_reqs[i - 1].arrival_time_s
            inter_arrivals.append(ia)
        ia_cv = 0.0
        if inter_arrivals and len(inter_arrivals) > 1:
            ia_mean = statistics.mean(inter_arrivals)
            ia_std = statistics.stdev(inter_arrivals)
            ia_cv = ia_std / ia_mean if ia_mean > 0 else 0.0

        # Burstiness = (cv - 1) / (cv + 1)
        burstiness = (ia_cv - 1) / (ia_cv + 1) if ia_cv > 0 else 0.0

        # Peak rate (1s window)
        peak_rate = 0.0
        if duration > 0:
            t = sorted_reqs[0].arrival_time_s
            while t < sorted_reqs[-1].arrival_time_s:
                count = sum(1 for r in sorted_reqs if t <= r.arrival_time_s < t + 1.0)
                peak_rate = max(peak_rate, float(count))
                t += 1.0

        # Classify requests
        input_med = statistics.median(input_tokens) if input_tokens else 0
        output_med = statistics.median(output_tokens) if output_tokens else 0
        for r in self._requests:
            if r.input_tokens > 0 and r.output_tokens > 0:
                r.category = _classify_request(r.input_tokens, r.output_tokens, input_med, output_med)

        # Category distribution
        cat_counts: Dict[str, int] = {}
        for r in self._requests:
            if r.category:
                cat_counts[r.category.value] = cat_counts.get(r.category.value, 0) + 1
        total_categorized = sum(cat_counts.values())
        cat_dist = {k: 100.0 * v / total_categorized for k, v in cat_counts.items()} if total_categorized > 0 else {}

        # Intensity classification
        mean_io_ratio = statistics.mean([r.io_ratio for r in self._requests if r.io_ratio > 0]) if self._requests else 1.0
        if mean_io_ratio > 3.0:
            intensity = IntensityClass.COMPUTE_INTENSIVE  # long prompts = compute heavy
        elif mean_io_ratio < 0.5:
            intensity = IntensityClass.MEMORY_INTENSIVE  # long generation = memory heavy
        else:
            intensity = IntensityClass.BALANCED

        # Phase detection
        phases = self._detect_phases(sorted_reqs)

        # Synthetic profile
        synthetic = self._generate_synthetic(input_stats, output_stats, mean_rate, cat_dist, phases)

        recs = self._generate_recommendations(input_stats, output_stats, ia_cv, intensity, phases)

        return WorkloadProfile(
            total_requests=len(self._requests),
            duration_s=duration,
            mean_arrival_rate_rps=mean_rate,
            peak_arrival_rate_rps=peak_rate,
            input_token_stats=input_stats,
            output_token_stats=output_stats,
            latency_stats=latency_stats,
            intensity_class=intensity,
            category_distribution=cat_dist,
            phases=phases,
            burstiness_index=burstiness,
            inter_arrival_cv=ia_cv,
            synthetic_profile=synthetic,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        input_stats: DistributionStats,
        output_stats: DistributionStats,
        ia_cv: float,
        intensity: IntensityClass,
        phases: List[PhaseInfo],
    ) -> List[str]:
        recs: List[str] = []

        if input_stats.cv > 2.0:
            recs.append(
                f"High input length variance (CV={input_stats.cv:.1f}). "
                "Consider chunked prefill to avoid head-of-line blocking."
            )

        if ia_cv > 2.0:
            burst_count = sum(1 for p in phases if p.phase == WorkloadPhase.BURST)
            recs.append(
                f"Bursty arrival pattern (CV={ia_cv:.1f}, {burst_count} bursts). "
                "Size KV cache for peak, not mean, request rate."
            )

        if intensity == IntensityClass.COMPUTE_INTENSIVE:
            recs.append(
                "Workload is compute-intensive (long prompts). "
                "Prioritize prefill throughput and consider disaggregated prefill/decode."
            )
        elif intensity == IntensityClass.MEMORY_INTENSIVE:
            recs.append(
                "Workload is memory-intensive (long generation). "
                "Optimize KV cache and decode batch size."
            )

        return recs

    def reset(self) -> None:
        self._requests.clear()
