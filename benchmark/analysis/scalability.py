"""Scalability analyzer for distributed LLM serving.

Inspired by:
* Amdahl's Law & Gustafson's Law
* "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al. 2021, Megatron-LM)
* Strong/weak scaling analysis from HPC literature
* "Scaling Laws for Neural Language Models" (Kaplan et al. 2020)

Analyses weak scaling (constant work per GPU) and strong scaling
(fixed total work), fits Amdahl/Gustafson models, measures communication
overhead scaling, and predicts optimal GPU counts.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ScalingType(str, Enum):
    STRONG = "strong"  # fixed total work
    WEAK = "weak"  # fixed work per GPU


class ParallelismDimension(str, Enum):
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    SEQUENCE_PARALLEL = "sequence_parallel"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class ScalingDataPoint:
    """A single scaling measurement."""

    num_gpus: int
    throughput_tps: float
    latency_ms: float = 0.0
    gpu_utilization_pct: float = 0.0
    communication_time_pct: float = 0.0
    parallelism_config: Dict[str, int] = field(default_factory=dict)
    memory_per_gpu_gb: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class AmdahlFit:
    """Amdahl's law fit: Speedup = 1 / (s + (1-s)/N)."""

    serial_fraction: float = 0.0  # s
    max_theoretical_speedup: float = 0.0  # 1/s
    r_squared: float = 0.0

    def predicted_speedup(self, num_gpus: int) -> float:
        s = self.serial_fraction
        if s >= 1.0:
            return 1.0
        return 1.0 / (s + (1.0 - s) / num_gpus)


@dataclass
class GustafsonFit:
    """Gustafson's law fit: Scaled_speedup = N - s*(N-1)."""

    serial_fraction: float = 0.0

    def predicted_speedup(self, num_gpus: int) -> float:
        return num_gpus - self.serial_fraction * (num_gpus - 1)


@dataclass
class ScalingEfficiency:
    """Scaling efficiency metrics for a GPU count transition."""

    from_gpus: int
    to_gpus: int
    speedup: float = 0.0
    ideal_speedup: float = 0.0
    efficiency: float = 0.0  # speedup / ideal_speedup
    communication_overhead_pct: float = 0.0


@dataclass
class OptimalConfig:
    """Recommended optimal GPU configuration."""

    num_gpus: int
    throughput_tps: float
    cost_efficiency: float = 0.0  # tokens per dollar
    scaling_efficiency: float = 0.0
    parallelism_config: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ScalabilityProfile:
    """Full scalability analysis results."""

    scaling_type: ScalingType = ScalingType.STRONG
    data_points: List[ScalingDataPoint] = field(default_factory=list)
    efficiencies: List[ScalingEfficiency] = field(default_factory=list)
    amdahl_fit: Optional[AmdahlFit] = None
    gustafson_fit: Optional[GustafsonFit] = None
    optimal_config: Optional[OptimalConfig] = None
    comm_scaling_slope: float = 0.0  # how fast comm overhead grows
    diminishing_returns_gpu: int = 0  # GPU count beyond which eff < 50%
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Scalability Analysis ({self.scaling_type.value} scaling)",
            f"  Data points          : {len(self.data_points)}",
        ]
        if self.amdahl_fit:
            lines.append(f"  Amdahl serial frac.  : {self.amdahl_fit.serial_fraction:.4f}")
            lines.append(f"  Max theoretical S    : {self.amdahl_fit.max_theoretical_speedup:.1f}x")
            lines.append(f"  Amdahl R²            : {self.amdahl_fit.r_squared:.4f}")
        if self.gustafson_fit:
            lines.append(f"  Gustafson serial frac: {self.gustafson_fit.serial_fraction:.4f}")
        if self.efficiencies:
            for e in self.efficiencies:
                lines.append(
                    f"  {e.from_gpus}→{e.to_gpus} GPUs: "
                    f"speedup={e.speedup:.2f}x, eff={e.efficiency:.1%}, "
                    f"comm={e.communication_overhead_pct:.1f}%"
                )
        if self.diminishing_returns_gpu > 0:
            lines.append(f"  Diminishing returns  : >{self.diminishing_returns_gpu} GPUs")
        if self.optimal_config:
            oc = self.optimal_config
            lines.append(f"  Optimal: {oc.num_gpus} GPUs ({oc.reason})")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ScalabilityAnalyzer:
    """Analyses scaling efficiency across GPU counts.

    Usage::

        sa = ScalabilityAnalyzer()
        sa.add_point(ScalingDataPoint(num_gpus=1, throughput_tps=100))
        sa.add_point(ScalingDataPoint(num_gpus=2, throughput_tps=185))
        sa.add_point(ScalingDataPoint(num_gpus=4, throughput_tps=340))
        profile = sa.analyse()
        print(profile.summary())
    """

    def __init__(self, scaling_type: ScalingType = ScalingType.STRONG) -> None:
        self._scaling_type = scaling_type
        self._points: List[ScalingDataPoint] = []

    def add_point(self, point: ScalingDataPoint) -> None:
        self._points.append(point)

    def add_points(self, points: List[ScalingDataPoint]) -> None:
        self._points.extend(points)

    def _fit_amdahl(self, baseline_tps: float) -> Optional[AmdahlFit]:
        """Fit Amdahl's law to scaling data using least-squares."""
        if len(self._points) < 2 or baseline_tps <= 0:
            return None

        # Observed speedups
        observed = [(p.num_gpus, p.throughput_tps / baseline_tps) for p in self._points]

        # Grid search for serial fraction s ∈ (0, 1)
        best_s = 0.0
        best_sse = float("inf")
        for s_int in range(1, 100):
            s = s_int / 100.0
            sse = 0.0
            for n, obs_speedup in observed:
                pred = 1.0 / (s + (1.0 - s) / n)
                sse += (obs_speedup - pred) ** 2
            if sse < best_sse:
                best_sse = sse
                best_s = s

        # R²
        mean_speedup = statistics.mean([s for _, s in observed])
        ss_tot = sum((s - mean_speedup) ** 2 for _, s in observed)
        r_sq = 1.0 - best_sse / ss_tot if ss_tot > 0 else 0.0

        return AmdahlFit(
            serial_fraction=best_s,
            max_theoretical_speedup=1.0 / best_s if best_s > 0 else float("inf"),
            r_squared=max(0.0, r_sq),
        )

    def _fit_gustafson(self, baseline_tps: float) -> Optional[GustafsonFit]:
        """Fit Gustafson's law to weak scaling data."""
        if len(self._points) < 2 or baseline_tps <= 0:
            return None

        # s = mean((N - S(N)) / (N - 1)) for N > 1
        fractions: List[float] = []
        for p in self._points:
            if p.num_gpus > 1:
                speedup = p.throughput_tps / baseline_tps
                s = (p.num_gpus - speedup) / (p.num_gpus - 1)
                fractions.append(max(0.0, min(1.0, s)))

        if not fractions:
            return None

        return GustafsonFit(serial_fraction=statistics.mean(fractions))

    def analyse(self) -> ScalabilityProfile:
        """Run full scalability analysis."""
        if not self._points:
            return ScalabilityProfile(scaling_type=self._scaling_type)

        # Sort by GPU count
        sorted_pts = sorted(self._points, key=lambda p: p.num_gpus)

        # Baseline (smallest GPU count)
        baseline = sorted_pts[0]
        baseline_tps = baseline.throughput_tps

        # Compute efficiencies
        efficiencies: List[ScalingEfficiency] = []
        for i in range(1, len(sorted_pts)):
            prev = sorted_pts[i - 1]
            curr = sorted_pts[i]
            speedup = curr.throughput_tps / baseline_tps if baseline_tps > 0 else 0.0

            if self._scaling_type == ScalingType.STRONG:
                ideal = curr.num_gpus / baseline.num_gpus
            else:
                ideal = curr.num_gpus / baseline.num_gpus  # weak: same

            eff = speedup / ideal if ideal > 0 else 0.0

            efficiencies.append(
                ScalingEfficiency(
                    from_gpus=baseline.num_gpus,
                    to_gpus=curr.num_gpus,
                    speedup=speedup,
                    ideal_speedup=ideal,
                    efficiency=min(1.0, eff),
                    communication_overhead_pct=curr.communication_time_pct,
                )
            )

        # Fit models
        amdahl = self._fit_amdahl(baseline_tps)
        gustafson = self._fit_gustafson(baseline_tps) if self._scaling_type == ScalingType.WEAK else None

        # Diminishing returns
        dim_gpu = 0
        for e in efficiencies:
            if e.efficiency < 0.5:
                dim_gpu = e.to_gpus
                break

        # Comm scaling slope
        comm_pts = [(p.num_gpus, p.communication_time_pct) for p in sorted_pts if p.communication_time_pct > 0]
        comm_slope = 0.0
        if len(comm_pts) >= 2:
            x1, y1 = comm_pts[0]
            x2, y2 = comm_pts[-1]
            if x2 != x1:
                comm_slope = (y2 - y1) / (x2 - x1)

        # Optimal config (best cost efficiency)
        optimal = None
        if sorted_pts:
            best_cost_eff = 0.0
            for p in sorted_pts:
                if p.cost_per_hour > 0:
                    ce = p.throughput_tps / p.cost_per_hour
                    if ce > best_cost_eff:
                        best_cost_eff = ce
                        optimal = OptimalConfig(
                            num_gpus=p.num_gpus,
                            throughput_tps=p.throughput_tps,
                            cost_efficiency=ce,
                            parallelism_config=p.parallelism_config,
                            reason="best cost efficiency (tokens/$/hr)",
                        )

        recs = self._generate_recommendations(efficiencies, amdahl, comm_slope)

        return ScalabilityProfile(
            scaling_type=self._scaling_type,
            data_points=sorted_pts,
            efficiencies=efficiencies,
            amdahl_fit=amdahl,
            gustafson_fit=gustafson,
            optimal_config=optimal,
            comm_scaling_slope=comm_slope,
            diminishing_returns_gpu=dim_gpu,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        efficiencies: List[ScalingEfficiency],
        amdahl: Optional[AmdahlFit],
        comm_slope: float,
    ) -> List[str]:
        recs: List[str] = []

        if amdahl and amdahl.serial_fraction > 0.1:
            recs.append(
                f"Serial fraction is {amdahl.serial_fraction:.0%}. "
                "Reduce sequential bottlenecks (e.g. scheduling, tokenization) "
                "to improve parallel scaling."
            )

        if comm_slope > 2.0:
            recs.append(
                "Communication overhead grows steeply with GPU count. "
                "Consider pipeline parallelism or reducing TP degree."
            )

        # Check for super-linear degradation
        for e in efficiencies:
            if e.efficiency < 0.3:
                recs.append(
                    f"Efficiency at {e.to_gpus} GPUs is only {e.efficiency:.0%}. "
                    "This configuration is heavily communication-bound."
                )
                break

        return recs

    def predict_throughput(self, num_gpus: int) -> Optional[float]:
        """Predict throughput for a given GPU count using Amdahl's fit."""
        if not self._points:
            return None
        baseline = min(self._points, key=lambda p: p.num_gpus)
        amdahl = self._fit_amdahl(baseline.throughput_tps)
        if not amdahl:
            return None
        return baseline.throughput_tps * amdahl.predicted_speedup(num_gpus)

    def reset(self) -> None:
        self._points.clear()
