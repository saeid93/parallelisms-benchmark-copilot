"""Speculative decoding analyzer for LLM serving.

Inspired by:
* "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al. 2023)
* SpecInfer: Accelerating Generative LLM via Tree-Based Speculative Inference
* Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads
* Eagle: Speculative Sampling Requires Rethinking Feature Uncertainty
* "Speculative Decoding with Big Little Decoder" (Kim et al. 2023)

Analyses draft model acceptance rates, speculation depth optimization,
token tree verification overhead, and speedup vs overhead tradeoffs.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SpeculationMethod(str, Enum):
    """Speculative decoding methods."""

    STANDARD = "standard"  # single draft model
    TREE_BASED = "tree_based"  # SpecInfer-style token trees
    MEDUSA = "medusa"  # multiple decoding heads
    EAGLE = "eagle"  # feature-level speculation
    LOOKAHEAD = "lookahead"  # n-gram based
    SELF_SPECULATIVE = "self_speculative"  # layer skipping


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class SpeculationStep:
    """A single speculation + verification step."""

    step_index: int
    request_id: str = ""
    draft_tokens: int = 0
    accepted_tokens: int = 0
    draft_time_us: float = 0.0
    verify_time_us: float = 0.0
    baseline_time_us: float = 0.0  # what single-model decode would take
    tree_width: int = 1  # for tree-based methods
    tree_depth: int = 1

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens <= 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens

    @property
    def total_time_us(self) -> float:
        return self.draft_time_us + self.verify_time_us

    @property
    def speedup(self) -> float:
        if self.total_time_us <= 0:
            return 0.0
        return self.baseline_time_us / self.total_time_us

    @property
    def tokens_per_step(self) -> float:
        """Average tokens accepted per step (including the verified one)."""
        return self.accepted_tokens + 1.0  # +1 for the correction token


@dataclass
class DraftModelStats:
    """Statistics about draft model performance."""

    draft_model_name: str = ""
    draft_model_params_b: float = 0.0
    target_model_params_b: float = 0.0
    size_ratio: float = 0.0  # draft/target parameter ratio
    mean_acceptance_rate: float = 0.0
    acceptance_by_position: Dict[int, float] = field(default_factory=dict)
    mean_draft_latency_us: float = 0.0
    mean_verify_latency_us: float = 0.0
    draft_verify_ratio: float = 0.0


@dataclass
class OptimalDepthResult:
    """Optimal speculation depth analysis."""

    optimal_depth: int = 1
    acceptance_by_depth: Dict[int, float] = field(default_factory=dict)
    speedup_by_depth: Dict[int, float] = field(default_factory=dict)
    tokens_per_step_by_depth: Dict[int, float] = field(default_factory=dict)
    diminishing_returns_depth: int = 0  # depth beyond which marginal gain < 5%


@dataclass
class SpeculativeProfile:
    """Full speculative decoding analysis."""

    method: SpeculationMethod = SpeculationMethod.STANDARD
    total_steps: int = 0
    mean_acceptance_rate: float = 0.0
    mean_tokens_per_step: float = 0.0
    overall_speedup: float = 0.0
    draft_overhead_pct: float = 0.0
    verification_overhead_pct: float = 0.0
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    wasted_compute_pct: float = 0.0  # rejected draft tokens compute
    draft_model_stats: Optional[DraftModelStats] = None
    optimal_depth: Optional[OptimalDepthResult] = None
    acceptance_rate_trend: List[float] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Speculative Decoding Analysis ({self.method.value})",
            f"  Total steps          : {self.total_steps}",
            f"  Mean acceptance rate  : {self.mean_acceptance_rate:.1%}",
            f"  Mean tokens/step      : {self.mean_tokens_per_step:.2f}",
            f"  Overall speedup       : {self.overall_speedup:.2f}x",
            f"  Draft overhead        : {self.draft_overhead_pct:.1f}%",
            f"  Verification overhead : {self.verification_overhead_pct:.1f}%",
            f"  Wasted compute        : {self.wasted_compute_pct:.1f}%",
            f"  Total draft tokens    : {self.total_draft_tokens}",
            f"  Accepted tokens       : {self.total_accepted_tokens}",
        ]
        if self.draft_model_stats:
            dm = self.draft_model_stats
            lines.append(f"  Draft model           : {dm.draft_model_name}")
            lines.append(f"  Size ratio            : {dm.size_ratio:.2f}")
        if self.optimal_depth:
            od = self.optimal_depth
            lines.append(f"  Optimal depth         : {od.optimal_depth}")
            lines.append(f"  Diminishing at        : {od.diminishing_returns_depth}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SpeculativeAnalyzer:
    """Analyses speculative decoding performance.

    Usage::

        sa = SpeculativeAnalyzer(method=SpeculationMethod.STANDARD)
        sa.record_step(SpeculationStep(...))
        profile = sa.analyse()
        print(profile.summary())
    """

    def __init__(
        self,
        method: SpeculationMethod = SpeculationMethod.STANDARD,
        draft_model_name: str = "",
        draft_params_b: float = 0.0,
        target_params_b: float = 0.0,
    ) -> None:
        self._method = method
        self._draft_name = draft_model_name
        self._draft_params = draft_params_b
        self._target_params = target_params_b
        self._steps: List[SpeculationStep] = []

    def record_step(self, step: SpeculationStep) -> None:
        self._steps.append(step)

    def record_steps(self, steps: List[SpeculationStep]) -> None:
        self._steps.extend(steps)

    def _compute_optimal_depth(self) -> Optional[OptimalDepthResult]:
        """Find optimal speculation depth from recorded data."""
        if not self._steps:
            return None

        # Group by draft_tokens count (as proxy for speculation depth)
        by_depth: Dict[int, List[SpeculationStep]] = {}
        for s in self._steps:
            by_depth.setdefault(s.draft_tokens, []).append(s)

        acceptance_by_depth: Dict[int, float] = {}
        speedup_by_depth: Dict[int, float] = {}
        tps_by_depth: Dict[int, float] = {}

        for depth, steps in sorted(by_depth.items()):
            acceptance_by_depth[depth] = statistics.mean(
                [s.acceptance_rate for s in steps]
            )
            speedups = [s.speedup for s in steps if s.speedup > 0]
            speedup_by_depth[depth] = statistics.mean(speedups) if speedups else 0.0
            tps_by_depth[depth] = statistics.mean(
                [s.tokens_per_step for s in steps]
            )

        # Optimal: highest speedup
        optimal = max(speedup_by_depth.items(), key=lambda x: x[1])[0] if speedup_by_depth else 1

        # Diminishing returns
        dim_depth = optimal
        sorted_depths = sorted(speedup_by_depth.items())
        for i in range(1, len(sorted_depths)):
            prev_depth, prev_su = sorted_depths[i - 1]
            curr_depth, curr_su = sorted_depths[i]
            if prev_su > 0:
                marginal = (curr_su - prev_su) / prev_su
                if marginal < 0.05:
                    dim_depth = curr_depth
                    break

        return OptimalDepthResult(
            optimal_depth=optimal,
            acceptance_by_depth=acceptance_by_depth,
            speedup_by_depth=speedup_by_depth,
            tokens_per_step_by_depth=tps_by_depth,
            diminishing_returns_depth=dim_depth,
        )

    def analyse(self) -> SpeculativeProfile:
        if not self._steps:
            return SpeculativeProfile(method=self._method)

        acceptance_rates = [s.acceptance_rate for s in self._steps]
        tps = [s.tokens_per_step for s in self._steps]
        speedups = [s.speedup for s in self._steps if s.speedup > 0]

        total_draft = sum(s.draft_tokens for s in self._steps)
        total_accepted = sum(s.accepted_tokens for s in self._steps)
        total_draft_time = sum(s.draft_time_us for s in self._steps)
        total_verify_time = sum(s.verify_time_us for s in self._steps)
        total_time = total_draft_time + total_verify_time

        draft_overhead = 100.0 * total_draft_time / total_time if total_time > 0 else 0.0
        verify_overhead = 100.0 * total_verify_time / total_time if total_time > 0 else 0.0
        wasted = total_draft - total_accepted
        wasted_pct = 100.0 * wasted / total_draft if total_draft > 0 else 0.0

        # Draft model stats
        dm_stats = None
        if self._draft_name or self._draft_params > 0:
            dm_stats = DraftModelStats(
                draft_model_name=self._draft_name,
                draft_model_params_b=self._draft_params,
                target_model_params_b=self._target_params,
                size_ratio=(
                    self._draft_params / self._target_params
                    if self._target_params > 0
                    else 0.0
                ),
                mean_acceptance_rate=statistics.mean(acceptance_rates),
                mean_draft_latency_us=(
                    total_draft_time / len(self._steps)
                ),
                mean_verify_latency_us=(
                    total_verify_time / len(self._steps)
                ),
                draft_verify_ratio=(
                    total_draft_time / total_verify_time if total_verify_time > 0 else 0.0
                ),
            )

        optimal = self._compute_optimal_depth()

        # Acceptance rate trend (windowed average)
        trend: List[float] = []
        window = max(1, len(acceptance_rates) // 10)
        for i in range(0, len(acceptance_rates), window):
            chunk = acceptance_rates[i : i + window]
            trend.append(statistics.mean(chunk))

        recs = self._generate_recommendations(
            statistics.mean(acceptance_rates),
            statistics.mean(speedups) if speedups else 0.0,
            wasted_pct,
            optimal,
        )

        return SpeculativeProfile(
            method=self._method,
            total_steps=len(self._steps),
            mean_acceptance_rate=statistics.mean(acceptance_rates),
            mean_tokens_per_step=statistics.mean(tps),
            overall_speedup=statistics.mean(speedups) if speedups else 0.0,
            draft_overhead_pct=draft_overhead,
            verification_overhead_pct=verify_overhead,
            total_tokens_generated=total_accepted + len(self._steps),  # +correction tokens
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            wasted_compute_pct=wasted_pct,
            draft_model_stats=dm_stats,
            optimal_depth=optimal,
            acceptance_rate_trend=trend,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        mean_acceptance: float,
        mean_speedup: float,
        wasted_pct: float,
        optimal: Optional[OptimalDepthResult],
    ) -> List[str]:
        recs: List[str] = []

        if mean_acceptance < 0.5:
            recs.append(
                f"Low acceptance rate ({mean_acceptance:.1%}). "
                "Consider a better-aligned draft model or reducing speculation depth."
            )

        if mean_speedup < 1.0:
            recs.append(
                "Speculative decoding is slower than baseline! "
                "Draft model overhead exceeds benefit. Reduce depth or disable."
            )

        if wasted_pct > 50:
            recs.append(
                f"High wasted compute ({wasted_pct:.0f}% rejected tokens). "
                "Consider tree-based speculation for better acceptance coverage."
            )

        if optimal and optimal.optimal_depth > 1:
            recs.append(
                f"Optimal speculation depth is {optimal.optimal_depth}. "
                f"Diminishing returns begin at depth {optimal.diminishing_returns_depth}."
            )

        if self._method == SpeculationMethod.STANDARD and mean_acceptance < 0.7:
            recs.append(
                "Consider tree-based (SpecInfer) or Medusa-style speculation "
                "for better acceptance rates with branching candidates."
            )

        return recs

    def reset(self) -> None:
        self._steps.clear()
