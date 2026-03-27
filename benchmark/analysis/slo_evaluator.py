"""
Stage 5 — SLO attainment evaluator.

For each (config, dataset, slo_target) triple, sweeps request_rate_rps
and computes the joint SLO attainment curve.  Finds goodput: the maximum
request rate at which joint_slo_attainment_pct >= target.  Also sweeps
slo_scale to find the tightest viable SLO.

The goodput search uses binary search (bisection) over request_rate_rps
to find the goodput efficiently rather than exhaustive sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics

logger = logging.getLogger(__name__)

# Callable type: (request_rate_rps, ttft_slo_ms, tpot_slo_ms) -> BenchmarkMetrics
MetricsFn = Callable[[float, float, float], BenchmarkMetrics]


# ---------------------------------------------------------------------------
# Goodput binary search (Section 4)
# ---------------------------------------------------------------------------

def binary_search_goodput(
    metrics_fn: MetricsFn,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    attainment_target_pct: float,
    rate_lo: float = 0.1,
    rate_hi: float = 20.0,
    max_iterations: int = 20,
    tolerance: float = 0.05,
) -> Tuple[float, BenchmarkMetrics]:
    """Binary-search for the maximum feasible request rate (goodput).

    Bisects on request_rate_rps to find the highest rate at which
    joint_slo_attainment_pct >= attainment_target_pct.

    Args:
        metrics_fn: Callable that runs the benchmark at a given rate and
            returns BenchmarkMetrics.
        ttft_slo_ms: TTFT SLO threshold in milliseconds.
        tpot_slo_ms: TPOT SLO threshold in milliseconds.
        attainment_target_pct: Required joint SLO attainment percentage.
        rate_lo: Lower bound for the search interval (req/s).
        rate_hi: Upper bound for the search interval (req/s).
        max_iterations: Maximum number of bisection steps.
        tolerance: Stop when the interval is narrower than this (req/s).

    Returns:
        Tuple of (goodput_rps, BenchmarkMetrics at that rate).
    """
    best_rate = 0.0
    best_metrics = BenchmarkMetrics()

    for _ in range(max_iterations):
        if rate_hi - rate_lo < tolerance:
            break
        mid = (rate_lo + rate_hi) / 2.0
        metrics = metrics_fn(mid, ttft_slo_ms, tpot_slo_ms)
        if metrics.joint_slo_attainment_pct >= attainment_target_pct:
            best_rate = mid
            best_metrics = metrics
            rate_lo = mid
        else:
            rate_hi = mid

    best_metrics.goodput_rps = best_rate
    return best_rate, best_metrics


# ---------------------------------------------------------------------------
# SLO sweep
# ---------------------------------------------------------------------------

@dataclass
class SLOAttainmentResult:
    """Result of a single (rate, slo_scale) evaluation."""

    request_rate_rps: float
    slo_scale: float
    ttft_slo_ms: float
    tpot_slo_ms: float
    joint_slo_attainment_pct: float
    goodput_rps: float
    metrics: BenchmarkMetrics


class SLOEvaluator:
    """Evaluates SLO attainment curves across request rates and SLO scales.

    Args:
        metrics_fn: Callable (rate_rps, ttft_slo_ms, tpot_slo_ms) -> BenchmarkMetrics.
        attainment_target_pct: Minimum joint SLO attainment to count as
            feasible (default: 90 %).
        rate_sweep: Request rates to evaluate for the attainment curve.
        slo_scale_sweep: SLO scale multipliers to sweep.
    """

    DEFAULT_RATE_SWEEP = [0.5, 1.0, 1.6, 2.0, 3.0, 4.0, 5.6, 7.0, 10.0]
    DEFAULT_SLO_SCALE_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    def __init__(
        self,
        metrics_fn: MetricsFn,
        attainment_target_pct: float = 90.0,
        rate_sweep: Optional[List[float]] = None,
        slo_scale_sweep: Optional[List[float]] = None,
    ) -> None:
        self.metrics_fn = metrics_fn
        self.attainment_target_pct = attainment_target_pct
        self.rate_sweep = rate_sweep or self.DEFAULT_RATE_SWEEP
        self.slo_scale_sweep = slo_scale_sweep or self.DEFAULT_SLO_SCALE_SWEEP

    def evaluate_rate_sweep(
        self,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
    ) -> List[SLOAttainmentResult]:
        """Sweep request rates and return attainment at each rate.

        Args:
            ttft_slo_ms: TTFT SLO target in milliseconds.
            tpot_slo_ms: TPOT SLO target in milliseconds.

        Returns:
            List of SLOAttainmentResult, one per rate in rate_sweep.
        """
        results = []
        for rate in self.rate_sweep:
            metrics = self.metrics_fn(rate, ttft_slo_ms, tpot_slo_ms)
            results.append(
                SLOAttainmentResult(
                    request_rate_rps=rate,
                    slo_scale=1.0,
                    ttft_slo_ms=ttft_slo_ms,
                    tpot_slo_ms=tpot_slo_ms,
                    joint_slo_attainment_pct=metrics.joint_slo_attainment_pct,
                    goodput_rps=metrics.goodput_rps,
                    metrics=metrics,
                )
            )
        return results

    def find_goodput(
        self,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
    ) -> Tuple[float, BenchmarkMetrics]:
        """Binary-search for goodput at the given SLO targets.

        Args:
            ttft_slo_ms: TTFT SLO target in milliseconds.
            tpot_slo_ms: TPOT SLO target in milliseconds.

        Returns:
            Tuple of (goodput_rps, BenchmarkMetrics at that rate).
        """
        return binary_search_goodput(
            metrics_fn=self.metrics_fn,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            attainment_target_pct=self.attainment_target_pct,
        )

    def sweep_slo_scale(
        self,
        base_ttft_slo_ms: float,
        base_tpot_slo_ms: float,
    ) -> List[SLOAttainmentResult]:
        """Sweep slo_scale multipliers to find the tightest viable SLO.

        For each scale factor s, tightened SLOs are:
            ttft_slo = base_ttft_slo_ms * s
            tpot_slo = base_tpot_slo_ms * s

        Args:
            base_ttft_slo_ms: Base TTFT SLO in milliseconds.
            base_tpot_slo_ms: Base TPOT SLO in milliseconds.

        Returns:
            List of SLOAttainmentResult, one per scale value.
        """
        results = []
        for scale in self.slo_scale_sweep:
            ttft = base_ttft_slo_ms * scale
            tpot = base_tpot_slo_ms * scale
            goodput_rps, metrics = self.find_goodput(ttft, tpot)
            results.append(
                SLOAttainmentResult(
                    request_rate_rps=goodput_rps,
                    slo_scale=scale,
                    ttft_slo_ms=ttft,
                    tpot_slo_ms=tpot,
                    joint_slo_attainment_pct=metrics.joint_slo_attainment_pct,
                    goodput_rps=goodput_rps,
                    metrics=metrics,
                )
            )
        return results

    def tightest_viable_slo_scale(
        self,
        base_ttft_slo_ms: float,
        base_tpot_slo_ms: float,
        min_goodput_rps: float = 0.5,
    ) -> Optional[float]:
        """Find the smallest slo_scale at which goodput >= min_goodput_rps.

        Args:
            base_ttft_slo_ms: Base TTFT SLO in milliseconds.
            base_tpot_slo_ms: Base TPOT SLO in milliseconds.
            min_goodput_rps: Minimum acceptable goodput (req/s).

        Returns:
            The tightest (smallest) feasible slo_scale, or None if no scale
            in the sweep is feasible.
        """
        sweep_results = self.sweep_slo_scale(base_ttft_slo_ms, base_tpot_slo_ms)
        viable = [r for r in sweep_results if r.goodput_rps >= min_goodput_rps]
        if not viable:
            return None
        return min(r.slo_scale for r in viable)
