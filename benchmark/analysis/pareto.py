"""
Stage 6 — Pareto analyser.

Identifies the Pareto frontier across multiple objectives for each
benchmark suite:

- vllm_parallelism : throughput vs latency vs memory
- distserve        : goodput per GPU
- sarathi          : decode_speedup vs end_to_end_throughput
- seesaw           : prefill_time vs decode_time balance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Generic Pareto dominance
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if objective vector *a* weakly dominates *b*.

    A dominates B if A is no worse in every objective and strictly better
    in at least one objective.  All objectives are assumed to be
    **maximised** (negate costs before calling).

    Args:
        a: Objective vector for candidate A.
        b: Objective vector for candidate B.

    Returns:
        True if A dominates B, False otherwise.
    """
    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same length")
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


def pareto_frontier(
    points: List[Sequence[float]],
) -> List[int]:
    """Return indices of non-dominated (Pareto-optimal) points.

    Args:
        points: List of objective vectors (all objectives maximised).

    Returns:
        Sorted list of indices into *points* that form the Pareto frontier.
    """
    n = len(points)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(list(points[j]), list(points[i])):
                is_dominated[i] = True
                break
    return sorted(idx for idx, dominated in enumerate(is_dominated) if not dominated)


# ---------------------------------------------------------------------------
# Benchmark-specific objective extractors
# ---------------------------------------------------------------------------

def _vllm_parallelism_objectives(metrics: BenchmarkMetrics) -> List[float]:
    """Maximise throughput, minimise latency (negate), minimise memory (negate)."""
    return [
        metrics.throughput_tps,
        -metrics.e2e_latency_p90_ms,
        -metrics.gpu_mem_used_gb,
    ]


def _distserve_objectives(metrics: BenchmarkMetrics, num_gpus: int) -> List[float]:
    """Maximise goodput per GPU."""
    gpu_count = max(num_gpus, 1)
    return [metrics.goodput_rps / gpu_count]


def _sarathi_objectives(metrics: BenchmarkMetrics) -> List[float]:
    """Maximise decode speedup and end-to-end throughput."""
    return [
        metrics.decode_speedup_vs_baseline,
        metrics.end_to_end_throughput_rps,
    ]


def _seesaw_objectives(metrics: BenchmarkMetrics) -> List[float]:
    """Balance prefill and decode times (maximise minimum of the two ratios)."""
    total = metrics.prefill_phase_time_pct + metrics.decode_phase_time_pct
    if total <= 0:
        balance = 0.0
    else:
        prefill_ratio = metrics.prefill_phase_time_pct / total
        decode_ratio = metrics.decode_phase_time_pct / total
        # Perfect balance = 0.5/0.5; penalise imbalance
        balance = 1.0 - abs(prefill_ratio - decode_ratio)
    return [balance, metrics.end_to_end_throughput_rps]


# ---------------------------------------------------------------------------
# ParetoPoint result
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """A single Pareto-optimal benchmark result."""

    config: ConfigPoint
    metrics: BenchmarkMetrics
    objectives: List[float]
    suite: str


# ---------------------------------------------------------------------------
# ParetoAnalyser
# ---------------------------------------------------------------------------

class ParetoAnalyser:
    """Identifies Pareto-optimal configurations across all benchmark suites.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples from Stage 4.
    """

    def __init__(
        self, results: List[Tuple[ConfigPoint, BenchmarkMetrics]]
    ) -> None:
        self.results = results

    def _group_by_suite(
        self,
    ) -> Dict[str, List[Tuple[ConfigPoint, BenchmarkMetrics]]]:
        groups: Dict[str, List[Tuple[ConfigPoint, BenchmarkMetrics]]] = {}
        for cfg, metrics in self.results:
            suite = cfg.benchmark_suite
            groups.setdefault(suite, []).append((cfg, metrics))
        return groups

    def _frontier_for_suite(
        self,
        suite: str,
        items: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    ) -> List[ParetoPoint]:
        if not items:
            return []

        objectives: List[List[float]] = []
        for cfg, metrics in items:
            if suite == "vllm_parallelism":
                obj = _vllm_parallelism_objectives(metrics)
            elif suite == "distserve":
                num_gpus = cfg.disaggregated_gpu_count()
                obj = _distserve_objectives(metrics, num_gpus)
            elif suite == "sarathi":
                obj = _sarathi_objectives(metrics)
            elif suite == "seesaw":
                obj = _seesaw_objectives(metrics)
            else:
                obj = [metrics.throughput_tps]
            objectives.append(obj)

        frontier_indices = pareto_frontier(objectives)
        return [
            ParetoPoint(
                config=items[i][0],
                metrics=items[i][1],
                objectives=objectives[i],
                suite=suite,
            )
            for i in frontier_indices
        ]

    def analyse(self) -> Dict[str, List[ParetoPoint]]:
        """Compute Pareto frontiers for all suites.

        Returns:
            Dict mapping suite name to list of Pareto-optimal ParetoPoint
            instances.
        """
        groups = self._group_by_suite()
        return {
            suite: self._frontier_for_suite(suite, items)
            for suite, items in groups.items()
        }

    def best_for_suite(
        self,
        suite: str,
        objective_index: int = 0,
    ) -> Optional[ParetoPoint]:
        """Return the single best Pareto-optimal point for a suite.

        When there are multiple Pareto-optimal points, breaks ties by the
        objective at *objective_index* (higher is better).

        Args:
            suite: Benchmark suite name.
            objective_index: Index into the objective vector to use for
                tie-breaking.

        Returns:
            Best ParetoPoint, or None if no results for this suite.
        """
        frontiers = self.analyse()
        points = frontiers.get(suite, [])
        if not points:
            return None
        return max(
            points,
            key=lambda p: (
                p.objectives[objective_index]
                if objective_index < len(p.objectives)
                else 0.0
            ),
        )
