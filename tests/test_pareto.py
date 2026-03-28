"""Tests for the Pareto analyser (Stage 6)."""

import pytest

from benchmark.analysis.pareto import (
    ParetoAnalyser,
    ParetoPoint,
    dominates,
    pareto_frontier,
)
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


class TestDominates:
    def test_strict_dominance(self):
        assert dominates([3.0, 4.0], [2.0, 3.0])

    def test_no_dominance_equal(self):
        assert not dominates([2.0, 3.0], [2.0, 3.0])

    def test_no_dominance_worse_one_dim(self):
        assert not dominates([3.0, 2.0], [2.0, 3.0])

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            dominates([1.0, 2.0], [1.0])

    def test_single_dim(self):
        assert dominates([5.0], [3.0])
        assert not dominates([3.0], [5.0])


class TestParetoFrontier:
    def test_empty(self):
        assert pareto_frontier([]) == []

    def test_single_point(self):
        assert pareto_frontier([[1.0, 2.0]]) == [0]

    def test_all_dominated(self):
        # Point 2 dominates points 0 and 1
        points = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        frontier = pareto_frontier(points)
        assert frontier == [2]

    def test_true_pareto_frontier(self):
        # A tradeoff curve: no point dominates another
        points = [
            [3.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
        ]
        frontier = pareto_frontier(points)
        assert set(frontier) == {0, 1, 2}

    def test_mixed(self):
        points = [
            [5.0, 5.0],  # best — on frontier
            [3.0, 6.0],  # on frontier (better in dim 1)
            [1.0, 1.0],  # dominated
            [5.0, 4.0],  # dominated by point 0
        ]
        frontier = pareto_frontier(points)
        assert 0 in frontier
        assert 1 in frontier
        assert 2 not in frontier


class TestParetoAnalyser:
    def _make_result(self, suite, tps, goodput, decode_speedup, e2e_rps, gpu_mem):
        cfg = ConfigPoint(benchmark_suite=suite, tp=1, pp=1)
        cfg.disaggregation_mode = "none"
        if suite in ("distserve", "seesaw_resharding"):
            cfg.disaggregation_mode = suite
        metrics = BenchmarkMetrics()
        metrics.throughput_tps = tps
        metrics.goodput_rps = goodput
        metrics.decode_speedup_vs_baseline = decode_speedup
        metrics.end_to_end_throughput_rps = e2e_rps
        metrics.gpu_mem_used_gb = gpu_mem
        metrics.e2e_latency_p90_ms = 100.0
        metrics.prefill_phase_time_pct = 50.0
        metrics.decode_phase_time_pct = 50.0
        return cfg, metrics

    def test_analyse_vllm_parallelism(self):
        results = [
            self._make_result("vllm_parallelism", tps=1000, goodput=5, decode_speedup=1, e2e_rps=5, gpu_mem=40),
            self._make_result("vllm_parallelism", tps=2000, goodput=8, decode_speedup=1, e2e_rps=8, gpu_mem=60),
            self._make_result("vllm_parallelism", tps=1500, goodput=6, decode_speedup=1, e2e_rps=6, gpu_mem=30),
        ]
        analyser = ParetoAnalyser(results)
        frontiers = analyser.analyse()
        assert "vllm_parallelism" in frontiers
        # Point 1 (tps=2000, goodput=8, mem=60) is not dominated by point 2
        # because 2 has lower mem but lower tps/goodput too.
        assert len(frontiers["vllm_parallelism"]) >= 1

    def test_best_for_suite(self):
        results = [
            self._make_result("sarathi", tps=1000, goodput=5, decode_speedup=1.0, e2e_rps=5, gpu_mem=40),
            self._make_result("sarathi", tps=2000, goodput=8, decode_speedup=3.0, e2e_rps=8, gpu_mem=60),
        ]
        analyser = ParetoAnalyser(results)
        best = analyser.best_for_suite("sarathi", objective_index=0)
        assert best is not None
        assert best.metrics.decode_speedup_vs_baseline == 3.0

    def test_best_for_suite_no_results(self):
        analyser = ParetoAnalyser([])
        assert analyser.best_for_suite("sarathi") is None

    def test_multiple_suites(self):
        results = [
            self._make_result("vllm_parallelism", 1000, 5, 1, 5, 40),
            self._make_result("sarathi", 2000, 8, 2, 8, 60),
        ]
        analyser = ParetoAnalyser(results)
        frontiers = analyser.analyse()
        assert "vllm_parallelism" in frontiers
        assert "sarathi" in frontiers
