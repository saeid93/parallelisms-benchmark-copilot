"""Tests for the SLO attainment evaluator (Stage 5)."""

import pytest

from benchmark.analysis.slo_evaluator import (
    SLOEvaluator,
    binary_search_goodput,
)
from benchmark.config.schema import BenchmarkMetrics


def _make_metrics_fn(
    base_ttft_ms: float = 100.0,
    base_tpot_ms: float = 50.0,
    goodput_rps: float = 5.0,
):
    """Return a synthetic metrics function.

    The function returns good attainment when rate <= goodput_rps and
    zero attainment otherwise.
    """
    def fn(rate: float, ttft_slo: float, tpot_slo: float) -> BenchmarkMetrics:
        m = BenchmarkMetrics()
        if rate <= goodput_rps:
            m.ttft_p90_ms = base_ttft_ms
            m.tpot_p90_ms = base_tpot_ms
            m.joint_slo_attainment_pct = 95.0
        else:
            m.ttft_p90_ms = base_ttft_ms * (rate / goodput_rps)
            m.tpot_p90_ms = base_tpot_ms * (rate / goodput_rps)
            m.joint_slo_attainment_pct = 0.0
        m.end_to_end_throughput_rps = rate
        return m

    return fn


class TestBinarySearchGoodput:
    def test_finds_goodput(self):
        fn = _make_metrics_fn(goodput_rps=5.0)
        goodput, metrics = binary_search_goodput(
            metrics_fn=fn,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
            attainment_target_pct=90.0,
            rate_lo=0.1,
            rate_hi=20.0,
        )
        assert abs(goodput - 5.0) < 0.2
        assert metrics.joint_slo_attainment_pct >= 90.0

    def test_goodput_zero_when_impossible(self):
        # SLO target is 100% but fn never achieves it (max is 95%)
        def strict_fn(rate, ttft_slo, tpot_slo):
            m = BenchmarkMetrics()
            m.joint_slo_attainment_pct = 50.0
            return m

        goodput, _ = binary_search_goodput(
            metrics_fn=strict_fn,
            ttft_slo_ms=100.0,
            tpot_slo_ms=50.0,
            attainment_target_pct=99.0,
        )
        assert goodput == 0.0

    def test_tolerance_convergence(self):
        fn = _make_metrics_fn(goodput_rps=3.0)
        goodput, _ = binary_search_goodput(
            metrics_fn=fn,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
            attainment_target_pct=90.0,
            rate_lo=0.1,
            rate_hi=10.0,
            tolerance=0.01,
            max_iterations=50,
        )
        assert abs(goodput - 3.0) < 0.05


class TestSLOEvaluator:
    def test_evaluate_rate_sweep(self):
        fn = _make_metrics_fn(goodput_rps=4.0)
        evaluator = SLOEvaluator(
            metrics_fn=fn,
            attainment_target_pct=90.0,
            rate_sweep=[1.0, 2.0, 4.0, 5.0, 7.0],
        )
        results = evaluator.evaluate_rate_sweep(
            ttft_slo_ms=250.0, tpot_slo_ms=100.0
        )
        assert len(results) == 5
        # Below goodput: attainment >= 90
        below = [r for r in results if r.request_rate_rps <= 4.0]
        assert all(r.joint_slo_attainment_pct >= 90.0 for r in below)

    def test_find_goodput(self):
        fn = _make_metrics_fn(goodput_rps=6.0)
        evaluator = SLOEvaluator(metrics_fn=fn, attainment_target_pct=90.0)
        goodput, metrics = evaluator.find_goodput(250.0, 100.0)
        assert abs(goodput - 6.0) < 0.3

    def test_sweep_slo_scale(self):
        fn = _make_metrics_fn(goodput_rps=5.0)
        evaluator = SLOEvaluator(
            metrics_fn=fn,
            attainment_target_pct=90.0,
            slo_scale_sweep=[0.5, 1.0, 2.0],
        )
        results = evaluator.sweep_slo_scale(
            base_ttft_slo_ms=250.0,
            base_tpot_slo_ms=100.0,
        )
        assert len(results) == 3

    def test_tightest_viable_slo_scale(self):
        fn = _make_metrics_fn(goodput_rps=5.0)
        evaluator = SLOEvaluator(
            metrics_fn=fn,
            attainment_target_pct=90.0,
            slo_scale_sweep=[0.5, 1.0, 2.0],
        )
        scale = evaluator.tightest_viable_slo_scale(
            base_ttft_slo_ms=250.0,
            base_tpot_slo_ms=100.0,
            min_goodput_rps=1.0,
        )
        assert scale is not None
        assert scale <= 2.0

    def test_no_viable_scale(self):
        def zero_fn(rate, ttft_slo, tpot_slo):
            return BenchmarkMetrics()  # goodput_rps=0

        evaluator = SLOEvaluator(
            metrics_fn=zero_fn,
            attainment_target_pct=90.0,
        )
        scale = evaluator.tightest_viable_slo_scale(
            base_ttft_slo_ms=250.0,
            base_tpot_slo_ms=100.0,
            min_goodput_rps=1.0,
        )
        assert scale is None
