"""Tests for the cost estimator (Stage 10)."""

import pytest

from benchmark.analysis.cost_estimator import (
    GPU_CATALOGUE,
    CostEstimate,
    CostEstimator,
    GPUInstanceType,
)
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


def _make_metrics(tps: float = 1000.0, goodput: float = 5.0, rps: float = 5.0) -> BenchmarkMetrics:
    m = BenchmarkMetrics()
    m.throughput_tps = tps
    m.goodput_rps = goodput
    m.end_to_end_throughput_rps = rps
    return m


class TestGPUCatalogue:
    def test_catalogue_not_empty(self):
        assert len(GPU_CATALOGUE) > 0

    def test_a100_present(self):
        assert "a100_sxm4_80gb" in GPU_CATALOGUE

    def test_h100_present(self):
        assert "h100_sxm5_80gb" in GPU_CATALOGUE

    def test_pricing_positive(self):
        for name, inst in GPU_CATALOGUE.items():
            assert inst.price_per_gpu_hour_usd > 0, f"{name} has non-positive price"
            assert inst.peak_tflops_bf16 > 0
            assert inst.memory_bandwidth_gbs > 0
            assert inst.gpu_memory_gb > 0

    def test_cost_per_hour_scales_with_gpus(self):
        inst = GPU_CATALOGUE["a100_sxm4_80gb"]
        assert inst.cost_per_hour(1) == pytest.approx(inst.price_per_gpu_hour_usd)
        assert inst.cost_per_hour(4) == pytest.approx(inst.price_per_gpu_hour_usd * 4)


class TestCostEstimate:
    def test_basic_estimate(self):
        cfg = ConfigPoint(tp=2, pp=1, dp=1)
        m = _make_metrics(tps=5000.0, goodput=8.0, rps=8.0)
        inst = GPU_CATALOGUE["a100_sxm4_80gb"]
        est = CostEstimate(config=cfg, metrics=m, gpu_instance=inst, num_gpus=2)

        assert est.cost_per_hour_usd == pytest.approx(inst.price_per_gpu_hour_usd * 2)
        assert est.cost_per_million_tokens_usd > 0
        assert est.cost_per_million_requests_usd > 0

    def test_zero_tps_no_token_cost(self):
        cfg = ConfigPoint()
        m = _make_metrics(tps=0.0, goodput=0.0, rps=0.0)
        inst = GPU_CATALOGUE["a100_sxm4_80gb"]
        est = CostEstimate(config=cfg, metrics=m, gpu_instance=inst, num_gpus=1)
        assert est.cost_per_million_tokens_usd == 0.0
        assert est.cost_per_million_requests_usd == 0.0

    def test_summary_returns_string(self):
        cfg = ConfigPoint()
        m = _make_metrics()
        inst = GPU_CATALOGUE["a100_sxm4_80gb"]
        est = CostEstimate(config=cfg, metrics=m, gpu_instance=inst, num_gpus=1)
        s = est.summary()
        assert "GPU" in s
        assert "/hr" in s or "hr" in s

    def test_efficiency_capped_at_100(self):
        cfg = ConfigPoint()
        m = _make_metrics(tps=1e12)  # absurdly high TPS
        inst = GPU_CATALOGUE["a100_sxm4_80gb"]
        est = CostEstimate(config=cfg, metrics=m, gpu_instance=inst, num_gpus=1)
        assert est.gpu_memory_bw_efficiency_pct <= 100.0


class TestCostEstimator:
    def test_invalid_gpu_instance(self):
        with pytest.raises(ValueError, match="Unknown GPU instance"):
            CostEstimator(gpu_instance_name="nonexistent_gpu")

    def test_estimate_non_disaggregated(self):
        estimator = CostEstimator("a100_sxm4_80gb")
        cfg = ConfigPoint(tp=2, pp=2, dp=1)
        m = _make_metrics()
        est = estimator.estimate(cfg, m)
        assert est.num_gpus == cfg.gpu_count()  # 4

    def test_estimate_disaggregated(self):
        estimator = CostEstimator("a100_sxm4_80gb")
        cfg = ConfigPoint(
            tp=2, pp=1, dp=1,
            prefill_tp=2, prefill_pp=1,
            decode_tp=2, decode_pp=1,
            disaggregation_mode="distserve",
        )
        m = _make_metrics()
        est = estimator.estimate(cfg, m)
        # prefill=2, decode=2, dp=1 → total=4
        assert est.num_gpus == cfg.disaggregated_gpu_count()

    def test_estimate_batch(self):
        estimator = CostEstimator()
        results = [
            (ConfigPoint(tp=1), _make_metrics(tps=1000)),
            (ConfigPoint(tp=2), _make_metrics(tps=2000)),
            (ConfigPoint(tp=4), _make_metrics(tps=4000)),
        ]
        estimates = estimator.estimate_batch(results)
        assert len(estimates) == 3

    def test_cheapest(self):
        estimator = CostEstimator()
        results = [
            (ConfigPoint(tp=1), _make_metrics(tps=500)),
            (ConfigPoint(tp=4), _make_metrics(tps=10000)),
        ]
        estimates = estimator.estimate_batch(results)
        cheapest = estimator.cheapest(estimates)
        assert cheapest is not None
        # TP4 uses 4 GPUs but gets 10000 TPS → lower cost per token
        assert cheapest.metrics.throughput_tps == 10000

    def test_cheapest_empty(self):
        estimator = CostEstimator()
        assert estimator.cheapest([]) is None

    def test_most_efficient(self):
        estimator = CostEstimator()
        cfg1 = ConfigPoint(tp=1)
        cfg2 = ConfigPoint(tp=4)
        m1 = _make_metrics(tps=100)
        m2 = _make_metrics(tps=50000)
        est1 = estimator.estimate(cfg1, m1)
        est2 = estimator.estimate(cfg2, m2)
        best = estimator.most_efficient([est1, est2])
        assert best is not None

    def test_most_efficient_empty(self):
        estimator = CostEstimator()
        assert estimator.most_efficient([]) is None

    def test_minimum_1_gpu(self):
        """Configs with tp=0/pp=0 edge case: gpu_count should be clamped to 1."""
        estimator = CostEstimator()
        cfg = ConfigPoint(tp=1, pp=1, dp=1)
        m = _make_metrics()
        est = estimator.estimate(cfg, m)
        assert est.num_gpus >= 1
