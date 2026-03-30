"""Tests for the GPU profiler (profiler stage)."""

import time

import pytest

from benchmark.profiler.gpu_profiler import (
    GPUProfiler,
    GPUSample,
    GPUStats,
    _aggregate,
)


def _make_sample(gpu_index: int = 0, **kwargs) -> GPUSample:
    s = GPUSample(gpu_index=gpu_index, timestamp_s=time.time())
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


class TestGPUSample:
    def test_defaults(self):
        s = GPUSample(gpu_index=0)
        assert s.gpu_utilization_pct == 0.0
        assert s.memory_used_mib == 0.0
        assert s.power_draw_w == 0.0

    def test_custom_values(self):
        s = GPUSample(
            gpu_index=1,
            gpu_utilization_pct=75.0,
            memory_used_mib=30_000.0,
            memory_total_mib=80_000.0,
        )
        assert s.gpu_utilization_pct == 75.0
        assert s.memory_used_mib == 30_000.0


class TestAggregate:
    def test_empty_returns_zero_stats(self):
        stats = _aggregate([], gpu_index=0)
        assert stats.gpu_index == 0
        assert stats.num_samples == 0
        assert stats.mean_gpu_util_pct == 0.0

    def test_single_sample(self):
        s = _make_sample(0, gpu_utilization_pct=50.0, power_draw_w=200.0)
        stats = _aggregate([s], gpu_index=0)
        assert stats.num_samples == 1
        assert stats.mean_gpu_util_pct == 50.0
        assert stats.max_gpu_util_pct == 50.0
        assert stats.mean_power_draw_w == 200.0

    def test_multiple_samples_mean(self):
        samples = [
            _make_sample(0, gpu_utilization_pct=20.0),
            _make_sample(0, gpu_utilization_pct=40.0),
            _make_sample(0, gpu_utilization_pct=60.0),
        ]
        stats = _aggregate(samples, gpu_index=0)
        assert stats.mean_gpu_util_pct == pytest.approx(40.0)

    def test_max_tracked(self):
        samples = [
            _make_sample(0, gpu_utilization_pct=30.0),
            _make_sample(0, gpu_utilization_pct=90.0),
            _make_sample(0, gpu_utilization_pct=10.0),
        ]
        stats = _aggregate(samples, gpu_index=0)
        assert stats.max_gpu_util_pct == 90.0


class TestGPUProfiler:
    def test_init_no_nvml(self):
        """GPUProfiler should initialise without error even without pynvml."""
        profiler = GPUProfiler(gpu_indices=[0, 1])
        assert profiler.num_gpus == 2

    def test_inject_sample(self):
        profiler = GPUProfiler(gpu_indices=[0])
        s = _make_sample(0, gpu_utilization_pct=70.0)
        profiler.inject_sample(s)
        assert profiler.total_samples() == 1

    def test_inject_multiple_samples(self):
        profiler = GPUProfiler(gpu_indices=[0])
        for util in [20.0, 40.0, 60.0, 80.0]:
            profiler.inject_sample(_make_sample(0, gpu_utilization_pct=util))
        assert profiler.total_samples() == 4

    def test_get_stats_for_gpu(self):
        profiler = GPUProfiler(gpu_indices=[0])
        for util in [30.0, 50.0, 70.0]:
            profiler.inject_sample(_make_sample(0, gpu_utilization_pct=util))
        stats = profiler.get_stats_for_gpu(0)
        assert stats is not None
        assert stats.mean_gpu_util_pct == pytest.approx(50.0)

    def test_get_stats_for_gpu_unknown(self):
        profiler = GPUProfiler(gpu_indices=[0])
        assert profiler.get_stats_for_gpu(99) is None

    def test_get_stats_all_gpus(self):
        profiler = GPUProfiler(gpu_indices=[0, 1])
        profiler.inject_sample(_make_sample(0, gpu_utilization_pct=40.0))
        profiler.inject_sample(_make_sample(1, gpu_utilization_pct=60.0))
        stats = profiler.get_stats()
        assert 0 in stats
        assert 1 in stats
        assert stats[0].mean_gpu_util_pct == 40.0
        assert stats[1].mean_gpu_util_pct == 60.0

    def test_all_samples(self):
        profiler = GPUProfiler(gpu_indices=[0])
        samples = [_make_sample(0, gpu_utilization_pct=float(i * 10)) for i in range(5)]
        for s in samples:
            profiler.inject_sample(s)
        returned = profiler.all_samples(0)
        assert len(returned) == 5
        assert returned[0].gpu_utilization_pct == 0.0
        assert returned[4].gpu_utilization_pct == 40.0

    def test_reset_clears_samples(self):
        profiler = GPUProfiler(gpu_indices=[0])
        profiler.inject_sample(_make_sample(0))
        profiler.inject_sample(_make_sample(0))
        profiler.reset()
        assert profiler.total_samples() == 0

    def test_sample_once_no_nvml(self):
        """sample_once should return zero samples when NVML is unavailable."""
        profiler = GPUProfiler(gpu_indices=[0])
        result = profiler.sample_once()
        assert 0 in result
        assert result[0].gpu_utilization_pct == 0.0  # zero because no NVML

    def test_background_sampling_lifecycle(self):
        """Start and stop background sampling without crashing."""
        profiler = GPUProfiler(gpu_indices=[0], sample_interval_s=0.05)
        profiler.start_background_sampling()
        time.sleep(0.15)  # let it sample a few times
        profiler.stop_background_sampling()
        # At least 1 sample should have been collected (zero-value without NVML)
        assert profiler.total_samples() >= 1

    def test_gpu_stats_summary_string(self):
        profiler = GPUProfiler(gpu_indices=[0])
        for util in [50.0, 60.0]:
            profiler.inject_sample(_make_sample(0, gpu_utilization_pct=util))
        stats = profiler.get_stats_for_gpu(0)
        assert stats is not None
        s = stats.summary()
        assert "GPU[0]" in s
        assert "util_mean" in s
