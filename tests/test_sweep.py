"""Tests for sweep grid generation."""

from benchmark_config.schema import (
    BenchmarkRunConfig,
    ChunkedPrefillConfig,
    KVCacheConfig,
    ParallelismConfig,
    WorkloadConfig,
)
from benchmark_config.sweep import SweepGenerator


class TestSweepParallelism:
    def test_basic_tp_pp_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_parallelism(
            tp_sizes=[1, 2, 4],
            pp_sizes=[1, 2],
        ))
        assert len(configs) > 0
        # All should be valid (TP×PP ≤ 8)
        for cfg in configs:
            tp = cfg.parallelism.tensor_parallel_size
            pp = cfg.parallelism.pipeline_parallel_size
            assert tp * pp <= 8

    def test_filters_exceeding_gpus(self):
        gen = SweepGenerator(max_gpus=4)
        configs = list(gen.sweep_parallelism(
            tp_sizes=[1, 2, 4, 8],
            pp_sizes=[1, 2, 4, 8],
        ))
        for cfg in configs:
            tp = cfg.parallelism.tensor_parallel_size
            pp = cfg.parallelism.pipeline_parallel_size
            assert tp * pp <= 4

    def test_with_dp(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_parallelism(
            tp_sizes=[2],
            pp_sizes=[2],
            dp_replicas=[1, 2, 4],
        ))
        # TP=2, PP=2 → base 4 GPUs; DP=1 (4), DP=2 (8) fit; DP=4 (16) not
        assert len(configs) == 2
        dp_values = sorted(c.parallelism.data_parallel_replicas for c in configs)
        assert dp_values == [1, 2]


class TestSweepDisaggregated:
    def test_distserve_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_disaggregated(
            prefill_tp=[1, 2],
            prefill_pp=[1],
            decode_tp=[1, 2],
            decode_pp=[1],
        ))
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.benchmark_suite == "distserve"
            p = cfg.disaggregated.prefill_tp * cfg.disaggregated.prefill_pp
            d = cfg.disaggregated.decode_tp * cfg.disaggregated.decode_pp
            assert p + d <= 8


class TestSweepChunkedPrefill:
    def test_sarathi_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_chunked_prefill(
            chunk_sizes=[64, 128, 256],
            batching_schemes=["decode_maximal"],
            pd_ratios=[1, 10],
        ))
        assert len(configs) == 6  # 3 chunks × 1 scheme × 2 ratios
        for cfg in configs:
            assert cfg.benchmark_suite == "sarathi"


class TestSweepKVCache:
    def test_kv_cache_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_kv_cache(
            dtypes=["auto", "fp8"],
            block_sizes=[16, 32],
            gpu_mem_utils=[0.90],
        ))
        assert len(configs) == 4  # 2 dtypes × 2 blocks × 1 gpu_mem


class TestSweepWorkload:
    def test_workload_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_workload(
            datasets=["sharegpt"],
            rps_values=[1.0, 2.0],
            input_lengths=[512],
            output_lengths=[128],
        ))
        assert len(configs) == 2

    def test_filters_exceeding_model_len(self):
        base = BenchmarkRunConfig(
            kv_cache=KVCacheConfig(max_model_len=2048),
        )
        gen = SweepGenerator(max_gpus=8, base_config=base)
        configs = list(gen.sweep_workload(
            datasets=["sharegpt"],
            rps_values=[1.0],
            input_lengths=[2048],
            output_lengths=[64, 128, 512],
        ))
        # 2048 + 64 = 2112 > 2048 → all filtered out
        assert len(configs) == 0


class TestSweepSLO:
    def test_slo_sweep(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.sweep_slo(
            ttft_values=[250, 2500],
            tpot_values=[100, 150],
            scales=[1.0],
        ))
        assert len(configs) == 4
        for cfg in configs:
            assert cfg.benchmark_suite == "distserve"


class TestCustomSweep:
    def test_multi_dimension(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.custom_sweep({
            "parallelism": {
                "tensor_parallel_size": [1, 2],
            },
            "kv_cache": {
                "block_size": [16, 32],
            },
        }))
        assert len(configs) == 4  # 2 TP × 2 block_size

    def test_empty_dimensions(self):
        gen = SweepGenerator(max_gpus=8)
        configs = list(gen.custom_sweep({}))
        # No dimensions → single config (base)
        assert len(configs) == 1
