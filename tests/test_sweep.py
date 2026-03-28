"""Tests for the config space generator (Stage 1)."""

import pytest

from benchmark.config.sweep import (
    ConfigPoint,
    _is_feasible,
    generate_distserve_sweep,
    generate_full_sweep,
    generate_parallelism_sweep,
    generate_sarathi_sweep,
    generate_seesaw_sweep,
)


class TestIsFeasible:
    def test_feasible_default(self):
        cfg = ConfigPoint()
        assert _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_infeasible_too_many_gpus(self):
        cfg = ConfigPoint(tp=4, pp=4)  # 16 GPUs
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_infeasible_chunk_larger_than_max_model_len(self):
        cfg = ConfigPoint(chunked_prefill=True, chunk_size=512, max_model_len=256)
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_feasible_chunk_equal_to_max_model_len(self):
        cfg = ConfigPoint(
            chunked_prefill=True,
            chunk_size=256,
            max_model_len=256,
            max_seq_len_to_capture=256,
        )
        assert _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_infeasible_max_seq_len_capture_exceeds_max_model_len(self):
        cfg = ConfigPoint(max_seq_len_to_capture=8192, max_model_len=4096)
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_infeasible_oom(self):
        # 80 GB model on 1 GPU, but only 80*0.9=72 GB usable
        cfg = ConfigPoint(tp=1, pp=1, gpu_mem_util=0.90)
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=80.0)

    def test_feasible_oom_with_more_gpus(self):
        # 80 GB model across 8 TP shards = 10 GB/GPU, well within 80*0.9
        cfg = ConfigPoint(tp=8, pp=1, gpu_mem_util=0.90)
        assert _is_feasible(cfg, max_gpus=8, model_params_gb=80.0)

    def test_infeasible_pd_ratio_out_of_range(self):
        cfg = ConfigPoint(pd_ratio=500.0)
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_feasible_pd_ratio_in_range(self):
        cfg = ConfigPoint(pd_ratio=50.0)
        assert _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)

    def test_infeasible_disaggregated_gpu_count(self):
        cfg = ConfigPoint(
            prefill_tp=4, prefill_pp=2,
            decode_tp=4, decode_pp=2,
            disaggregation_mode="distserve",
        )
        # prefill: 8 GPUs + decode: 8 GPUs = 16 total, exceeds max_gpus=8
        assert not _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)


class TestConfigPointGpuCount:
    def test_simple(self):
        cfg = ConfigPoint(tp=2, pp=2, dp=2)
        assert cfg.gpu_count() == 8

    def test_disaggregated(self):
        cfg = ConfigPoint(
            prefill_tp=2, prefill_pp=1,
            decode_tp=2, decode_pp=2,
            dp=1,
            disaggregation_mode="distserve",
        )
        # prefill=2, decode=4 => total=6
        assert cfg.disaggregated_gpu_count() == 6


class TestParallelismSweep:
    def test_returns_configs(self):
        configs = list(
            generate_parallelism_sweep(
                tp_sizes=[1, 2],
                pp_sizes=[1],
                dp_replicas=[1],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.benchmark_suite == "vllm_parallelism"

    def test_all_feasible(self):
        configs = list(
            generate_parallelism_sweep(
                tp_sizes=[1, 2, 4],
                pp_sizes=[1, 2],
                dp_replicas=[1],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        for cfg in configs:
            assert _is_feasible(cfg, max_gpus=8, model_params_gb=7.0)


class TestDistserveSweep:
    def test_returns_distserve_suite(self):
        configs = list(
            generate_distserve_sweep(
                prefill_tp_sizes=[1, 2],
                prefill_pp_sizes=[1],
                decode_tp_sizes=[1, 2],
                decode_pp_sizes=[1],
                ttft_slos=[250],
                tpot_slos=[100],
                slo_scales=[1.0],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.benchmark_suite == "distserve"
            assert cfg.disaggregation_mode == "distserve"


class TestSarathiSweep:
    def test_returns_sarathi_suite(self):
        configs = list(
            generate_sarathi_sweep(
                chunk_sizes=[256],
                batching_schemes=["decode_maximal"],
                pd_ratios=[10],
                tp_sizes=[1],
                pp_sizes=[1],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        assert len(configs) == 1
        assert configs[0].benchmark_suite == "sarathi"
        assert configs[0].chunked_prefill is True

    def test_chunk_size_pruning(self):
        # chunk_size=512 > max_model_len=256 should be pruned
        configs = list(
            generate_sarathi_sweep(
                chunk_sizes=[512],
                batching_schemes=["decode_maximal"],
                pd_ratios=[10],
                tp_sizes=[1],
                pp_sizes=[1],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        # Default max_model_len=8192, so chunk_size=512 is fine
        assert len(configs) == 1


class TestSeesawSweep:
    def test_returns_seesaw_suite(self):
        configs = list(
            generate_seesaw_sweep(
                cpu_kv_buffer_gb_values=[40],
                kv_cache_layouts=["HND"],
                transition_policies=["prefill_prioritizing"],
                tp_sizes=[1, 2],
                pp_sizes=[1],
                datasets=["sharegpt"],
                max_gpus=8,
                model_params_gb=7.0,
            )
        )
        assert len(configs) == 2
        for cfg in configs:
            assert cfg.benchmark_suite == "seesaw"
            assert cfg.disaggregation_mode == "seesaw_resharding"
            assert cfg.resharding_pair is not None


class TestGenerateFullSweep:
    def test_all_suites(self):
        configs = generate_full_sweep(
            max_gpus=4,
            model_params_gb=7.0,
        )
        suites = {cfg.benchmark_suite for cfg in configs}
        assert "vllm_parallelism" in suites
        assert "distserve" in suites
        assert "sarathi" in suites
        assert "seesaw" in suites

    def test_single_suite_filter(self):
        configs = generate_full_sweep(
            max_gpus=4,
            model_params_gb=7.0,
            suites=["sarathi"],
        )
        for cfg in configs:
            assert cfg.benchmark_suite == "sarathi"
