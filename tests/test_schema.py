"""Tests for configuration schema definitions."""

from benchmark_config.schema import (
    AttentionBackendConfig,
    BatchingConfig,
    BenchmarkRunConfig,
    ChunkedPrefillConfig,
    ComputePrecisionConfig,
    DisaggregatedConfig,
    DynamicReshardingConfig,
    KVCacheConfig,
    ParallelismConfig,
    SLOTargetConfig,
    WorkloadConfig,
)


class TestParallelismConfig:
    def test_defaults(self):
        cfg = ParallelismConfig()
        assert cfg.tensor_parallel_size == 1
        assert cfg.pipeline_parallel_size == 1
        assert cfg.data_parallel_replicas == 1
        assert cfg.expert_parallel is False
        assert cfg.ep_all2all_backend == "naive"
        assert cfg.distributed_executor_backend == "mp"

    def test_custom_values(self):
        cfg = ParallelismConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
        assert cfg.tensor_parallel_size == 4
        assert cfg.pipeline_parallel_size == 2

    def test_frozen(self):
        cfg = ParallelismConfig()
        try:
            cfg.tensor_parallel_size = 2  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestDisaggregatedConfig:
    def test_defaults(self):
        cfg = DisaggregatedConfig()
        assert cfg.prefill_tp == 1
        assert cfg.decode_tp == 1
        assert cfg.disaggregation_mode == "none"

    def test_valid_modes(self):
        assert "distserve" in DisaggregatedConfig.VALID_MODES
        assert "seesaw_resharding" in DisaggregatedConfig.VALID_MODES


class TestDynamicReshardingConfig:
    def test_defaults(self):
        cfg = DynamicReshardingConfig()
        assert cfg.cpu_kv_buffer_gb == 20
        assert cfg.kv_cache_layout == "NHD"

    def test_valid_layouts(self):
        assert "NHD" in DynamicReshardingConfig.VALID_KV_LAYOUTS
        assert "HND" in DynamicReshardingConfig.VALID_KV_LAYOUTS


class TestBatchingConfig:
    def test_defaults(self):
        cfg = BatchingConfig()
        assert cfg.max_num_batched_tokens == 2048
        assert cfg.enable_chunked_prefill is True

    def test_valid_seqs(self):
        assert 256 in BatchingConfig.VALID_MAX_SEQS


class TestChunkedPrefillConfig:
    def test_defaults(self):
        cfg = ChunkedPrefillConfig()
        assert cfg.chunk_size == 256
        assert cfg.batching_scheme == "decode_maximal"

    def test_valid_pd_ratios(self):
        assert 1 in ChunkedPrefillConfig.VALID_PD_RATIOS
        assert 200 in ChunkedPrefillConfig.VALID_PD_RATIOS


class TestKVCacheConfig:
    def test_defaults(self):
        cfg = KVCacheConfig()
        assert cfg.kv_cache_dtype == "auto"
        assert cfg.gpu_memory_utilization == 0.90
        assert cfg.block_size == 16

    def test_valid_dtypes(self):
        assert "fp8_e4m3" in KVCacheConfig.VALID_KV_DTYPES


class TestComputePrecisionConfig:
    def test_defaults(self):
        cfg = ComputePrecisionConfig()
        assert cfg.dtype == "bfloat16"
        assert cfg.quantization == "none"

    def test_valid_quantizations(self):
        assert "awq" in ComputePrecisionConfig.VALID_QUANTIZATIONS
        assert "gptq_marlin" in ComputePrecisionConfig.VALID_QUANTIZATIONS


class TestAttentionBackendConfig:
    def test_defaults(self):
        cfg = AttentionBackendConfig()
        assert cfg.attention_backend == "auto"
        assert cfg.flash_attn_version == 2

    def test_valid_backends(self):
        assert "flashinfer" in AttentionBackendConfig.VALID_BACKENDS


class TestWorkloadConfig:
    def test_defaults(self):
        cfg = WorkloadConfig()
        assert cfg.workload_dataset == "sharegpt"
        assert cfg.arrival_process == "offline"

    def test_valid_datasets(self):
        assert "longbench" in WorkloadConfig.VALID_DATASETS


class TestSLOTargetConfig:
    def test_defaults(self):
        cfg = SLOTargetConfig()
        assert cfg.ttft_slo_ms == 250
        assert cfg.tpot_slo_ms == 150

    def test_valid_scales(self):
        assert 0.5 in SLOTargetConfig.VALID_SCALE
        assert 3.0 in SLOTargetConfig.VALID_SCALE


class TestBenchmarkRunConfig:
    def test_defaults(self):
        cfg = BenchmarkRunConfig()
        assert cfg.benchmark_suite == "unified"
        assert cfg.model == "meta-llama/Llama-2-7b-hf"

    def test_to_dict(self):
        cfg = BenchmarkRunConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["benchmark_suite"] == "unified"
        assert "parallelism" in d
        assert d["parallelism"]["tensor_parallel_size"] == 1

    def test_frozen_sub_configs(self):
        cfg = BenchmarkRunConfig()
        assert cfg.parallelism.tensor_parallel_size == 1
