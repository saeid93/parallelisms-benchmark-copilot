"""Tests for configuration validation."""

import pytest

from benchmark_config.schema import (
    BatchingConfig,
    BenchmarkRunConfig,
    ChunkedPrefillConfig,
    DisaggregatedConfig,
    KVCacheConfig,
    ParallelismConfig,
    WorkloadConfig,
)
from benchmark_config.validation import ConfigValidationError, ConfigValidator


class TestFieldValidation:
    def test_valid_default_config(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig()
        errors = validator.validate(cfg)
        assert errors == []

    def test_invalid_tp(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            parallelism=ParallelismConfig(tensor_parallel_size=3),
        )
        errors = validator.validate(cfg)
        assert any("tensor_parallel_size" in e for e in errors)

    def test_invalid_suite(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(benchmark_suite="unknown")
        errors = validator.validate(cfg)
        assert any("benchmark_suite" in e for e in errors)

    def test_invalid_kv_dtype(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            kv_cache=KVCacheConfig(kv_cache_dtype="int4"),
        )
        errors = validator.validate(cfg)
        assert any("kv_cache_dtype" in e for e in errors)


class TestCrossFieldValidation:
    def test_tp_pp_exceeds_gpus(self):
        validator = ConfigValidator(max_gpus=4)
        cfg = BenchmarkRunConfig(
            parallelism=ParallelismConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=2,
            ),
        )
        errors = validator.validate(cfg)
        assert any("exceeds max_gpus" in e for e in errors)

    def test_tp_pp_dp_exceeds_gpus(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            parallelism=ParallelismConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=2,
                data_parallel_replicas=2,
            ),
        )
        errors = validator.validate(cfg)
        assert any("TP×PP×DP" in e for e in errors)

    def test_disaggregated_exceeds_gpus(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            benchmark_suite="distserve",
            disaggregated=DisaggregatedConfig(
                prefill_tp=4,
                prefill_pp=2,
                decode_tp=4,
                decode_pp=2,
                disaggregation_mode="distserve",
            ),
        )
        errors = validator.validate(cfg)
        assert any("Disaggregated" in e for e in errors)

    def test_chunk_size_gt_max_batched(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            benchmark_suite="sarathi",
            batching=BatchingConfig(max_num_batched_tokens=512),
            chunked_prefill=ChunkedPrefillConfig(chunk_size=512),
        )
        # chunk_size == max_num_batched_tokens is OK (not >)
        errors = validator.validate(cfg)
        assert not any("chunk_size" in e for e in errors)

        cfg2 = BenchmarkRunConfig(
            benchmark_suite="sarathi",
            batching=BatchingConfig(max_num_batched_tokens=512),
            chunked_prefill=ChunkedPrefillConfig(chunk_size=64),
        )
        errors2 = validator.validate(cfg2)
        assert not any("chunk_size" in e for e in errors2)

    def test_model_len_too_short(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            kv_cache=KVCacheConfig(max_model_len=2048),
            workload=WorkloadConfig(
                input_length_tokens=2048,
                output_length_tokens=512,
            ),
        )
        errors = validator.validate(cfg)
        assert any("max_model_len" in e for e in errors)

    def test_valid_distserve(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(
            benchmark_suite="distserve",
            disaggregated=DisaggregatedConfig(
                prefill_tp=2,
                prefill_pp=1,
                decode_tp=2,
                decode_pp=1,
                disaggregation_mode="distserve",
            ),
        )
        errors = validator.validate(cfg)
        assert errors == []


class TestValidateOrRaise:
    def test_raises_on_invalid(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig(benchmark_suite="invalid")
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_or_raise(cfg)
        assert len(exc_info.value.errors) >= 1

    def test_no_raise_on_valid(self):
        validator = ConfigValidator(max_gpus=8)
        cfg = BenchmarkRunConfig()
        validator.validate_or_raise(cfg)  # should not raise
