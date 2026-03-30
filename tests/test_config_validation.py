"""Tests for the config validator."""

import pytest

from benchmark.config.validation import (
    ConfigValidator,
    Severity,
    ValidationIssue,
    ValidationResult,
    _check_chunked_prefill,
    _check_disaggregation,
    _check_expert_parallel,
    _check_memory,
    _check_pipeline,
    _check_quantization,
    _check_speculative,
)
from benchmark.config.schema import BenchmarkConfig
from benchmark.config.sweep import ConfigPoint


class TestValidationResult:
    def test_is_valid_no_issues(self):
        result = ValidationResult(issues=[])
        assert result.is_valid

    def test_is_invalid_with_error(self):
        issue = ValidationIssue(Severity.ERROR, "rule", "msg")
        result = ValidationResult(issues=[issue])
        assert not result.is_valid

    def test_is_valid_with_only_warnings(self):
        issue = ValidationIssue(Severity.WARNING, "rule", "msg")
        result = ValidationResult(issues=[issue])
        assert result.is_valid

    def test_has_warnings(self):
        issue = ValidationIssue(Severity.WARNING, "rule", "msg")
        result = ValidationResult(issues=[issue])
        assert result.has_warnings

    def test_errors_filter(self):
        e = ValidationIssue(Severity.ERROR, "r1", "err")
        w = ValidationIssue(Severity.WARNING, "r2", "warn")
        result = ValidationResult(issues=[e, w])
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_summary_ok(self):
        result = ValidationResult(issues=[])
        assert "OK" in result.summary()

    def test_summary_with_issues(self):
        e = ValidationIssue(Severity.ERROR, "r1", "Something bad")
        result = ValidationResult(issues=[e])
        s = result.summary()
        assert "error" in s.lower()
        assert "Something bad" in s


class TestCheckSpeculative:
    def test_speculative_no_draft_model(self):
        cfg = ConfigPoint()
        cfg.speculative = True
        cfg.draft_model = None
        cfg.num_speculative_tokens = None
        issues = _check_speculative(cfg)
        kinds = [i.severity for i in issues]
        assert Severity.ERROR in kinds

    def test_speculative_with_draft_model(self):
        cfg = ConfigPoint()
        cfg.speculative = True
        cfg.draft_model = "gpt2"
        cfg.num_speculative_tokens = 5
        issues = _check_speculative(cfg)
        assert not any(i.severity == Severity.ERROR for i in issues)

    def test_not_speculative_no_issues(self):
        cfg = ConfigPoint()
        cfg.speculative = False
        issues = _check_speculative(cfg)
        assert len(issues) == 0


class TestCheckExpertParallel:
    def test_ep_no_disaggregation_warns(self):
        cfg = ConfigPoint()
        cfg.expert_parallel = True
        cfg.disaggregation_mode = "none"
        issues = _check_expert_parallel(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_ep_with_disaggregation_ok(self):
        cfg = ConfigPoint()
        cfg.expert_parallel = True
        cfg.disaggregation_mode = "distserve"
        cfg.ep_all2all_backend = "deepep_low_latency"
        issues = _check_expert_parallel(cfg)
        # Should have no warning about disaggregation
        disagg_warnings = [i for i in issues if "disaggregation" in i.rule]
        assert len(disagg_warnings) == 0

    def test_ep_naive_backend_warns(self):
        cfg = ConfigPoint()
        cfg.expert_parallel = True
        cfg.ep_all2all_backend = "naive"
        issues = _check_expert_parallel(cfg)
        naive_warns = [i for i in issues if "naive" in i.rule]
        assert len(naive_warns) > 0


class TestCheckDisaggregation:
    def test_distserve_missing_tp_pp(self):
        cfg = ConfigPoint()
        cfg.disaggregation_mode = "distserve"
        cfg.prefill_tp = None
        cfg.prefill_pp = None
        cfg.decode_tp = None
        cfg.decode_pp = None
        issues = _check_disaggregation(cfg)
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_distserve_complete(self):
        cfg = ConfigPoint(
            disaggregation_mode="distserve",
            prefill_tp=2, prefill_pp=1,
            decode_tp=2, decode_pp=1,
        )
        issues = _check_disaggregation(cfg)
        assert not any(i.severity == Severity.ERROR for i in issues)

    def test_seesaw_needs_cpu_kv_buffer(self):
        cfg = ConfigPoint(
            disaggregation_mode="seesaw_resharding",
            prefill_tp=2, prefill_pp=1,
            decode_tp=2, decode_pp=1,
        )
        cfg.cpu_kv_buffer_gb = None
        issues = _check_disaggregation(cfg)
        assert any("cpu_kv_buffer" in i.rule for i in issues)

    def test_none_disaggregation_no_issues(self):
        cfg = ConfigPoint(disaggregation_mode="none")
        issues = _check_disaggregation(cfg)
        assert len(issues) == 0


class TestCheckQuantization:
    def test_fp8_quant_float32_error(self):
        cfg = ConfigPoint()
        cfg.quantization = "fp8"
        cfg.dtype = "float32"
        issues = _check_quantization(cfg)
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_fp8_quant_bfloat16_ok(self):
        cfg = ConfigPoint()
        cfg.quantization = "fp8"
        cfg.dtype = "bfloat16"
        cfg.kv_dtype = "auto"
        cfg.attention_backend = "flash_attn"
        cfg.flash_attn_version = 3
        issues = _check_quantization(cfg)
        assert not any(i.severity == Severity.ERROR for i in issues)

    def test_fp8_kv_triton_warns(self):
        cfg = ConfigPoint()
        cfg.quantization = "none"
        cfg.kv_dtype = "fp8"
        cfg.attention_backend = "triton"
        issues = _check_quantization(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_flash_v3_non_bf16_warns(self):
        cfg = ConfigPoint()
        cfg.quantization = "none"
        cfg.kv_dtype = "auto"
        cfg.attention_backend = "flash_attn"
        cfg.flash_attn_version = 3
        cfg.dtype = "float16"
        issues = _check_quantization(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)


class TestCheckChunkedPrefill:
    def test_no_chunked_no_issues(self):
        cfg = ConfigPoint(chunked_prefill=False)
        issues = _check_chunked_prefill(cfg)
        assert len(issues) == 0

    def test_zero_chunk_size_error(self):
        cfg = ConfigPoint(chunked_prefill=True, chunk_size=0)
        issues = _check_chunked_prefill(cfg)
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_chunk_exceeds_max_model_len_error(self):
        cfg = ConfigPoint(chunked_prefill=True, chunk_size=512, max_model_len=256)
        issues = _check_chunked_prefill(cfg)
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_chunk_exceeds_max_batched_tokens_warns(self):
        cfg = ConfigPoint(
            chunked_prefill=True,
            chunk_size=4096,
            max_batched_tokens=2048,
            max_model_len=8192,
        )
        issues = _check_chunked_prefill(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_valid_chunked_prefill(self):
        cfg = ConfigPoint(
            chunked_prefill=True,
            chunk_size=256,
            max_batched_tokens=2048,
            max_model_len=8192,
        )
        issues = _check_chunked_prefill(cfg)
        assert len(issues) == 0


class TestCheckMemory:
    def test_swap_and_cpu_offload_warns(self):
        cfg = ConfigPoint()
        cfg.swap_space_gb = 8
        cfg.cpu_offload_gb = 8
        issues = _check_memory(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_high_gpu_mem_util_warns(self):
        cfg = ConfigPoint()
        cfg.swap_space_gb = 0
        cfg.cpu_offload_gb = 0
        cfg.gpu_mem_util = 0.99
        issues = _check_memory(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_normal_memory_no_issues(self):
        cfg = ConfigPoint()
        cfg.swap_space_gb = 0
        cfg.cpu_offload_gb = 0
        cfg.gpu_mem_util = 0.90
        issues = _check_memory(cfg)
        assert len(issues) == 0


class TestCheckPipeline:
    def test_pp_with_eager_warns(self):
        cfg = ConfigPoint()
        cfg.pp = 4
        cfg.enforce_eager = True
        issues = _check_pipeline(cfg)
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_pp1_eager_no_issue(self):
        cfg = ConfigPoint()
        cfg.pp = 1
        cfg.enforce_eager = True
        issues = _check_pipeline(cfg)
        assert len(issues) == 0


class TestConfigValidator:
    def test_default_config_valid(self):
        validator = ConfigValidator()
        cfg = ConfigPoint()
        result = validator.validate(cfg)
        assert result.is_valid

    def test_strict_mode_promotes_warnings_to_errors(self):
        validator = ConfigValidator(strict=True)
        cfg = ConfigPoint()
        cfg.pp = 4
        cfg.enforce_eager = True
        result = validator.validate(cfg)
        # In strict mode, the PP+eager warning becomes an error
        assert not result.is_valid

    def test_validate_batch(self):
        validator = ConfigValidator()
        configs = [ConfigPoint(tp=1), ConfigPoint(tp=2), ConfigPoint(tp=4)]
        results = validator.validate_batch(configs)
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_filter_valid(self):
        validator = ConfigValidator()
        valid = ConfigPoint()
        # Make an invalid config
        invalid = ConfigPoint()
        invalid.speculative = True
        invalid.draft_model = None
        invalid.num_speculative_tokens = None

        result = validator.filter_valid([valid, invalid])
        assert valid in result
        assert invalid not in result

    def test_benchmark_config_validates(self):
        validator = ConfigValidator()
        cfg = BenchmarkConfig()
        result = validator.validate(cfg)
        assert result.is_valid
