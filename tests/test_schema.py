"""Tests for the BenchmarkRun schema (Section 2)."""

import uuid

import pytest

from benchmark.config.schema import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkRun,
    PerOpBreakdownMs,
)


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.tp == 1
        assert cfg.pp == 1
        assert cfg.dp == 1
        assert cfg.disaggregation_mode == "none"
        assert cfg.chunked_prefill is True
        assert cfg.gpu_mem_util == 0.90
        assert cfg.dtype == "bfloat16"
        assert cfg.attention_backend == "flash_attn"
        assert cfg.dataset == "sharegpt"

    def test_custom_values(self):
        cfg = BenchmarkConfig(
            tp=4,
            pp=2,
            dp=2,
            disaggregation_mode="distserve",
            prefill_tp=4,
            prefill_pp=1,
            decode_tp=2,
            decode_pp=2,
            chunked_prefill=False,
            kv_dtype="fp8",
            gpu_mem_util=0.85,
            dtype="float16",
            dataset="longbench",
            ttft_slo_ms=125.0,
        )
        assert cfg.tp == 4
        assert cfg.pp == 2
        assert cfg.disaggregation_mode == "distserve"
        assert cfg.prefill_tp == 4
        assert cfg.kv_dtype == "fp8"
        assert cfg.ttft_slo_ms == 125.0

    def test_invalid_disaggregation_mode(self):
        with pytest.raises(Exception):
            BenchmarkConfig(disaggregation_mode="invalid_mode")

    def test_invalid_dtype(self):
        with pytest.raises(Exception):
            BenchmarkConfig(dtype="int4")

    def test_invalid_attention_backend(self):
        with pytest.raises(Exception):
            BenchmarkConfig(attention_backend="xformers")


class TestBenchmarkMetrics:
    def test_defaults(self):
        m = BenchmarkMetrics()
        assert m.throughput_tps == 0.0
        assert m.goodput_rps == 0.0
        assert m.joint_slo_attainment_pct == 0.0
        assert m.speculative_acceptance_rate is None
        assert isinstance(m.per_op_breakdown_ms, PerOpBreakdownMs)

    def test_per_op_breakdown_defaults(self):
        breakdown = PerOpBreakdownMs()
        assert breakdown.preproj == 0.0
        assert breakdown.attn == 0.0
        assert breakdown.postproj == 0.0
        assert breakdown.ffn == 0.0

    def test_model_copy(self):
        m = BenchmarkMetrics(throughput_tps=1234.5)
        copy = m.model_copy()
        copy.throughput_tps = 9999.0
        assert m.throughput_tps == 1234.5


class TestBenchmarkRun:
    def test_run_id_is_uuid(self):
        run = BenchmarkRun()
        # Should not raise
        parsed = uuid.UUID(run.run_id)
        assert str(parsed) == run.run_id

    def test_default_suite(self):
        run = BenchmarkRun()
        assert run.benchmark_suite == "vllm_parallelism"

    def test_serialise_deserialise(self):
        run = BenchmarkRun(benchmark_suite="distserve")
        run.config.tp = 4
        run.metrics.goodput_rps = 7.5
        data = run.model_dump()
        restored = BenchmarkRun.model_validate(data)
        assert restored.run_id == run.run_id
        assert restored.benchmark_suite == "distserve"
        assert restored.config.tp == 4
        assert restored.metrics.goodput_rps == 7.5

    def test_json_round_trip(self):
        run = BenchmarkRun()
        run.config.dataset = "humaneval"
        json_str = run.model_dump_json()
        restored = BenchmarkRun.model_validate_json(json_str)
        assert restored.config.dataset == "humaneval"
