"""Tests for the structured result schema."""

import json

from benchmark_config.result_schema import (
    BenchmarkResult,
    LatencyMetrics,
    ResourceMetrics,
    ThroughputMetrics,
)
from benchmark_config.schema import BenchmarkRunConfig


class TestLatencyMetrics:
    def test_defaults(self):
        m = LatencyMetrics()
        assert m.mean_ttft_ms == 0.0
        assert m.p99_e2e_latency_ms == 0.0


class TestThroughputMetrics:
    def test_defaults(self):
        m = ThroughputMetrics()
        assert m.total_tokens_per_sec == 0.0
        assert m.goodput_requests_per_sec == 0.0


class TestResourceMetrics:
    def test_defaults(self):
        m = ResourceMetrics()
        assert m.peak_gpu_memory_gb == 0.0
        assert m.num_gpus_used == 0


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult()
        assert r.status == "pending"
        assert r.run_id  # non-empty UUID
        assert r.timestamp  # non-empty ISO string

    def test_with_config(self):
        cfg = BenchmarkRunConfig()
        r = BenchmarkResult(config=cfg, status="completed")
        assert r.config is not None
        assert r.status == "completed"

    def test_to_dict(self):
        cfg = BenchmarkRunConfig()
        r = BenchmarkResult(config=cfg)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "run_id" in d
        assert "config" in d
        assert "latency" in d
        # VALID_* constants should be stripped
        config_section = d["config"]
        for key in config_section:
            if isinstance(config_section[key], dict):
                for k in config_section[key]:
                    assert not k.startswith("VALID_"), (
                        f"VALID_ constant leaked: {key}.{k}"
                    )

    def test_to_json(self):
        cfg = BenchmarkRunConfig()
        r = BenchmarkResult(config=cfg)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "pending"
        assert "config" in parsed

    def test_with_metrics(self):
        r = BenchmarkResult(
            latency=LatencyMetrics(mean_ttft_ms=12.5, p99_tpot_ms=8.3),
            throughput=ThroughputMetrics(
                total_tokens_per_sec=1500.0,
                requests_per_sec=25.0,
            ),
            resources=ResourceMetrics(peak_gpu_memory_gb=38.5, num_gpus_used=4),
        )
        d = r.to_dict()
        assert d["latency"]["mean_ttft_ms"] == 12.5
        assert d["throughput"]["total_tokens_per_sec"] == 1500.0
        assert d["resources"]["num_gpus_used"] == 4

    def test_error_result(self):
        r = BenchmarkResult(status="failed", error_message="OOM")
        assert r.status == "failed"
        d = r.to_dict()
        assert d["error_message"] == "OOM"

    def test_metadata(self):
        r = BenchmarkResult(metadata={"gpu_type": "A100", "vllm_version": "0.5.0"})
        d = r.to_dict()
        assert d["metadata"]["gpu_type"] == "A100"
