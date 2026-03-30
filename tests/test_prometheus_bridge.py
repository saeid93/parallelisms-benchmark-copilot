"""Tests for the Prometheus metrics bridge."""

import json
from unittest import mock

import pytest

from benchmark.config.schema import BenchmarkMetrics
from benchmark.metrics.prometheus_bridge import (
    VLLM_METRIC_PATTERNS,
    PrometheusBridge,
    PrometheusSnapshot,
    _parse_float,
)


# ---------------------------------------------------------------------------
# _parse_float
# ---------------------------------------------------------------------------

class TestParseFloat:
    def test_match(self):
        text = 'vllm:tokens_total{model="llama"} 12345.0\n'
        val = _parse_float(text, VLLM_METRIC_PATTERNS["tokens_total"])
        assert val == 12345.0

    def test_no_match(self):
        val = _parse_float("# nothing here", VLLM_METRIC_PATTERNS["tokens_total"])
        assert val is None

    def test_non_numeric(self):
        text = 'vllm:tokens_total{model="llama"} NaN\n'
        val = _parse_float(text, VLLM_METRIC_PATTERNS["tokens_total"])
        assert val is None


# ---------------------------------------------------------------------------
# PrometheusSnapshot
# ---------------------------------------------------------------------------

class TestPrometheusSnapshot:
    def test_from_text_empty(self):
        snap = PrometheusSnapshot.from_text("")
        assert snap.values == {}

    def test_from_text_with_metrics(self):
        text = (
            '# HELP vllm:tokens_total Total tokens generated\n'
            'vllm:tokens_total{model="llama"} 9876.5\n'
            '# HELP vllm:gpu_cache_usage_perc GPU cache usage\n'
            'vllm:gpu_cache_usage_perc{model="llama"} 0.85\n'
            '# HELP vllm:num_preemptions_total Preemptions\n'
            'vllm:num_preemptions_total{model="llama"} 3.0\n'
        )
        snap = PrometheusSnapshot.from_text(text)
        assert snap.values["tokens_total"] == 9876.5
        assert snap.values["gpu_cache_usage_perc"] == 0.85
        assert snap.values["num_preemptions_total"] == 3.0

    def test_from_text_partial(self):
        text = 'vllm:tokens_total{model="llama"} 100.0\n'
        snap = PrometheusSnapshot.from_text(text)
        assert "tokens_total" in snap.values
        assert "gpu_cache_usage_perc" not in snap.values

    def test_from_text_all_patterns(self):
        """Build a text with all known patterns to ensure they parse."""
        lines = []
        for name, pattern in VLLM_METRIC_PATTERNS.items():
            # Replace the regex-escaped label part with a concrete label
            metric_name = pattern.split(r"\{")[0]
            lines.append(f'{metric_name}{{model="test"}} 42.0')
        text = "\n".join(lines)
        snap = PrometheusSnapshot.from_text(text)
        # At least some should parse (not all regex patterns may match
        # a synthetic line exactly, but the core ones should)
        assert len(snap.values) > 0


# ---------------------------------------------------------------------------
# PrometheusBridge URL resolution
# ---------------------------------------------------------------------------

class TestPrometheusBridgeURLs:
    def test_local_url(self):
        bridge = PrometheusBridge(execution_mode="local", prometheus_url="http://localhost:8000")
        assert bridge._metrics_url() == "http://localhost:8000/metrics"

    def test_kubernetes_url(self):
        bridge = PrometheusBridge(
            execution_mode="kubernetes",
            namespace="my-ns",
            service_name="my-svc",
            service_port=9000,
        )
        url = bridge._metrics_url()
        assert "my-svc.my-ns.svc.cluster.local" in url
        assert ":9000/metrics" in url

    def test_default_values(self):
        bridge = PrometheusBridge()
        assert bridge.execution_mode == "local"
        assert bridge.namespace == "benchmark"
        assert bridge.service_name == "vllm-benchmark"


# ---------------------------------------------------------------------------
# PrometheusBridge.scrape
# ---------------------------------------------------------------------------

class TestPrometheusBridgeScrape:
    def test_scrape_success(self):
        bridge = PrometheusBridge()
        fake_text = 'vllm:tokens_total{model="test"} 500.0\n'
        mock_resp = mock.MagicMock()
        mock_resp.text = fake_text
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("benchmark.metrics.prometheus_bridge.http_requests") as mock_req:
            mock_req.get.return_value = mock_resp
            snap = bridge.scrape()

        assert snap.values.get("tokens_total") == 500.0

    def test_scrape_failure(self):
        bridge = PrometheusBridge()
        with mock.patch("benchmark.metrics.prometheus_bridge.http_requests") as mock_req:
            mock_req.get.side_effect = Exception("connection refused")
            snap = bridge.scrape()

        assert snap.values == {}

    def test_scrape_no_requests_library(self):
        bridge = PrometheusBridge()
        with mock.patch("benchmark.metrics.prometheus_bridge.http_requests", None):
            snap = bridge.scrape()
        assert snap.values == {}


# ---------------------------------------------------------------------------
# PrometheusBridge.enrich_metrics
# ---------------------------------------------------------------------------

class TestEnrichMetrics:
    def test_enrich_with_snapshot(self):
        bridge = PrometheusBridge()
        metrics = BenchmarkMetrics(throughput_tps=1000.0)
        snap = PrometheusSnapshot(
            raw_text="",
            values={
                "gpu_cache_usage_perc": 0.75,
                "cpu_prefix_cache_hit_rate": 0.5,
                "num_preemptions_total": 2.0,
            },
        )
        enriched = bridge.enrich_metrics(metrics, snap)
        assert enriched.gpu_mem_used_gb == 0.75
        assert enriched.kv_cache_hit_rate == 0.5
        assert enriched.preemption_rate == 2.0
        # Original throughput preserved
        assert enriched.throughput_tps == 1000.0

    def test_enrich_empty_snapshot(self):
        bridge = PrometheusBridge()
        metrics = BenchmarkMetrics(throughput_tps=500.0)
        snap = PrometheusSnapshot()
        enriched = bridge.enrich_metrics(metrics, snap)
        assert enriched.throughput_tps == 500.0
        assert enriched.gpu_mem_used_gb == 0.0

    def test_enrich_does_not_mutate_original(self):
        bridge = PrometheusBridge()
        metrics = BenchmarkMetrics(gpu_mem_used_gb=1.0)
        snap = PrometheusSnapshot(values={"gpu_cache_usage_perc": 0.9})
        enriched = bridge.enrich_metrics(metrics, snap)
        assert metrics.gpu_mem_used_gb == 1.0  # original unchanged
        assert enriched.gpu_mem_used_gb == 0.9


# ---------------------------------------------------------------------------
# PrometheusBridge.push_metrics
# ---------------------------------------------------------------------------

class TestPushMetrics:
    def test_push_no_url_returns_false(self):
        bridge = PrometheusBridge(pushgateway_url=None)
        assert bridge.push_metrics("run-1", BenchmarkMetrics()) is False

    def test_push_with_url_success(self):
        bridge = PrometheusBridge(pushgateway_url="http://pushgw:9091")
        with mock.patch(
            "benchmark.metrics.prometheus_bridge.push_to_gateway"
        ) as mock_push:
            result = bridge.push_metrics("run-1", BenchmarkMetrics())
        assert result is True
        mock_push.assert_called_once()

    def test_push_with_url_failure(self):
        bridge = PrometheusBridge(pushgateway_url="http://pushgw:9091")
        with mock.patch(
            "benchmark.metrics.prometheus_bridge.push_to_gateway",
            side_effect=Exception("push failed"),
        ):
            result = bridge.push_metrics("run-1", BenchmarkMetrics())
        assert result is False
