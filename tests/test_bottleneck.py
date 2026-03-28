"""Tests for the bottleneck analyser (Stage 9)."""

import pytest

from benchmark.analysis.bottleneck import (
    BottleneckAnalyser,
    BottleneckKind,
    BottleneckResult,
    KV_CACHE_PRESSURE_THRESHOLD,
    NETWORK_BOUND_THRESHOLD,
    PIPELINE_BUBBLE_THRESHOLD,
)
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


def _make_metrics(**kwargs) -> BenchmarkMetrics:
    m = BenchmarkMetrics()
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def _make_cfg(**kwargs) -> ConfigPoint:
    cfg = ConfigPoint()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


class TestBottleneckResult:
    def test_add_bottleneck(self):
        cfg = ConfigPoint()
        m = BenchmarkMetrics()
        result = BottleneckResult(config=cfg, metrics=m)
        result.add(BottleneckKind.NETWORK_BOUND, 0.5, "test reason")
        assert len(result.bottlenecks) == 1
        assert result.bottlenecks[0]["kind"] == "network_bound"
        assert result.bottlenecks[0]["severity"] == 0.5

    def test_severity_clamped(self):
        result = BottleneckResult(config=ConfigPoint(), metrics=BenchmarkMetrics())
        result.add(BottleneckKind.KV_CACHE_PRESSURE, 2.0, "too high")
        result.add(BottleneckKind.PIPELINE_BUBBLE, -1.0, "negative")
        assert result.bottlenecks[0]["severity"] == 1.0
        assert result.bottlenecks[1]["severity"] == 0.0

    def test_primary_most_severe(self):
        result = BottleneckResult(config=ConfigPoint(), metrics=BenchmarkMetrics())
        result.add(BottleneckKind.NETWORK_BOUND, 0.3, "")
        result.add(BottleneckKind.PIPELINE_BUBBLE, 0.8, "")
        primary = result.primary
        assert primary is not None
        assert primary["kind"] == "pipeline_bubble"
        assert primary["severity"] == 0.8

    def test_primary_none_when_empty(self):
        result = BottleneckResult(config=ConfigPoint(), metrics=BenchmarkMetrics())
        assert result.primary is None

    def test_summary_string(self):
        result = BottleneckResult(config=ConfigPoint(), metrics=BenchmarkMetrics())
        result.add(BottleneckKind.MEMORY_BW_BOUND, 0.6, "HBM saturated")
        s = result.summary()
        assert "memory_bw_bound" in s
        assert "0.60" in s


class TestBottleneckAnalyser:
    def setup_method(self):
        self.analyser = BottleneckAnalyser()

    def test_no_bottleneck_clean_config(self):
        cfg = ConfigPoint(pp=1)
        m = _make_metrics(
            transmission_time_pct=5.0,
            pipeline_bubble_ratio=0.01,
            preemption_rate=0.01,
            prefill_phase_time_pct=50.0,
            decode_phase_time_pct=50.0,
            joint_slo_attainment_pct=98.0,
            avg_decode_batch_size=16.0,
        )
        result = self.analyser.analyse(cfg, m)
        assert len(result.bottlenecks) == 0

    def test_network_bound_detected(self):
        cfg = ConfigPoint(disaggregation_mode="distserve")
        m = _make_metrics(
            transmission_time_pct=30.0,  # > 20% threshold
        )
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.NETWORK_BOUND.value in kinds

    def test_pipeline_bubble_detected(self):
        cfg = _make_cfg(pp=4)
        m = _make_metrics(pipeline_bubble_ratio=0.15)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.PIPELINE_BUBBLE.value in kinds

    def test_pipeline_bubble_not_detected_pp1(self):
        cfg = _make_cfg(pp=1)
        m = _make_metrics(pipeline_bubble_ratio=0.3)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.PIPELINE_BUBBLE.value not in kinds

    def test_kv_cache_pressure(self):
        cfg = ConfigPoint()
        m = _make_metrics(preemption_rate=0.20)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.KV_CACHE_PRESSURE.value in kinds

    def test_prefill_dominated(self):
        cfg = ConfigPoint()
        m = _make_metrics(
            prefill_phase_time_pct=80.0,
            decode_phase_time_pct=20.0,
        )
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.PREFILL_DOMINATED.value in kinds

    def test_decode_dominated(self):
        cfg = ConfigPoint()
        m = _make_metrics(
            prefill_phase_time_pct=10.0,
            decode_phase_time_pct=90.0,
        )
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.DECODE_DOMINATED.value in kinds

    def test_memory_bw_bound(self):
        cfg = ConfigPoint()
        m = _make_metrics(
            decode_time_per_token_ms=8.0,
            prefill_phase_time_pct=5.0,
            decode_phase_time_pct=85.0,
        )
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.MEMORY_BW_BOUND.value in kinds

    def test_latency_bound(self):
        cfg = ConfigPoint()
        m = _make_metrics(avg_decode_batch_size=1.5)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.LATENCY_BOUND.value in kinds

    def test_latency_bound_not_flagged_large_batch(self):
        cfg = ConfigPoint()
        m = _make_metrics(avg_decode_batch_size=64.0)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.LATENCY_BOUND.value not in kinds

    def test_slo_constrained(self):
        cfg = ConfigPoint()
        m = _make_metrics(joint_slo_attainment_pct=60.0)
        result = self.analyser.analyse(cfg, m)
        kinds = [b["kind"] for b in result.bottlenecks]
        assert BottleneckKind.SLO_CONSTRAINED.value in kinds

    def test_analyse_batch(self):
        results = [
            (ConfigPoint(), _make_metrics(preemption_rate=0.3)),
            (ConfigPoint(), _make_metrics()),
        ]
        batch = self.analyser.analyse_batch(results)
        assert len(batch) == 2

    def test_top_bottlenecks(self):
        results = [
            (ConfigPoint(), _make_metrics(preemption_rate=0.01)),
            (ConfigPoint(), _make_metrics(preemption_rate=0.50)),
            (ConfigPoint(), _make_metrics(preemption_rate=0.80)),
        ]
        top = self.analyser.top_bottlenecks(results, top_n=2)
        assert len(top) <= 2
