"""Tests for the regression detector (Stage 11)."""

import json
import os
import tempfile

import pytest

from benchmark.analysis.regression import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    _METRIC_SPEC,
)
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


def _make_metrics(**kwargs) -> BenchmarkMetrics:
    m = BenchmarkMetrics()
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def _cfg(suite: str = "vllm_parallelism", tp: int = 1, pp: int = 1) -> ConfigPoint:
    return ConfigPoint(benchmark_suite=suite, tp=tp, pp=pp)


class TestMetricRegression:
    def test_regression_higher_is_better(self):
        mr = MetricRegression(
            metric_name="throughput_tps",
            baseline_value=1000.0,
            current_value=900.0,
            tolerance_pct=5.0,
            higher_is_better=True,
        )
        # 900/1000 = -10%, which exceeds 5% tolerance
        assert mr.is_regression
        assert abs(mr.delta_pct - (-10.0)) < 0.01

    def test_no_regression_within_tolerance(self):
        mr = MetricRegression(
            metric_name="throughput_tps",
            baseline_value=1000.0,
            current_value=960.0,
            tolerance_pct=5.0,
            higher_is_better=True,
        )
        # -4%, within 5% tolerance
        assert not mr.is_regression

    def test_regression_lower_is_better(self):
        mr = MetricRegression(
            metric_name="ttft_p90_ms",
            baseline_value=100.0,
            current_value=115.0,
            tolerance_pct=10.0,
            higher_is_better=False,
        )
        # +15% increase in latency > 10% tolerance
        assert mr.is_regression

    def test_no_regression_lower_is_better_within_tol(self):
        mr = MetricRegression(
            metric_name="ttft_p90_ms",
            baseline_value=100.0,
            current_value=108.0,
            tolerance_pct=10.0,
            higher_is_better=False,
        )
        assert not mr.is_regression

    def test_zero_baseline_no_division_error(self):
        mr = MetricRegression(
            metric_name="goodput_rps",
            baseline_value=0.0,
            current_value=5.0,
            tolerance_pct=5.0,
            higher_is_better=True,
        )
        assert mr.delta_pct == 0.0

    def test_str_contains_metric_name(self):
        mr = MetricRegression(
            metric_name="throughput_tps",
            baseline_value=1000.0,
            current_value=800.0,
            tolerance_pct=5.0,
            higher_is_better=True,
        )
        s = str(mr)
        assert "throughput_tps" in s
        assert "REGRESSION" in s


class TestRegressionReport:
    def test_has_regression(self):
        mr = MetricRegression("tps", 1000.0, 800.0, 5.0, True)
        report = RegressionReport(config_key="k", regressions=[mr])
        assert report.has_regression

    def test_no_regression_when_empty(self):
        report = RegressionReport(config_key="k")
        assert not report.has_regression

    def test_regression_score(self):
        mr = MetricRegression("tps", 1000.0, 800.0, 5.0, True)
        report = RegressionReport(config_key="k", regressions=[mr])
        assert report.regression_score > 0.0

    def test_summary_string(self):
        mr = MetricRegression("tps", 1000.0, 800.0, 5.0, True)
        report = RegressionReport(config_key="test_key", regressions=[mr])
        s = report.summary()
        assert "test_key" in s


class TestRegressionDetector:
    def test_config_key_stable(self):
        cfg = ConfigPoint(benchmark_suite="sarathi", tp=4, pp=2)
        key1 = RegressionDetector.config_key(cfg)
        key2 = RegressionDetector.config_key(cfg)
        assert key1 == key2

    def test_config_key_different_configs(self):
        cfg1 = ConfigPoint(tp=1, pp=1)
        cfg2 = ConfigPoint(tp=4, pp=2)
        assert RegressionDetector.config_key(cfg1) != RegressionDetector.config_key(cfg2)

    def test_save_and_load_baseline(self):
        results = [
            (_cfg(), _make_metrics(throughput_tps=1000.0, goodput_rps=5.0)),
            (_cfg(tp=2), _make_metrics(throughput_tps=2000.0, goodput_rps=8.0)),
        ]
        detector = RegressionDetector()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            detector.save_baseline(results, path)
            baseline = detector.load_baseline(path)
            assert len(baseline) == len(results)
            for cfg, m in results:
                key = RegressionDetector.config_key(cfg)
                assert key in baseline
                assert baseline[key].throughput_tps == m.throughput_tps
        finally:
            os.unlink(path)

    def test_compare_no_regression(self):
        cfg = _cfg()
        key = RegressionDetector.config_key(cfg)
        baseline = {key: _make_metrics(throughput_tps=1000.0)}
        current = [
            (cfg, _make_metrics(throughput_tps=1000.0))
        ]
        detector = RegressionDetector()
        reports = detector.compare(current, baseline)
        assert len(reports) == 1
        assert not reports[0].has_regression

    def test_compare_detects_throughput_regression(self):
        cfg = _cfg()
        key = RegressionDetector.config_key(cfg)
        baseline = {key: _make_metrics(throughput_tps=1000.0)}
        current = [(cfg, _make_metrics(throughput_tps=700.0))]
        detector = RegressionDetector()
        reports = detector.compare(current, baseline)
        assert reports[0].has_regression
        tps_reg = [r for r in reports[0].regressions if r.metric_name == "throughput_tps"]
        assert len(tps_reg) == 1

    def test_compare_no_baseline_entry_skipped(self):
        cfg = _cfg(tp=99)  # key won't match baseline
        baseline = {}
        current = [(cfg, _make_metrics(throughput_tps=1000.0))]
        detector = RegressionDetector()
        reports = detector.compare(current, baseline)
        assert len(reports) == 0  # skipped

    def test_tolerance_override(self):
        cfg = _cfg()
        key = RegressionDetector.config_key(cfg)
        baseline = {key: _make_metrics(throughput_tps=1000.0)}
        # Only a 3% regression, but tolerance override is 1%
        current = [(cfg, _make_metrics(throughput_tps=970.0))]
        detector = RegressionDetector(tolerance_overrides={"throughput_tps": 1.0})
        reports = detector.compare(current, baseline)
        assert reports[0].has_regression

    def test_any_regression_true(self):
        mr = MetricRegression("tps", 1000.0, 700.0, 5.0, True)
        report = RegressionReport(config_key="k", regressions=[mr])
        detector = RegressionDetector()
        assert detector.any_regression([report])

    def test_any_regression_false(self):
        report = RegressionReport(config_key="k")
        detector = RegressionDetector()
        assert not detector.any_regression([report])

    def test_render_report_no_regressions(self):
        report = RegressionReport(config_key="k")
        detector = RegressionDetector()
        text = detector.render_report([report])
        assert "No regressions" in text

    def test_render_report_with_regressions(self):
        mr = MetricRegression("tps", 1000.0, 700.0, 5.0, True)
        report = RegressionReport(config_key="k", regressions=[mr])
        detector = RegressionDetector()
        text = detector.render_report([report])
        assert "REGRESSION" in text

    def test_compare_from_file_missing_path(self):
        detector = RegressionDetector()
        with pytest.raises(FileNotFoundError):
            detector.compare_from_file([], "/nonexistent/path.json")
