"""Tests for the report exporter (reporting stage)."""

import csv
import io
import json
import os
import tempfile

import pytest

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint
from benchmark.reporting.exporter import (
    BenchmarkExporter,
    export_csv,
    export_html,
    export_json,
    export_markdown,
)


def _make_result(suite="vllm_parallelism", tp=1, pp=1, tps=1000.0, goodput=5.0, slo=90.0):
    cfg = ConfigPoint(benchmark_suite=suite, tp=tp, pp=pp)
    m = BenchmarkMetrics()
    m.throughput_tps = tps
    m.goodput_rps = goodput
    m.joint_slo_attainment_pct = slo
    m.ttft_p90_ms = 200.0
    m.tpot_p90_ms = 80.0
    m.gpu_mem_used_gb = 30.0
    return cfg, m


class TestExportJSON:
    def test_returns_valid_json(self):
        results = [_make_result(), _make_result(tp=2)]
        text = export_json(results)
        data = json.loads(text)
        assert "results" in data
        assert data["num_results"] == 2

    def test_each_result_has_config_and_metrics(self):
        results = [_make_result()]
        data = json.loads(export_json(results))
        row = data["results"][0]
        assert "tp" in row
        assert "throughput_tps" in row

    def test_empty_results(self):
        data = json.loads(export_json([]))
        assert data["num_results"] == 0
        assert data["results"] == []

    def test_report_text_embedded(self):
        results = [_make_result()]
        text = export_json(results, report_text="REPORT HERE")
        data = json.loads(text)
        assert data["recommendation_report"] == "REPORT HERE"

    def test_no_report_text_key_absent(self):
        results = [_make_result()]
        data = json.loads(export_json(results))
        assert "recommendation_report" not in data

    def test_json_file_written(self, tmp_path):
        from benchmark.reporting.exporter import export_json_file
        results = [_make_result()]
        path = str(tmp_path / "out.json")
        export_json_file(results, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["num_results"] == 1


class TestExportCSV:
    def test_returns_csv_with_header(self):
        results = [_make_result(), _make_result(tp=4)]
        text = export_csv(results)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2

    def test_header_contains_expected_fields(self):
        results = [_make_result()]
        text = export_csv(results)
        first_line = text.split("\n")[0]
        assert "throughput_tps" in first_line
        assert "tp" in first_line

    def test_empty_results_returns_empty_string(self):
        assert export_csv([]) == ""

    def test_csv_file_written(self, tmp_path):
        from benchmark.reporting.exporter import export_csv_file
        results = [_make_result()]
        path = str(tmp_path / "out.csv")
        export_csv_file(results, path)
        assert os.path.exists(path)

    def test_values_correct(self):
        cfg, m = _make_result(tp=8, tps=9999.0)
        text = export_csv([(cfg, m)])
        reader = csv.DictReader(io.StringIO(text))
        row = next(reader)
        assert float(row["throughput_tps"]) == pytest.approx(9999.0)
        assert row["tp"] == "8"


class TestExportMarkdown:
    def test_returns_markdown_with_header(self):
        results = [_make_result()]
        text = export_markdown(results)
        assert "# Benchmark Results" in text

    def test_table_present(self):
        results = [_make_result(tp=2), _make_result(tp=4)]
        text = export_markdown(results)
        # Markdown table has | separators
        assert "|" in text

    def test_report_text_appended(self):
        results = [_make_result()]
        text = export_markdown(results, report_text="MY REPORT")
        assert "MY REPORT" in text

    def test_empty_results(self):
        text = export_markdown([])
        assert "0" in text

    def test_markdown_file_written(self, tmp_path):
        from benchmark.reporting.exporter import export_markdown_file
        results = [_make_result()]
        path = str(tmp_path / "out.md")
        export_markdown_file(results, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "Benchmark" in content


class TestExportHTML:
    def test_returns_html_string(self):
        results = [_make_result()]
        html = export_html(results)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html

    def test_contains_chart_js(self):
        results = [_make_result()]
        html = export_html(results)
        assert "chart.js" in html.lower() or "Chart" in html

    def test_contains_suite_name(self):
        results = [_make_result(suite="sarathi")]
        html = export_html(results)
        assert "sarathi" in html

    def test_slo_badge_ok(self):
        results = [_make_result(slo=95.0)]
        html = export_html(results)
        assert "badge-ok" in html

    def test_slo_badge_bad(self):
        results = [_make_result(slo=20.0)]
        html = export_html(results)
        assert "badge-bad" in html

    def test_report_section_present(self):
        results = [_make_result()]
        html = export_html(results, report_text="CONFIG REPORT")
        assert "CONFIG REPORT" in html

    def test_empty_results(self):
        html = export_html([])
        assert "No results" in html

    def test_html_file_written(self, tmp_path):
        from benchmark.reporting.exporter import export_html_file
        results = [_make_result()]
        path = str(tmp_path / "out.html")
        export_html_file(results, path)
        assert os.path.exists(path)

    def test_multiple_suites(self):
        results = [
            _make_result(suite="vllm_parallelism"),
            _make_result(suite="distserve"),
            _make_result(suite="sarathi"),
        ]
        html = export_html(results)
        assert "vllm_parallelism" in html
        assert "distserve" in html
        assert "sarathi" in html


class TestBenchmarkExporter:
    def test_to_json(self):
        results = [_make_result()]
        exp = BenchmarkExporter(results, "REPORT")
        data = json.loads(exp.to_json())
        assert data["num_results"] == 1

    def test_to_csv(self):
        results = [_make_result()]
        exp = BenchmarkExporter(results)
        text = exp.to_csv()
        assert "throughput_tps" in text

    def test_to_markdown(self):
        results = [_make_result()]
        exp = BenchmarkExporter(results)
        text = exp.to_markdown()
        assert "#" in text

    def test_to_html(self):
        results = [_make_result()]
        exp = BenchmarkExporter(results)
        html = exp.to_html()
        assert "<!DOCTYPE html>" in html

    def test_write_all(self, tmp_path):
        results = [_make_result(), _make_result(tp=2)]
        exp = BenchmarkExporter(results, "REPORT")
        paths = exp.write_all(str(tmp_path), prefix="test")
        assert "json" in paths
        assert "csv" in paths
        assert "markdown" in paths
        assert "html" in paths
        for fmt, path in paths.items():
            assert os.path.exists(path), f"{fmt} file not found at {path}"
            assert os.path.getsize(path) > 0, f"{fmt} file is empty"
