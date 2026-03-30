"""Integration tests for the full benchmark pipeline."""

import os
import tempfile

import pytest

from benchmark.analysis.bottleneck import BottleneckAnalyser
from benchmark.analysis.cost_estimator import CostEstimator
from benchmark.analysis.regression import RegressionDetector
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint
from benchmark.config.validation import ConfigValidator
from benchmark.dag.pipeline import BenchmarkPipeline, PipelineConfig
from benchmark.reporting.exporter import BenchmarkExporter


def _synthetic_metrics_fn(cfg: ConfigPoint) -> BenchmarkMetrics:
    """Synthetic metrics function for pipeline testing."""
    m = BenchmarkMetrics()
    m.throughput_tps = cfg.tp * 1000.0
    m.goodput_rps = cfg.tp * 2.0
    m.end_to_end_throughput_rps = cfg.tp * 2.0
    m.ttft_p90_ms = 200.0 / cfg.tp
    m.tpot_p90_ms = 80.0
    m.joint_slo_attainment_pct = 92.0
    m.gpu_mem_used_gb = 30.0 * cfg.tp
    m.prefill_phase_time_pct = 40.0
    m.decode_phase_time_pct = 60.0
    return m


class TestBenchmarkPipeline:
    def _small_pipeline(self, tmp_dir: str) -> BenchmarkPipeline:
        config = PipelineConfig(
            max_gpus=4,
            model_params_gb=7.0,
            suites=["vllm_parallelism"],
            dry_run=True,
            results_dir=tmp_dir,
            validate_configs=True,
            enable_gpu_profiler=False,
            enable_trace_recorder=False,
        )
        return BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )

    def test_stage1_generates_configs(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        configs = pipeline.stage1_generate_configs()
        assert len(configs) > 0
        assert all(cfg.benchmark_suite == "vllm_parallelism" for cfg in configs)

    def test_stage8_validates_configs(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        configs = pipeline.stage1_generate_configs()
        valid = pipeline.stage8_validate_configs(configs)
        # All default configs should be valid
        assert len(valid) > 0
        assert len(valid) <= len(configs)

    def test_full_pipeline_run(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        report = pipeline.run()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_pipeline_writes_report_txt(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        pipeline.run()
        report_path = os.path.join(str(tmp_path), "report.txt")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            content = f.read()
        assert len(content) > 0

    def test_pipeline_exports_all_formats(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        pipeline.run()
        for ext in [".json", ".csv", ".md", ".html"]:
            path = os.path.join(str(tmp_path), f"benchmark{ext}")
            assert os.path.exists(path), f"Missing {ext} export"

    def test_stage9_bottleneck_analysis(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        # Run enough of the pipeline to collect results
        configs = pipeline.stage1_generate_configs()[:3]
        for cfg in configs:
            m = _synthetic_metrics_fn(cfg)
            pipeline._results.append((cfg, m))
        bottlenecks = pipeline.stage9_analyse_bottlenecks()
        assert len(bottlenecks) == len(pipeline._results)

    def test_stage10_cost_estimation(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        configs = pipeline.stage1_generate_configs()[:3]
        for cfg in configs:
            m = _synthetic_metrics_fn(cfg)
            pipeline._results.append((cfg, m))
        estimates = pipeline.stage10_estimate_costs()
        assert len(estimates) == len(pipeline._results)
        assert all(e.cost_per_hour_usd > 0 for e in estimates)

    def test_stage11_no_baseline_returns_empty(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        configs = pipeline.stage1_generate_configs()[:2]
        for cfg in configs:
            pipeline._results.append((cfg, _synthetic_metrics_fn(cfg)))
        # No baseline_path set → empty reports
        reports = pipeline.stage11_detect_regressions()
        assert reports == []

    def test_stage11_with_baseline(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        configs = pipeline.stage1_generate_configs()[:3]
        results = [(cfg, _synthetic_metrics_fn(cfg)) for cfg in configs]
        pipeline._results.extend(results)

        # Save baseline
        baseline_path = os.path.join(str(tmp_path), "baseline.json")
        detector = RegressionDetector()
        detector.save_baseline(results, baseline_path)

        pipeline.config.baseline_path = baseline_path
        reports = pipeline.stage11_detect_regressions()
        # Same results → no regressions
        assert not detector.any_regression(reports)

    def test_pipeline_no_configs_returns_message(self, tmp_path):
        config = PipelineConfig(
            max_gpus=0,  # zero GPUs → all configs infeasible
            model_params_gb=7.0,
            suites=["vllm_parallelism"],
            dry_run=True,
            results_dir=str(tmp_path),
            validate_configs=False,
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        result = pipeline.run()
        assert "No feasible" in result

    def test_pipeline_results_stored(self, tmp_path):
        pipeline = self._small_pipeline(str(tmp_path))
        pipeline.run()
        assert len(pipeline._results) > 0


class TestPipelineConfigDefaults:
    def test_default_validate_configs(self):
        config = PipelineConfig()
        assert config.validate_configs is True

    def test_default_gpu_instance(self):
        config = PipelineConfig()
        assert config.gpu_instance == "a100_sxm4_80gb"

    def test_default_export_formats(self):
        config = PipelineConfig()
        assert "json" in config.export_formats
        assert "html" in config.export_formats

    def test_default_baseline_path_none(self):
        config = PipelineConfig()
        assert config.baseline_path is None


class TestIntegrationWithNewModules:
    """Integration tests verifying the new modules work together."""

    def test_cost_and_bottleneck_together(self):
        results = [
            (
                ConfigPoint(benchmark_suite="vllm_parallelism", tp=2, pp=2),
                BenchmarkMetrics(
                    throughput_tps=5000.0,
                    goodput_rps=10.0,
                    end_to_end_throughput_rps=10.0,
                    preemption_rate=0.01,
                    joint_slo_attainment_pct=92.0,
                ),
            ),
        ]
        cost_est = CostEstimator().estimate_batch(results)
        btl_results = BottleneckAnalyser().analyse_batch(results)
        assert len(cost_est) == 1
        assert len(btl_results) == 1

    def test_exporter_with_report(self, tmp_path):
        results = [
            (
                ConfigPoint(benchmark_suite="sarathi", tp=1),
                BenchmarkMetrics(throughput_tps=1000.0, goodput_rps=3.0),
            ),
        ]
        exp = BenchmarkExporter(results, report_text="TEST REPORT")
        paths = exp.write_all(str(tmp_path), prefix="integration")
        for ext in [".json", ".csv", ".md", ".html"]:
            path = os.path.join(str(tmp_path), f"integration{ext}")
            assert os.path.exists(path)

    def test_validator_then_exporter(self, tmp_path):
        configs = [
            ConfigPoint(tp=1, pp=1),
            ConfigPoint(tp=2, pp=2),
        ]
        validator = ConfigValidator()
        valid = validator.filter_valid(configs)
        assert len(valid) > 0

        results = [
            (cfg, BenchmarkMetrics(throughput_tps=float(cfg.tp * 1000)))
            for cfg in valid
        ]
        exp = BenchmarkExporter(results)
        paths = exp.write_all(str(tmp_path), prefix="validated")
        assert "json" in paths
