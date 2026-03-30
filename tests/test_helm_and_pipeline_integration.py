"""Tests for Helm chart structure and pipeline Prometheus/trace integration."""

import json
import os
from unittest import mock

import pytest
import yaml

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint
from benchmark.dag.pipeline import BenchmarkPipeline, PipelineConfig
from benchmark.metrics.prometheus_bridge import PrometheusBridge, PrometheusSnapshot


# ---------------------------------------------------------------------------
# Helm chart structure validation
# ---------------------------------------------------------------------------

HELM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "helm", "benchmark"
)


class TestHelmChartStructure:
    def test_chart_yaml_exists(self):
        path = os.path.join(HELM_DIR, "Chart.yaml")
        assert os.path.isfile(path)

    def test_chart_yaml_valid(self):
        path = os.path.join(HELM_DIR, "Chart.yaml")
        with open(path) as f:
            chart = yaml.safe_load(f)
        assert chart["apiVersion"] == "v2"
        assert "name" in chart
        assert "version" in chart

    def test_values_yaml_exists(self):
        path = os.path.join(HELM_DIR, "values.yaml")
        assert os.path.isfile(path)

    def test_values_yaml_defaults(self):
        path = os.path.join(HELM_DIR, "values.yaml")
        with open(path) as f:
            values = yaml.safe_load(f)
        assert values["namespace"] == "benchmark"
        assert "image" in values
        assert values["image"]["repository"] == "vllm/vllm-openai"
        assert "serviceAccount" in values
        assert "results" in values
        assert "prometheus" in values

    def test_templates_directory_exists(self):
        templates_dir = os.path.join(HELM_DIR, "templates")
        assert os.path.isdir(templates_dir)

    def test_required_templates_exist(self):
        templates_dir = os.path.join(HELM_DIR, "templates")
        required = [
            "_helpers.tpl",
            "namespace.yaml",
            "serviceaccount.yaml",
            "rbac.yaml",
            "pvc.yaml",
            "service.yaml",
            "servicemonitor.yaml",
            "job.yaml",
            "configmap.yaml",
        ]
        for name in required:
            path = os.path.join(templates_dir, name)
            assert os.path.isfile(path), f"Missing template: {name}"

    def test_servicemonitor_template_contains_prometheus_crd(self):
        path = os.path.join(HELM_DIR, "templates", "servicemonitor.yaml")
        with open(path) as f:
            content = f.read()
        assert "monitoring.coreos.com/v1" in content
        assert "ServiceMonitor" in content

    def test_rbac_template_contains_role_and_binding(self):
        path = os.path.join(HELM_DIR, "templates", "rbac.yaml")
        with open(path) as f:
            content = f.read()
        assert "Role" in content
        assert "RoleBinding" in content

    def test_job_template_has_gpu_resources(self):
        path = os.path.join(HELM_DIR, "templates", "job.yaml")
        with open(path) as f:
            content = f.read()
        assert "nvidia.com/gpu" in content

    def test_pvc_template_has_storage(self):
        path = os.path.join(HELM_DIR, "templates", "pvc.yaml")
        with open(path) as f:
            content = f.read()
        assert "PersistentVolumeClaim" in content
        assert "storage" in content

    def test_values_prometheus_section(self):
        path = os.path.join(HELM_DIR, "values.yaml")
        with open(path) as f:
            values = yaml.safe_load(f)
        prom = values["prometheus"]
        assert "serviceMonitor" in prom
        assert prom["serviceMonitor"]["enabled"] is True
        assert "pushgateway" in prom


# ---------------------------------------------------------------------------
# Pipeline config — new Prometheus fields
# ---------------------------------------------------------------------------

class TestPipelineConfigPrometheus:
    def test_default_pushgateway_url_none(self):
        config = PipelineConfig()
        assert config.pushgateway_url is None

    def test_default_prometheus_service_name(self):
        config = PipelineConfig()
        assert config.prometheus_service_name == "vllm-benchmark"

    def test_default_prometheus_service_port(self):
        config = PipelineConfig()
        assert config.prometheus_service_port == 8000

    def test_custom_pushgateway(self):
        config = PipelineConfig(pushgateway_url="http://pushgw:9091")
        assert config.pushgateway_url == "http://pushgw:9091"


# ---------------------------------------------------------------------------
# Pipeline — Prometheus bridge integration
# ---------------------------------------------------------------------------

def _synthetic_metrics_fn(cfg: ConfigPoint) -> BenchmarkMetrics:
    m = BenchmarkMetrics()
    m.throughput_tps = cfg.tp * 1000.0
    m.goodput_rps = cfg.tp * 2.0
    m.end_to_end_throughput_rps = cfg.tp * 2.0
    m.ttft_p90_ms = 200.0 / cfg.tp
    m.tpot_p90_ms = 80.0
    m.joint_slo_attainment_pct = 92.0
    m.gpu_mem_used_gb = 30.0 * cfg.tp
    return m


class TestPipelinePrometheusIntegration:
    def test_k8s_mode_creates_prometheus_bridge(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            execution_mode="kubernetes",
            dry_run=True,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        assert pipeline._prometheus_bridge is not None
        assert pipeline._prometheus_bridge.execution_mode == "kubernetes"

    def test_local_mode_no_bridge_without_pushgateway(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            execution_mode="local",
            dry_run=True,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        assert pipeline._prometheus_bridge is None

    def test_local_mode_with_pushgateway_creates_bridge(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            execution_mode="local",
            pushgateway_url="http://pushgw:9091",
            dry_run=True,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        assert pipeline._prometheus_bridge is not None
        assert pipeline._prometheus_bridge.execution_mode == "local"

    def test_k8s_pipeline_run_with_prometheus(self, tmp_path):
        """K8s pipeline run with Prometheus should work in dry_run mode."""
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            execution_mode="kubernetes",
            dry_run=True,
            results_dir=str(tmp_path),
            enable_trace_recorder=True,
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        report = pipeline.run()
        assert isinstance(report, str)
        assert len(pipeline._results) > 0


# ---------------------------------------------------------------------------
# Pipeline — trace export integration
# ---------------------------------------------------------------------------

class TestPipelineTraceExport:
    def test_stage14_skips_when_disabled(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            dry_run=True,
            results_dir=str(tmp_path),
            enable_trace_recorder=False,
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        # Should not raise
        pipeline.stage14_export_traces()

    def test_stage14_skips_when_no_traces(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            dry_run=True,
            results_dir=str(tmp_path),
            enable_trace_recorder=True,
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        # Trace recorder is enabled but empty
        pipeline.stage14_export_traces()
        # Should not create files
        assert not os.path.exists(os.path.join(str(tmp_path), "benchmark_trace.json"))


# ---------------------------------------------------------------------------
# Automation script existence
# ---------------------------------------------------------------------------

class TestAutomationScript:
    def test_script_exists(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_benchmark.sh"
        )
        assert os.path.isfile(script_path)

    def test_script_is_executable(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_benchmark.sh"
        )
        assert os.access(script_path, os.X_OK)

    def test_script_has_shebang(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_benchmark.sh"
        )
        with open(script_path) as f:
            first_line = f.readline()
        assert first_line.startswith("#!/")

    def test_script_references_helm(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_benchmark.sh"
        )
        with open(script_path) as f:
            content = f.read()
        assert "helm" in content
        assert "upgrade --install" in content
