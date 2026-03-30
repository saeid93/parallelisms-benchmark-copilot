"""Tests for Kubernetes integration features."""

import json
import os
import tempfile
from unittest import mock

import pytest

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint
from benchmark.dag.pipeline import BenchmarkPipeline, PipelineConfig
from benchmark.runner.benchmark_runner import (
    BenchmarkRunner,
    RunResult,
    RunStatus,
    render_k8s_job_manifest,
)


# ---------------------------------------------------------------------------
# PipelineConfig K8s fields
# ---------------------------------------------------------------------------

class TestPipelineConfigKubernetes:
    def test_default_execution_mode(self):
        config = PipelineConfig()
        assert config.execution_mode == "local"

    def test_kubernetes_execution_mode(self):
        config = PipelineConfig(execution_mode="kubernetes")
        assert config.execution_mode == "kubernetes"

    def test_default_kubeconfig_is_none(self):
        config = PipelineConfig()
        assert config.kubeconfig is None

    def test_default_service_account_is_none(self):
        config = PipelineConfig()
        assert config.service_account is None

    def test_default_node_selector_is_none(self):
        config = PipelineConfig()
        assert config.node_selector is None

    def test_default_tolerations_is_none(self):
        config = PipelineConfig()
        assert config.tolerations is None

    def test_default_job_timeout(self):
        config = PipelineConfig()
        assert config.job_timeout_seconds == 3600

    def test_custom_k8s_settings(self):
        config = PipelineConfig(
            execution_mode="kubernetes",
            kubeconfig="/home/user/.kube/config",
            service_account="benchmark-sa",
            node_selector={"gpu": "a100"},
            tolerations=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}],
            job_timeout_seconds=7200,
        )
        assert config.execution_mode == "kubernetes"
        assert config.kubeconfig == "/home/user/.kube/config"
        assert config.service_account == "benchmark-sa"
        assert config.node_selector == {"gpu": "a100"}
        assert len(config.tolerations) == 1
        assert config.job_timeout_seconds == 7200

    def test_model_variants_field(self):
        config = PipelineConfig(model_variants=["llama-2-7b", "mistral-7b"])
        assert config.model_variants == ["llama-2-7b", "mistral-7b"]


# ---------------------------------------------------------------------------
# K8s Job manifest rendering
# ---------------------------------------------------------------------------

class TestK8sJobManifest:
    def test_manifest_includes_model_id(self):
        cfg = ConfigPoint(model_id="meta-llama/Llama-2-7b-hf")
        manifest = render_k8s_job_manifest(cfg, "test-run-id", "vllm/vllm:latest")
        env_vars = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        model_env = [e for e in env_vars if e["name"] == "MODEL_ID"]
        assert len(model_env) == 1
        assert model_env[0]["value"] == "meta-llama/Llama-2-7b-hf"

    def test_manifest_with_service_account(self):
        cfg = ConfigPoint()
        manifest = render_k8s_job_manifest(
            cfg, "test-run-id", "vllm/vllm:latest",
            service_account="benchmark-sa",
        )
        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "benchmark-sa"

    def test_manifest_without_service_account(self):
        cfg = ConfigPoint()
        manifest = render_k8s_job_manifest(cfg, "test-run-id", "vllm/vllm:latest")
        pod_spec = manifest["spec"]["template"]["spec"]
        assert "serviceAccountName" not in pod_spec

    def test_manifest_with_node_selector(self):
        cfg = ConfigPoint()
        manifest = render_k8s_job_manifest(
            cfg, "test-run-id", "vllm/vllm:latest",
            node_selector={"gpu": "a100", "zone": "us-west-1a"},
        )
        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["nodeSelector"] == {"gpu": "a100", "zone": "us-west-1a"}

    def test_manifest_without_node_selector(self):
        cfg = ConfigPoint()
        manifest = render_k8s_job_manifest(cfg, "test-run-id", "vllm/vllm:latest")
        pod_spec = manifest["spec"]["template"]["spec"]
        assert "nodeSelector" not in pod_spec

    def test_manifest_with_tolerations(self):
        cfg = ConfigPoint()
        tolerations = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ]
        manifest = render_k8s_job_manifest(
            cfg, "test-run-id", "vllm/vllm:latest",
            tolerations=tolerations,
        )
        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["tolerations"] == tolerations

    def test_manifest_without_tolerations(self):
        cfg = ConfigPoint()
        manifest = render_k8s_job_manifest(cfg, "test-run-id", "vllm/vllm:latest")
        pod_spec = manifest["spec"]["template"]["spec"]
        assert "tolerations" not in pod_spec


# ---------------------------------------------------------------------------
# BenchmarkRunner execution modes
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerModes:
    def test_runner_default_execution_mode(self):
        runner = BenchmarkRunner()
        assert runner.execution_mode == "local"

    def test_runner_kubernetes_mode(self):
        runner = BenchmarkRunner(execution_mode="kubernetes")
        assert runner.execution_mode == "kubernetes"

    def test_runner_stores_kubeconfig(self):
        runner = BenchmarkRunner(kubeconfig="/path/to/kubeconfig")
        assert runner.kubeconfig == "/path/to/kubeconfig"

    def test_runner_stores_k8s_params(self):
        runner = BenchmarkRunner(
            service_account="sa",
            node_selector={"gpu": "a100"},
            tolerations=[{"key": "k", "effect": "NoSchedule"}],
            job_timeout_seconds=1800,
        )
        assert runner.service_account == "sa"
        assert runner.node_selector == {"gpu": "a100"}
        assert len(runner.tolerations) == 1
        assert runner.job_timeout_seconds == 1800

    def test_dry_run_local_returns_pending(self):
        runner = BenchmarkRunner(dry_run=True, execution_mode="local")
        cfg = ConfigPoint()
        result = runner.submit(cfg)
        assert result.status == RunStatus.PENDING

    def test_dry_run_kubernetes_returns_pending(self):
        runner = BenchmarkRunner(dry_run=True, execution_mode="kubernetes")
        cfg = ConfigPoint()
        result = runner.submit(cfg)
        assert result.status == RunStatus.PENDING

    def test_local_mode_writes_manifest(self, tmp_path):
        runner = BenchmarkRunner(
            dry_run=False,
            execution_mode="local",
            results_dir=str(tmp_path),
        )
        cfg = ConfigPoint(model_id="meta-llama/Llama-2-7b-hf")
        result = runner.submit(cfg)
        assert result.status == RunStatus.PENDING
        manifests = list(tmp_path.glob("*.manifest.json"))
        assert len(manifests) == 1
        with open(manifests[0]) as f:
            data = json.load(f)
        env_vars = data["spec"]["template"]["spec"]["containers"][0]["env"]
        model_env = [e for e in env_vars if e["name"] == "MODEL_ID"]
        assert model_env[0]["value"] == "meta-llama/Llama-2-7b-hf"

    def test_kubernetes_mode_submit_kubectl_failure(self, tmp_path):
        """When kubectl is not available, k8s submit should return FAILED."""
        runner = BenchmarkRunner(
            dry_run=False,
            execution_mode="kubernetes",
            results_dir=str(tmp_path),
        )
        cfg = ConfigPoint()
        # kubectl is not installed in test env, so _submit_k8s should fail
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("kubectl not found")):
            result = runner.submit(cfg)
        assert result.status == RunStatus.FAILED
        assert "kubectl not found" in result.error_message

    def test_kubernetes_mode_submit_success(self, tmp_path):
        """Simulate successful kubectl apply."""
        runner = BenchmarkRunner(
            dry_run=False,
            execution_mode="kubernetes",
            results_dir=str(tmp_path),
        )
        cfg = ConfigPoint()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            result = runner.submit(cfg)
        assert result.status == RunStatus.PENDING
        mock_run.assert_called_once()

    def test_kubectl_cmd_without_kubeconfig(self):
        runner = BenchmarkRunner()
        cmd = runner._kubectl_cmd()
        assert cmd == ["kubectl"]

    def test_kubectl_cmd_with_kubeconfig(self):
        runner = BenchmarkRunner(kubeconfig="/path/to/config")
        cmd = runner._kubectl_cmd()
        assert cmd == ["kubectl", "--kubeconfig", "/path/to/config"]


# ---------------------------------------------------------------------------
# K8s job monitoring
# ---------------------------------------------------------------------------

class TestK8sJobMonitoring:
    def test_wait_for_job_local_mode(self):
        runner = BenchmarkRunner(execution_mode="local")
        status = runner.wait_for_job("some-run-id")
        assert status == RunStatus.COMPLETED

    def test_wait_for_job_k8s_success(self):
        runner = BenchmarkRunner(execution_mode="kubernetes")
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            status = runner.wait_for_job("some-run-id")
        assert status == RunStatus.COMPLETED

    def test_wait_for_job_k8s_failure(self):
        runner = BenchmarkRunner(execution_mode="kubernetes")
        import subprocess as sp
        with mock.patch("subprocess.run", side_effect=sp.CalledProcessError(1, "kubectl")):
            status = runner.wait_for_job("some-run-id")
        assert status == RunStatus.FAILED

    def test_get_job_logs_local_returns_none(self):
        runner = BenchmarkRunner(execution_mode="local")
        logs = runner.get_job_logs("some-run-id")
        assert logs is None

    def test_get_job_logs_k8s_success(self):
        runner = BenchmarkRunner(execution_mode="kubernetes")
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="log output here")
            logs = runner.get_job_logs("some-run-id")
        assert logs == "log output here"

    def test_get_job_logs_k8s_failure(self):
        runner = BenchmarkRunner(execution_mode="kubernetes")
        import subprocess as sp
        with mock.patch("subprocess.run", side_effect=sp.CalledProcessError(1, "kubectl")):
            logs = runner.get_job_logs("some-run-id")
        assert logs is None


# ---------------------------------------------------------------------------
# Pipeline integration with new features
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


class TestPipelineWithModelVariants:
    def test_pipeline_with_single_variant(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            model_variants=["llama-2-7b"],
            dry_run=True,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        report = pipeline.run()
        assert isinstance(report, str)
        assert len(pipeline._results) > 0
        for cfg, _ in pipeline._results:
            assert cfg.model_id == "meta-llama/Llama-2-7b-hf"

    def test_pipeline_with_multiple_variants(self, tmp_path):
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            model_variants=["llama-2-7b", "mistral-7b"],
            dry_run=True,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        report = pipeline.run()
        model_ids = {cfg.model_id for cfg, _ in pipeline._results}
        assert "meta-llama/Llama-2-7b-hf" in model_ids
        assert "mistralai/Mistral-7B-v0.1" in model_ids


class TestPipelineWithKubernetes:
    def test_pipeline_k8s_mode_dry_run(self, tmp_path):
        """K8s mode with dry_run=True should work without kubectl."""
        config = PipelineConfig(
            max_gpus=4,
            suites=["vllm_parallelism"],
            execution_mode="kubernetes",
            dry_run=True,
            results_dir=str(tmp_path),
            namespace="test-ns",
            service_account="test-sa",
            node_selector={"gpu": "a100"},
        )
        pipeline = BenchmarkPipeline(
            config=config,
            metrics_fn_override=_synthetic_metrics_fn,
        )
        report = pipeline.run()
        assert isinstance(report, str)
        assert len(pipeline._results) > 0

    def test_pipeline_passes_k8s_config_to_runner(self, tmp_path):
        config = PipelineConfig(
            execution_mode="kubernetes",
            kubeconfig="/test/kubeconfig",
            service_account="test-sa",
            node_selector={"gpu": "a100"},
            tolerations=[{"key": "k"}],
            job_timeout_seconds=1800,
            results_dir=str(tmp_path),
        )
        pipeline = BenchmarkPipeline(config=config)
        runner = pipeline._runner
        assert runner.execution_mode == "kubernetes"
        assert runner.kubeconfig == "/test/kubeconfig"
        assert runner.service_account == "test-sa"
        assert runner.node_selector == {"gpu": "a100"}
        assert runner.tolerations == [{"key": "k"}]
        assert runner.job_timeout_seconds == 1800
