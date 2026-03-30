"""
Stage 3 — Parallel benchmark runner.

Responsible for spawning one vLLM instance per config as a Kubernetes Job,
or running locally for development purposes.  The actual vLLM execution
happens inside the Job container; this module provides the orchestration
layer that submits Jobs, monitors them, and triggers early stopping when
a run OOMs or falls below a throughput floor.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from benchmark.config.schema import BenchmarkConfig, BenchmarkMetrics, BenchmarkRun
from benchmark.config.sweep import ConfigPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run status
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OOM = "oom"
    EARLY_STOPPED = "early_stopped"


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    config: ConfigPoint
    status: RunStatus
    metrics: Optional[BenchmarkMetrics] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Early stopping policy
# ---------------------------------------------------------------------------

THROUGHPUT_FLOOR_TPS = 1.0


def should_early_stop(metrics: BenchmarkMetrics, floor_tps: float = THROUGHPUT_FLOOR_TPS) -> bool:
    """Return True when a run should be aborted due to poor performance.

    Args:
        metrics: Partial metrics collected so far.
        floor_tps: Minimum acceptable throughput in tokens per second.

    Returns:
        True if the run should be aborted, False otherwise.
    """
    if metrics.throughput_tps > 0 and metrics.throughput_tps < floor_tps:
        return True
    return False


# ---------------------------------------------------------------------------
# Kubernetes Job template rendering
# ---------------------------------------------------------------------------

def render_k8s_job_manifest(
    cfg: ConfigPoint,
    run_id: str,
    image: str,
    namespace: str = "benchmark",
    results_pvc: str = "benchmark-results",
    service_account: Optional[str] = None,
    node_selector: Optional[Dict[str, str]] = None,
    tolerations: Optional[List[Dict[str, str]]] = None,
) -> Dict:
    """Render a Kubernetes Job manifest for a single benchmark run.

    Args:
        cfg: The config point to benchmark.
        run_id: Unique run identifier.
        image: Container image to use (must include vLLM + benchmark scripts).
        namespace: Kubernetes namespace.
        results_pvc: PersistentVolumeClaim name for storing result JSONs.
        service_account: Optional Kubernetes ServiceAccount for pods.
        node_selector: Optional node selector labels for pods.
        tolerations: Optional list of toleration dicts for pods.

    Returns:
        Dict representing the Kubernetes Job manifest.
    """
    gpu_count = (
        cfg.disaggregated_gpu_count()
        if cfg.disaggregation_mode in ("distserve", "seesaw_resharding")
        else cfg.gpu_count()
    )

    env_vars = [
        {"name": "RUN_ID", "value": run_id},
        {"name": "BENCHMARK_SUITE", "value": cfg.benchmark_suite},
        {"name": "MODEL_ID", "value": cfg.model_id},
        {"name": "TP", "value": str(cfg.tp)},
        {"name": "PP", "value": str(cfg.pp)},
        {"name": "DP", "value": str(cfg.dp)},
        {"name": "DATASET", "value": cfg.dataset},
        {"name": "MAX_MODEL_LEN", "value": str(cfg.max_model_len)},
        {"name": "GPU_MEM_UTIL", "value": str(cfg.gpu_mem_util)},
        {"name": "CHUNKED_PREFILL", "value": str(cfg.chunked_prefill).lower()},
        {"name": "CHUNK_SIZE", "value": str(cfg.chunk_size)},
        {"name": "DTYPE", "value": cfg.dtype},
        {"name": "ATTENTION_BACKEND", "value": cfg.attention_backend},
        {"name": "ARRIVAL_PROCESS", "value": cfg.arrival_process},
        {"name": "REQUEST_RATE_RPS", "value": str(cfg.request_rate_rps)},
        {"name": "NUM_REQUESTS", "value": str(cfg.num_requests)},
        {"name": "DISAGGREGATION_MODE", "value": cfg.disaggregation_mode},
        {"name": "RESULTS_DIR", "value": "/results"},
    ]

    if cfg.disaggregation_mode in ("distserve", "seesaw_resharding"):
        env_vars += [
            {"name": "PREFILL_TP", "value": str(cfg.prefill_tp or 1)},
            {"name": "PREFILL_PP", "value": str(cfg.prefill_pp or 1)},
            {"name": "DECODE_TP", "value": str(cfg.decode_tp or 1)},
            {"name": "DECODE_PP", "value": str(cfg.decode_pp or 1)},
        ]

    pod_spec: Dict = {
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "benchmark",
                "image": image,
                "env": env_vars,
                "resources": {
                    "requests": {
                        "nvidia.com/gpu": str(gpu_count)
                    },
                    "limits": {
                        "nvidia.com/gpu": str(gpu_count)
                    },
                },
                "volumeMounts": [
                    {
                        "mountPath": "/results",
                        "name": "results-volume",
                    }
                ],
            }
        ],
        "volumes": [
            {
                "name": "results-volume",
                "persistentVolumeClaim": {
                    "claimName": results_pvc,
                },
            }
        ],
    }

    if service_account:
        pod_spec["serviceAccountName"] = service_account
    if node_selector:
        pod_spec["nodeSelector"] = dict(node_selector)
    if tolerations:
        pod_spec["tolerations"] = list(tolerations)

    manifest: Dict = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"benchmark-{run_id[:8]}",
            "namespace": namespace,
            "labels": {
                "app": "parallelisms-benchmark",
                "suite": cfg.benchmark_suite,
                "run-id": run_id,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "parallelisms-benchmark",
                        "run-id": run_id,
                    }
                },
                "spec": pod_spec,
            },
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates benchmark job submission and result collection.

    Supports two execution modes controlled by *execution_mode*:

    * ``"local"`` (default) — In dry-run mode renders manifests to disk.
      Without dry-run writes manifests for local inspection.
    * ``"kubernetes"`` — Submits Kubernetes Jobs via ``kubectl`` and polls
      for completion.  Requires ``kubectl`` on ``$PATH`` and a valid
      kubeconfig (set via *kubeconfig* or ``$KUBECONFIG``).

    Args:
        image: Container image for Kubernetes Job pods.
        namespace: Kubernetes namespace for Job submission.
        results_pvc: PVC name for storing result JSON files.
        results_dir: Local directory for results when running locally.
        dry_run: If True, do not submit Jobs; only render manifests.
        throughput_floor_tps: Minimum TPS before early stopping.
        execution_mode: ``"local"`` or ``"kubernetes"``.
        kubeconfig: Optional path to kubeconfig file.
        service_account: Optional Kubernetes ServiceAccount for pods.
        node_selector: Optional node selector labels for pods.
        tolerations: Optional tolerations for pods.
        job_timeout_seconds: Max seconds to wait for a K8s Job.
    """

    def __init__(
        self,
        image: str = "vllm/vllm-openai:latest",
        namespace: str = "benchmark",
        results_pvc: str = "benchmark-results",
        results_dir: str = "/tmp/benchmark-results",
        dry_run: bool = True,
        throughput_floor_tps: float = THROUGHPUT_FLOOR_TPS,
        execution_mode: str = "local",
        kubeconfig: Optional[str] = None,
        service_account: Optional[str] = None,
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict[str, str]]] = None,
        job_timeout_seconds: int = 3600,
    ) -> None:
        self.image = image
        self.namespace = namespace
        self.results_pvc = results_pvc
        self.results_dir = results_dir
        self.dry_run = dry_run
        self.throughput_floor_tps = throughput_floor_tps
        self.execution_mode = execution_mode
        self.kubeconfig = kubeconfig
        self.service_account = service_account
        self.node_selector = dict(node_selector) if node_selector is not None else {}
        self.tolerations = list(tolerations) if tolerations is not None else []
        self.job_timeout_seconds = job_timeout_seconds

    def submit(self, cfg: ConfigPoint) -> RunResult:
        """Submit a benchmark job for a single config.

        In ``"kubernetes"`` execution mode, renders a full K8s Job manifest
        and submits it via ``kubectl apply``.  In ``"local"`` mode (the
        default) the manifest is written to *results_dir* for inspection.
        When *dry_run* is True the manifest is only logged, regardless of
        execution mode.

        Args:
            cfg: Configuration point to benchmark.

        Returns:
            RunResult with status set to PENDING (or COMPLETED in dry_run).
        """
        run_id = str(uuid.uuid4())
        manifest = render_k8s_job_manifest(
            cfg,
            run_id,
            self.image,
            self.namespace,
            self.results_pvc,
            service_account=self.service_account,
            node_selector=self.node_selector or None,
            tolerations=self.tolerations or None,
        )
        if self.dry_run:
            logger.info(
                "dry_run=True — would submit Job %s for suite=%s model=%s tp=%d pp=%d",
                run_id[:8],
                cfg.benchmark_suite,
                cfg.model_id,
                cfg.tp,
                cfg.pp,
            )
            return RunResult(
                run_id=run_id,
                config=cfg,
                status=RunStatus.PENDING,
            )

        os.makedirs(self.results_dir, exist_ok=True)
        manifest_path = os.path.join(self.results_dir, f"{run_id}.manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        if self.execution_mode == "kubernetes":
            return self._submit_k8s(run_id, cfg, manifest_path)

        # local mode — manifest written for inspection only
        logger.info("Submitted Job %s manifest to %s", run_id[:8], manifest_path)
        return RunResult(
            run_id=run_id,
            config=cfg,
            status=RunStatus.PENDING,
        )

    # ------------------------------------------------------------------
    # Kubernetes-specific helpers
    # ------------------------------------------------------------------

    def _kubectl_cmd(self) -> List[str]:
        """Build the base kubectl command list."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        return cmd

    def _submit_k8s(
        self,
        run_id: str,
        cfg: ConfigPoint,
        manifest_path: str,
    ) -> RunResult:
        """Submit a single Job to Kubernetes via ``kubectl apply``.

        Args:
            run_id: Unique run identifier.
            cfg: The ConfigPoint being benchmarked.
            manifest_path: Path to the rendered manifest JSON on disk.

        Returns:
            RunResult reflecting the submission outcome.
        """
        cmd = self._kubectl_cmd() + [
            "apply", "-f", manifest_path, "-n", self.namespace,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(
                "Kubernetes Job %s submitted for model=%s suite=%s",
                run_id[:8], cfg.model_id, cfg.benchmark_suite,
            )
            return RunResult(
                run_id=run_id,
                config=cfg,
                status=RunStatus.PENDING,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.error("Failed to submit K8s Job %s: %s", run_id[:8], exc)
            return RunResult(
                run_id=run_id,
                config=cfg,
                status=RunStatus.FAILED,
                error_message=str(exc),
            )

    def wait_for_job(self, run_id: str) -> RunStatus:
        """Wait for a Kubernetes Job to complete.

        Uses ``kubectl wait`` with ``--timeout`` set to
        *job_timeout_seconds*.  Only meaningful in ``"kubernetes"``
        execution mode.

        Args:
            run_id: Run identifier (first 8 chars used as Job name suffix).

        Returns:
            RunStatus reflecting the Job outcome.
        """
        if self.execution_mode != "kubernetes":
            return RunStatus.COMPLETED

        job_name = f"benchmark-{run_id[:8]}"
        cmd = self._kubectl_cmd() + [
            "wait", f"job/{job_name}",
            "-n", self.namespace,
            f"--for=condition=complete",
            f"--timeout={self.job_timeout_seconds}s",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return RunStatus.COMPLETED
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Check if it failed (as opposed to just timing out).
            return RunStatus.FAILED

    def get_job_logs(self, run_id: str) -> Optional[str]:
        """Retrieve logs from a Kubernetes Job pod.

        Args:
            run_id: Run identifier (first 8 chars used as Job name suffix).

        Returns:
            Pod logs as a string, or None on failure.
        """
        if self.execution_mode != "kubernetes":
            return None

        job_name = f"benchmark-{run_id[:8]}"
        cmd = self._kubectl_cmd() + [
            "logs", f"job/{job_name}", "-n", self.namespace,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def submit_batch(
        self, configs: List[ConfigPoint], max_parallel: int = 8
    ) -> List[RunResult]:
        """Submit a batch of benchmark jobs (bin-packed to GPU budget).

        Args:
            configs: List of ConfigPoint instances to benchmark.
            max_parallel: Maximum number of jobs to run concurrently
                (bounded by cluster GPU budget).

        Returns:
            List of RunResult objects, one per config.
        """
        results: List[RunResult] = []
        running: List[RunResult] = []
        for cfg in configs:
            # Trivial bin-packing: limit parallel inflight jobs.
            while len(running) >= max_parallel:
                # Poll for completions (stub: treat all as completed).
                completed = running.pop(0)
                completed.status = RunStatus.COMPLETED
                results.append(completed)

            result = self.submit(cfg)
            running.append(result)

        # Drain remaining
        for r in running:
            r.status = RunStatus.COMPLETED
            results.append(r)

        return results

    def load_result(self, run_id: str) -> Optional[BenchmarkRun]:
        """Load a completed benchmark result from disk.

        Args:
            run_id: The UUID of the run to load.

        Returns:
            BenchmarkRun if the result file exists, else None.
        """
        result_path = os.path.join(self.results_dir, f"{run_id}.json")
        if not os.path.exists(result_path):
            return None
        with open(result_path) as fh:
            data = json.load(fh)
        return BenchmarkRun.model_validate(data)
