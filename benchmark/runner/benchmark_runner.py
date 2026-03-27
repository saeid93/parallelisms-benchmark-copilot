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
) -> Dict:
    """Render a Kubernetes Job manifest for a single benchmark run.

    Args:
        cfg: The config point to benchmark.
        run_id: Unique run identifier.
        image: Container image to use (must include vLLM + benchmark scripts).
        namespace: Kubernetes namespace.
        results_pvc: PersistentVolumeClaim name for storing result JSONs.

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
                "spec": {
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
                },
            },
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates benchmark job submission and result collection.

    In production this class submits Kubernetes Jobs and polls their status.
    In dry-run/local mode it can invoke a local benchmark script directly.

    Args:
        image: Container image for Kubernetes Job pods.
        namespace: Kubernetes namespace for Job submission.
        results_pvc: PVC name for storing result JSON files.
        results_dir: Local directory for results when running locally.
        dry_run: If True, do not submit Jobs; only render manifests.
        throughput_floor_tps: Minimum TPS before early stopping.
    """

    def __init__(
        self,
        image: str = "vllm/vllm-openai:latest",
        namespace: str = "benchmark",
        results_pvc: str = "benchmark-results",
        results_dir: str = "/tmp/benchmark-results",
        dry_run: bool = True,
        throughput_floor_tps: float = THROUGHPUT_FLOOR_TPS,
    ) -> None:
        self.image = image
        self.namespace = namespace
        self.results_pvc = results_pvc
        self.results_dir = results_dir
        self.dry_run = dry_run
        self.throughput_floor_tps = throughput_floor_tps

    def submit(self, cfg: ConfigPoint) -> RunResult:
        """Submit a benchmark job for a single config.

        Args:
            cfg: Configuration point to benchmark.

        Returns:
            RunResult with status set to PENDING (or COMPLETED in dry_run).
        """
        run_id = str(uuid.uuid4())
        manifest = render_k8s_job_manifest(
            cfg, run_id, self.image, self.namespace, self.results_pvc
        )
        if self.dry_run:
            logger.info(
                "dry_run=True — would submit Job %s for suite=%s tp=%d pp=%d",
                run_id[:8],
                cfg.benchmark_suite,
                cfg.tp,
                cfg.pp,
            )
            return RunResult(
                run_id=run_id,
                config=cfg,
                status=RunStatus.PENDING,
            )
        # In a real deployment, submit manifest via kubectl or the k8s API.
        # Placeholder: write manifest to results_dir for inspection.
        os.makedirs(self.results_dir, exist_ok=True)
        manifest_path = os.path.join(self.results_dir, f"{run_id}.manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("Submitted Job %s manifest to %s", run_id[:8], manifest_path)
        return RunResult(
            run_id=run_id,
            config=cfg,
            status=RunStatus.PENDING,
        )

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
