"""
DAG pipeline orchestrator.

Wires together all seven stages of the benchmark pipeline:

  Stage 1  — Config space generator (sweep.py)
  Stage 2  — Workload generator (workload/generator.py)
  Stage 3  — Parallel benchmark runners (runner/benchmark_runner.py)
  Stage 4  — Metrics collector (metrics/collector.py)
  Stage 5  — SLO attainment evaluator (analysis/slo_evaluator.py)
  Stage 6  — Pareto analyser (analysis/pareto.py)
  Stage 7  — Recommendation synthesiser (analysis/recommender.py)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from benchmark.analysis.pareto import ParetoAnalyser
from benchmark.analysis.recommender import RecommendationSynthesiser
from benchmark.analysis.slo_evaluator import SLOEvaluator
from benchmark.config.schema import BenchmarkMetrics, BenchmarkRun
from benchmark.config.sweep import ConfigPoint, generate_full_sweep
from benchmark.metrics.collector import MetricsCollector, RequestTiming
from benchmark.runner.benchmark_runner import BenchmarkRunner, RunResult, RunStatus
from benchmark.workload.generator import WorkloadGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Top-level configuration for the benchmark pipeline.

    Attributes:
        max_gpus: Total GPUs available for hardware feasibility pruning.
        model_params_gb: Approximate model parameter size in GiB.
        suites: Benchmark suites to run (None means all four).
        image: Container image for Kubernetes Jobs.
        namespace: Kubernetes namespace.
        results_pvc: PersistentVolumeClaim for result storage.
        results_dir: Local results directory (used in dry_run mode).
        dry_run: If True, skip actual Job submission.
        max_parallel_jobs: Maximum concurrent benchmark jobs.
        throughput_floor_tps: Minimum TPS for early stopping.
        seed: Random seed for workload generation.
    """

    max_gpus: int = 8
    model_params_gb: float = 14.0
    suites: Optional[List[str]] = None
    image: str = "vllm/vllm-openai:latest"
    namespace: str = "benchmark"
    results_pvc: str = "benchmark-results"
    results_dir: str = "/tmp/benchmark-results"
    dry_run: bool = True
    max_parallel_jobs: int = 8
    throughput_floor_tps: float = 1.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class BenchmarkPipeline:
    """End-to-end benchmark pipeline implementing the 7-stage DAG.

    Args:
        config: PipelineConfig controlling all pipeline parameters.
        metrics_fn_override: Optional callable that replaces the real vLLM
            metrics collection with a synthetic function.  Useful for
            testing without a live vLLM cluster.
            Signature: (cfg: ConfigPoint) -> BenchmarkMetrics
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        metrics_fn_override: Optional[
            Callable[[ConfigPoint], BenchmarkMetrics]
        ] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._metrics_fn_override = metrics_fn_override
        self._runner = BenchmarkRunner(
            image=self.config.image,
            namespace=self.config.namespace,
            results_pvc=self.config.results_pvc,
            results_dir=self.config.results_dir,
            dry_run=self.config.dry_run,
            throughput_floor_tps=self.config.throughput_floor_tps,
        )
        self._results: List[Tuple[ConfigPoint, BenchmarkMetrics]] = []

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def stage1_generate_configs(self) -> List[ConfigPoint]:
        """Generate and prune the full config sweep matrix."""
        logger.info(
            "Stage 1: generating config sweep (max_gpus=%d, model_params_gb=%.1f)",
            self.config.max_gpus,
            self.config.model_params_gb,
        )
        configs = generate_full_sweep(
            max_gpus=self.config.max_gpus,
            model_params_gb=self.config.model_params_gb,
            suites=self.config.suites,
        )
        logger.info("Stage 1: %d feasible configs generated", len(configs))
        return configs

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def stage2_generate_workload(self, cfg: ConfigPoint) -> WorkloadGenerator:
        """Build a WorkloadGenerator for a single config."""
        return WorkloadGenerator(
            dataset=cfg.dataset,
            arrival_process=cfg.arrival_process,
            request_rate_rps=cfg.request_rate_rps,
            num_requests=cfg.num_requests,
            avg_input_tokens=cfg.avg_input_tokens,
            avg_output_tokens=cfg.avg_output_tokens,
            seed=self.config.seed,
        )

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def stage3_submit_jobs(
        self, configs: List[ConfigPoint]
    ) -> List[RunResult]:
        """Submit benchmark jobs for all configs."""
        logger.info(
            "Stage 3: submitting %d jobs (max_parallel=%d)",
            len(configs),
            self.config.max_parallel_jobs,
        )
        return self._runner.submit_batch(
            configs, max_parallel=self.config.max_parallel_jobs
        )

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------

    def stage4_collect_metrics(
        self,
        cfg: ConfigPoint,
        run_result: RunResult,
        workload: WorkloadGenerator,
    ) -> BenchmarkMetrics:
        """Collect and aggregate metrics for a completed run."""
        if self._metrics_fn_override is not None:
            return self._metrics_fn_override(cfg)

        # In production: load the BenchmarkRun JSON written by the vLLM
        # container and populate a MetricsCollector from it.
        run = self._runner.load_result(run_result.run_id)
        if run is not None:
            return run.metrics

        # Fallback: return zero metrics (run not yet finished or dry_run).
        logger.debug(
            "No result found for run_id=%s; returning zero metrics",
            run_result.run_id,
        )
        return BenchmarkMetrics()

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------

    def stage5_evaluate_slo(
        self,
        cfg: ConfigPoint,
        metrics: BenchmarkMetrics,
    ) -> BenchmarkMetrics:
        """Run SLO attainment evaluation and update metrics.goodput_rps."""

        def _metrics_fn(
            rate: float, ttft_slo: float, tpot_slo: float
        ) -> BenchmarkMetrics:
            # In production this would re-run the benchmark at `rate`.
            # Here we scale from the already-collected metrics as a proxy.
            scaled = metrics.model_copy()
            if cfg.request_rate_rps > 0:
                scale_factor = rate / cfg.request_rate_rps
            else:
                scale_factor = 1.0
            # Latency grows roughly linearly with load (simplified model).
            scaled.ttft_p90_ms = metrics.ttft_p90_ms * scale_factor
            scaled.tpot_p90_ms = metrics.tpot_p90_ms * scale_factor
            scaled.joint_slo_attainment_pct = (
                100.0 if (
                    scaled.ttft_p90_ms <= ttft_slo
                    and scaled.tpot_p90_ms <= tpot_slo
                )
                else 0.0
            )
            return scaled

        evaluator = SLOEvaluator(
            metrics_fn=_metrics_fn,
            attainment_target_pct=cfg.slo_attainment_target_pct,
        )
        goodput_rps, _ = evaluator.find_goodput(
            cfg.ttft_slo_ms * cfg.slo_scale,
            cfg.tpot_slo_ms * cfg.slo_scale,
        )
        metrics = metrics.model_copy()
        metrics.goodput_rps = goodput_rps
        return metrics

    # ------------------------------------------------------------------
    # Stage 6 + 7
    # ------------------------------------------------------------------

    def stage6_pareto_analysis(
        self,
    ) -> Dict[str, List]:
        """Run Pareto analysis across all collected results."""
        analyser = ParetoAnalyser(self._results)
        frontiers = analyser.analyse()
        logger.info(
            "Stage 6: Pareto frontiers — %s",
            {suite: len(pts) for suite, pts in frontiers.items()},
        )
        return frontiers

    def stage7_recommend(self) -> str:
        """Synthesise and render the recommendation report."""
        synth = RecommendationSynthesiser(self._results)
        report = synth.render_report()
        logger.info("Stage 7: recommendation report generated")
        return report

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute the full 7-stage pipeline.

        Returns:
            Rendered recommendation report text.
        """
        # Stage 1
        configs = self.stage1_generate_configs()
        if not configs:
            return "No feasible configs generated."

        # Stage 3 — submit all jobs
        run_results = self.stage3_submit_jobs(configs)

        # Stages 2, 4, 5 — per-config
        for cfg, run_result in zip(configs, run_results):
            # Stage 2
            workload = self.stage2_generate_workload(cfg)

            # Stage 4
            metrics = self.stage4_collect_metrics(cfg, run_result, workload)

            # Stage 5 (DistServe suite)
            if cfg.benchmark_suite == "distserve":
                metrics = self.stage5_evaluate_slo(cfg, metrics)

            self._results.append((cfg, metrics))

        # Stage 6
        self.stage6_pareto_analysis()

        # Stage 7
        report = self.stage7_recommend()

        # Persist report
        os.makedirs(self.config.results_dir, exist_ok=True)
        report_path = os.path.join(self.config.results_dir, "report.txt")
        with open(report_path, "w") as fh:
            fh.write(report)
        logger.info("Report written to %s", report_path)

        return report
