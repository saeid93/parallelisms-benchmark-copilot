"""
DAG pipeline orchestrator.

Wires together all benchmark pipeline stages:

  Stage 1  — Config space generator (sweep.py)
  Stage 2  — Workload generator (workload/generator.py)
  Stage 3  — Parallel benchmark runners (runner/benchmark_runner.py)
  Stage 4  — Metrics collector (metrics/collector.py)
  Stage 5  — SLO attainment evaluator (analysis/slo_evaluator.py)
  Stage 6  — Pareto analyser (analysis/pareto.py)
  Stage 7  — Recommendation synthesiser (analysis/recommender.py)
  Stage 8  — Config validator (config/validation.py)
  Stage 9  — Bottleneck analyser (analysis/bottleneck.py)
  Stage 10 — Cost estimator (analysis/cost_estimator.py)
  Stage 11 — Regression detector (analysis/regression.py)
  Stage 12 — Report exporter (reporting/exporter.py)
  Stage 13 — GPU profiler (profiler/gpu_profiler.py)
  Stage 14 — Trace recorder (profiler/trace_recorder.py)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from benchmark.analysis.bottleneck import BottleneckAnalyser, BottleneckResult
from benchmark.analysis.cost_estimator import CostEstimate, CostEstimator
from benchmark.analysis.pareto import ParetoAnalyser
from benchmark.analysis.recommender import RecommendationSynthesiser
from benchmark.analysis.regression import RegressionDetector, RegressionReport
from benchmark.analysis.slo_evaluator import SLOEvaluator
from benchmark.config.schema import BenchmarkMetrics, BenchmarkRun
from benchmark.config.sweep import ConfigPoint, generate_full_sweep
from benchmark.config.validation import ConfigValidator, ValidationResult
from benchmark.metrics.collector import MetricsCollector, RequestTiming
from benchmark.metrics.prometheus_bridge import PrometheusBridge
from benchmark.profiler.gpu_profiler import GPUProfiler, GPUStats
from benchmark.profiler.trace_recorder import TraceRecorder
from benchmark.reporting.exporter import BenchmarkExporter
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
        model_params_gb: Approximate model parameter size in GiB (ignored when
            *model_variants* is provided).
        model_variants: Optional list of short model variant names from the
            MODEL_VARIANTS registry.  When provided the sweep iterates over
            each variant, using its ``params_gb`` for OOM checks and setting
            ``model_id`` on every emitted ConfigPoint.
        suites: Benchmark suites to run (None means all four).
        execution_mode: How to run benchmark jobs — ``"local"`` writes
            manifests / runs locally, ``"kubernetes"`` submits real K8s Jobs.
        image: Container image for Kubernetes Jobs.
        namespace: Kubernetes namespace.
        results_pvc: PersistentVolumeClaim for result storage.
        kubeconfig: Path to kubeconfig file.  ``None`` uses the in-cluster
            default or ``$KUBECONFIG``.
        service_account: Kubernetes ServiceAccount for Job pods.
        node_selector: Optional node selector labels for K8s Job pods.
        tolerations: Optional tolerations for K8s Job pods.
        job_timeout_seconds: Maximum wall-clock seconds to wait for a single
            Kubernetes Job before marking it FAILED.
        results_dir: Local results directory (used in dry_run / local mode).
        dry_run: If True, skip actual Job submission.
        max_parallel_jobs: Maximum concurrent benchmark jobs.
        throughput_floor_tps: Minimum TPS for early stopping.
        seed: Random seed for workload generation.
        validate_configs: If True, run ConfigValidator before submission.
        strict_validation: If True, treat config warnings as errors.
        gpu_instance: GPU instance type name for cost estimation.
        baseline_path: Path to a saved JSON baseline for regression detection.
        enable_gpu_profiler: Whether to run GPU hardware profiling.
        enable_trace_recorder: Whether to record per-request traces.
        export_formats: List of formats to export (json, csv, markdown, html).
        export_prefix: Filename prefix for exported reports.
    """

    max_gpus: int = 8
    model_params_gb: float = 14.0
    model_variants: Optional[List[str]] = None
    suites: Optional[List[str]] = None
    execution_mode: str = "local"  # "local" or "kubernetes"
    image: str = "vllm/vllm-openai:latest"
    namespace: str = "benchmark"
    results_pvc: str = "benchmark-results"
    kubeconfig: Optional[str] = None
    service_account: Optional[str] = None
    node_selector: Optional[Dict[str, str]] = None
    tolerations: Optional[List[Dict[str, str]]] = None
    job_timeout_seconds: int = 3600
    results_dir: str = "/tmp/benchmark-results"
    dry_run: bool = True
    max_parallel_jobs: int = 8
    throughput_floor_tps: float = 1.0
    seed: int = 42

    # New profiler options
    validate_configs: bool = True
    strict_validation: bool = False
    gpu_instance: str = "a100_sxm4_80gb"
    baseline_path: Optional[str] = None
    enable_gpu_profiler: bool = False
    enable_trace_recorder: bool = False
    export_formats: List[str] = field(
        default_factory=lambda: ["json", "csv", "markdown", "html"]
    )
    export_prefix: str = "benchmark"
    pushgateway_url: Optional[str] = None
    prometheus_service_name: str = "vllm-benchmark"
    prometheus_service_port: int = 8000


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class BenchmarkPipeline:
    """End-to-end benchmark pipeline implementing all pipeline stages.

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
            execution_mode=self.config.execution_mode,
            kubeconfig=self.config.kubeconfig,
            service_account=self.config.service_account,
            node_selector=self.config.node_selector,
            tolerations=self.config.tolerations,
            job_timeout_seconds=self.config.job_timeout_seconds,
        )
        self._results: List[Tuple[ConfigPoint, BenchmarkMetrics]] = []
        self._validation_results: List[Tuple[ConfigPoint, ValidationResult]] = []
        self._bottleneck_results: List[BottleneckResult] = []
        self._cost_estimates: List[CostEstimate] = []
        self._regression_reports: List[RegressionReport] = []
        self._gpu_profiler: Optional[GPUProfiler] = (
            GPUProfiler() if self.config.enable_gpu_profiler else None
        )
        self._trace_recorder: Optional[TraceRecorder] = (
            TraceRecorder() if self.config.enable_trace_recorder else None
        )
        self._prometheus_bridge: Optional[PrometheusBridge] = None
        if self.config.execution_mode == "kubernetes":
            self._prometheus_bridge = PrometheusBridge(
                execution_mode="kubernetes",
                namespace=self.config.namespace,
                service_name=self.config.prometheus_service_name,
                service_port=self.config.prometheus_service_port,
                pushgateway_url=self.config.pushgateway_url,
            )
        elif self.config.pushgateway_url:
            self._prometheus_bridge = PrometheusBridge(
                execution_mode="local",
                pushgateway_url=self.config.pushgateway_url,
            )

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
            model_variants=self.config.model_variants,
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
        """Collect and aggregate metrics for a completed run.

        In Kubernetes mode, also scrapes the Prometheus endpoint via the
        bridge and pushes aggregated metrics to the Pushgateway if
        configured.
        """
        if self._metrics_fn_override is not None:
            metrics = self._metrics_fn_override(cfg)
        else:
            # In production: load the BenchmarkRun JSON written by the vLLM
            # container and populate a MetricsCollector from it.
            run = self._runner.load_result(run_result.run_id)
            if run is not None:
                metrics = run.metrics
            else:
                # Fallback: return zero metrics (run not yet finished or dry_run).
                logger.debug(
                    "No result found for run_id=%s; returning zero metrics",
                    run_result.run_id,
                )
                metrics = BenchmarkMetrics()

        # Prometheus enrichment + push (Kubernetes mode)
        if self._prometheus_bridge is not None:
            snapshot = self._prometheus_bridge.scrape(run_result.run_id)
            if snapshot.values:
                metrics = self._prometheus_bridge.enrich_metrics(
                    metrics, snapshot
                )
            self._prometheus_bridge.push_metrics(
                run_result.run_id, metrics
            )

        return metrics

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
    # Stage 8 — Config validation
    # ------------------------------------------------------------------

    def stage8_validate_configs(
        self, configs: List[ConfigPoint]
    ) -> List[ConfigPoint]:
        """Validate all configs and optionally filter out invalid ones.

        Args:
            configs: Candidate ConfigPoint list from Stage 1.

        Returns:
            Filtered list of configs that pass validation.
        """
        validator = ConfigValidator(strict=self.config.strict_validation)
        valid_configs: List[ConfigPoint] = []
        for cfg in configs:
            result = validator.validate(cfg)
            self._validation_results.append((cfg, result))
            if result.is_valid:
                valid_configs.append(cfg)
            else:
                logger.warning(
                    "Stage 8: config tp=%d pp=%d suite=%s failed validation: %s",
                    cfg.tp, cfg.pp, cfg.benchmark_suite,
                    "; ".join(str(e) for e in result.errors),
                )
        logger.info(
            "Stage 8: %d / %d configs passed validation",
            len(valid_configs), len(configs),
        )
        return valid_configs

    # ------------------------------------------------------------------
    # Stage 9 — Bottleneck analysis
    # ------------------------------------------------------------------

    def stage9_analyse_bottlenecks(self) -> List[BottleneckResult]:
        """Classify bottlenecks for all collected results.

        Returns:
            List of BottleneckResult, one per result.
        """
        analyser = BottleneckAnalyser()
        self._bottleneck_results = analyser.analyse_batch(self._results)
        bottlenecked = sum(
            1 for r in self._bottleneck_results if r.bottlenecks
        )
        logger.info(
            "Stage 9: bottlenecks detected in %d / %d configs",
            bottlenecked, len(self._bottleneck_results),
        )
        return self._bottleneck_results

    # ------------------------------------------------------------------
    # Stage 10 — Cost estimation
    # ------------------------------------------------------------------

    def stage10_estimate_costs(self) -> List[CostEstimate]:
        """Estimate GPU cloud costs for all results.

        Returns:
            List of CostEstimate instances.
        """
        estimator = CostEstimator(gpu_instance_name=self.config.gpu_instance)
        self._cost_estimates = estimator.estimate_batch(self._results)
        cheapest = estimator.cheapest(self._cost_estimates)
        if cheapest:
            logger.info(
                "Stage 10: cheapest config — %s", cheapest.summary()
            )
        return self._cost_estimates

    # ------------------------------------------------------------------
    # Stage 11 — Regression detection
    # ------------------------------------------------------------------

    def stage11_detect_regressions(self) -> List[RegressionReport]:
        """Compare results against a saved baseline if configured.

        Returns:
            List of RegressionReport (empty if no baseline_path set).
        """
        if not self.config.baseline_path:
            logger.debug("Stage 11: no baseline_path set; skipping regression check")
            return []

        detector = RegressionDetector()
        try:
            self._regression_reports = detector.compare_from_file(
                self._results, self.config.baseline_path
            )
            if detector.any_regression(self._regression_reports):
                logger.warning(
                    "Stage 11: REGRESSIONS DETECTED — see regression report"
                )
            else:
                logger.info("Stage 11: no regressions detected vs baseline")
        except FileNotFoundError:
            logger.warning(
                "Stage 11: baseline file not found at %s; skipping",
                self.config.baseline_path,
            )
        return self._regression_reports

    # ------------------------------------------------------------------
    # Stage 12 — Report export
    # ------------------------------------------------------------------

    def stage12_export_reports(self, report_text: str) -> Dict[str, str]:
        """Export results to all configured formats.

        Args:
            report_text: Recommendation report text from Stage 7.

        Returns:
            Dict mapping format name to output file path.
        """
        exporter = BenchmarkExporter(self._results, report_text)
        output_dir = self.config.results_dir
        paths = exporter.write_all(output_dir, prefix=self.config.export_prefix)
        logger.info(
            "Stage 12: reports written — %s",
            {fmt: p for fmt, p in paths.items()},
        )
        return paths

    # ------------------------------------------------------------------
    # Stage 13 — GPU profiler
    # ------------------------------------------------------------------

    def stage13_start_gpu_profiling(self) -> None:
        """Start GPU hardware metric sampling (if enabled)."""
        if self._gpu_profiler is not None:
            self._gpu_profiler.start_background_sampling()
            logger.info("Stage 13: GPU profiler started")

    def stage13_stop_gpu_profiling(self) -> Dict[int, GPUStats]:
        """Stop GPU profiling and return aggregated stats.

        Returns:
            Dict mapping GPU index to GPUStats.
        """
        if self._gpu_profiler is None:
            return {}
        self._gpu_profiler.stop_background_sampling()
        stats = self._gpu_profiler.get_stats()
        for gpu_idx, s in stats.items():
            logger.info("Stage 13 GPU[%d]: %s", gpu_idx, s.summary())
        return stats

    # ------------------------------------------------------------------
    # Stage 14 — Trace recorder export
    # ------------------------------------------------------------------

    def stage14_export_traces(self) -> None:
        """Export recorded traces to Chrome TEF, folded stacks, and OTLP.

        Writes to the configured results_dir.
        """
        if self._trace_recorder is None or len(self._trace_recorder) == 0:
            logger.debug("Stage 14: no traces recorded; skipping export")
            return

        os.makedirs(self.config.results_dir, exist_ok=True)
        prefix = os.path.join(self.config.results_dir, self.config.export_prefix)

        # Chrome trace
        chrome_path = f"{prefix}_trace.json"
        self._trace_recorder.export_chrome_trace_json(chrome_path)
        logger.info("Stage 14: Chrome trace written to %s", chrome_path)

        # Folded stacks
        stacks_path = f"{prefix}_stacks.txt"
        self._trace_recorder.export_folded_stacks_file(stacks_path)
        logger.info("Stage 14: folded stacks written to %s", stacks_path)

        # OTLP spans
        otlp_path = f"{prefix}_otlp.json"
        self._trace_recorder.export_otlp_json(otlp_path)
        logger.info("Stage 14: OTLP spans written to %s", otlp_path)

        # Summary stats
        stats = self._trace_recorder.summary_stats()
        if stats:
            logger.info("Stage 14: trace summary — %s", stats)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute the full pipeline.

        Returns:
            Rendered recommendation report text.
        """
        # Stage 1 — generate configs
        configs = self.stage1_generate_configs()
        if not configs:
            return "No feasible configs generated."

        # Stage 8 — validate configs (optional, runs before submission)
        if self.config.validate_configs:
            configs = self.stage8_validate_configs(configs)
            if not configs:
                return "All configs failed validation."

        # Stage 13 — start GPU profiling
        self.stage13_start_gpu_profiling()

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

        # Stage 13 — stop GPU profiling
        self.stage13_stop_gpu_profiling()

        # Stage 14 — export traces
        self.stage14_export_traces()

        # Stage 6 — Pareto analysis
        self.stage6_pareto_analysis()

        # Stage 7 — recommendations
        report = self.stage7_recommend()

        # Stage 9 — bottleneck analysis
        self.stage9_analyse_bottlenecks()

        # Stage 10 — cost estimation
        self.stage10_estimate_costs()

        # Stage 11 — regression detection
        self.stage11_detect_regressions()

        # Persist plain text report (backward compat)
        os.makedirs(self.config.results_dir, exist_ok=True)
        report_path = os.path.join(self.config.results_dir, "report.txt")
        with open(report_path, "w") as fh:
            fh.write(report)
        logger.info("Report written to %s", report_path)

        # Stage 12 — export all formats
        self.stage12_export_reports(report)

        return report
