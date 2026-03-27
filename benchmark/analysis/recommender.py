"""
Stage 7 — Recommendation synthesiser.

Explains tradeoffs across all benchmark suites in plain text.  Maps
configs to their paper-equivalent results for cross-validation.  Flags
configs where simulation deviates from paper-reported numbers.  Recommends
the best config per optimization target.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from benchmark.analysis.pareto import ParetoAnalyser, ParetoPoint
from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Optimization targets (Section 3, Stage 7)
# ---------------------------------------------------------------------------

OPTIMIZATION_TARGETS = {
    "min_ttft_under_slo": "Minimize TTFT under SLO (DistServe target)",
    "max_decode_throughput": "Maximize decode throughput (SARATHI target)",
    "max_offline_throughput": "Maximize offline throughput (Seesaw target)",
    "min_cost_per_query": "Minimize cost per query (general target)",
}

# Paper-reported reference numbers for cross-validation.
# Format: { description: (metric_attr, expected_value, tolerance_pct) }
_PAPER_REFERENCES: Dict[str, Tuple[str, float, float]] = {
    "DistServe Fig-8 goodput@TP4PP1": ("goodput_rps", 7.0, 20.0),
    "SARATHI-Serve decode speedup TP4": ("decode_speedup_vs_baseline", 2.4, 25.0),
    "vLLM TP8 throughput sharegpt": ("throughput_tps", 18000.0, 30.0),
}


# ---------------------------------------------------------------------------
# Deviation checker
# ---------------------------------------------------------------------------

@dataclass
class DeviationFlag:
    """A flag raised when a measured metric deviates from paper reference."""

    description: str
    expected: float
    measured: float
    tolerance_pct: float

    @property
    def deviation_pct(self) -> float:
        if self.expected == 0:
            return 0.0
        return abs(self.measured - self.expected) / self.expected * 100.0

    @property
    def is_flagged(self) -> bool:
        return self.deviation_pct > self.tolerance_pct

    def __str__(self) -> str:
        direction = "above" if self.measured > self.expected else "below"
        return (
            f"[DEVIATION] {self.description}: measured={self.measured:.2f}, "
            f"expected={self.expected:.2f} ({self.deviation_pct:.1f}% "
            f"{direction}, tolerance={self.tolerance_pct:.0f}%)"
        )


def check_deviations(
    metrics: BenchmarkMetrics,
    references: Optional[Dict[str, Tuple[str, float, float]]] = None,
) -> List[DeviationFlag]:
    """Check metrics against paper-reported reference values.

    Args:
        metrics: Measured BenchmarkMetrics for a run.
        references: Dict mapping description to (metric_attr, expected, tol%).
            Defaults to the built-in _PAPER_REFERENCES table.

    Returns:
        List of DeviationFlag instances (only flagged ones are non-empty).
    """
    refs = references or _PAPER_REFERENCES
    flags = []
    for desc, (attr, expected, tolerance) in refs.items():
        measured = getattr(metrics, attr, 0.0) or 0.0
        flag = DeviationFlag(
            description=desc,
            expected=expected,
            measured=measured,
            tolerance_pct=tolerance,
        )
        if flag.is_flagged:
            flags.append(flag)
    return flags


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """A recommendation for a given optimization target."""

    target_key: str
    target_label: str
    config: ConfigPoint
    metrics: BenchmarkMetrics
    rationale: str
    deviation_flags: List[DeviationFlag] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RecommendationSynthesiser
# ---------------------------------------------------------------------------

class RecommendationSynthesiser:
    """Synthesises recommendations from Pareto-optimal results.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples for all runs.
    """

    def __init__(
        self,
        results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    ) -> None:
        self.results = results
        self._analyser = ParetoAnalyser(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_distserve(self) -> Optional[ParetoPoint]:
        return self._analyser.best_for_suite("distserve", objective_index=0)

    def _best_sarathi(self) -> Optional[ParetoPoint]:
        return self._analyser.best_for_suite("sarathi", objective_index=0)

    def _best_seesaw(self) -> Optional[ParetoPoint]:
        return self._analyser.best_for_suite("seesaw", objective_index=1)

    def _best_general(self) -> Optional[ParetoPoint]:
        return self._analyser.best_for_suite("vllm_parallelism", objective_index=0)

    def _config_summary(self, cfg: ConfigPoint) -> str:
        parts = [
            f"TP={cfg.tp}",
            f"PP={cfg.pp}",
            f"DP={cfg.dp}",
            f"dataset={cfg.dataset}",
            f"chunked_prefill={cfg.chunked_prefill}",
            f"chunk_size={cfg.chunk_size}",
            f"disaggregation_mode={cfg.disaggregation_mode}",
        ]
        if cfg.disaggregation_mode in ("distserve", "seesaw_resharding"):
            parts += [
                f"prefill_tp={cfg.prefill_tp}",
                f"prefill_pp={cfg.prefill_pp}",
                f"decode_tp={cfg.decode_tp}",
                f"decode_pp={cfg.decode_pp}",
            ]
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self) -> List[Recommendation]:
        """Generate one recommendation per optimization target.

        Returns:
            List of Recommendation objects (may be shorter than 4 if some
            suites have no results).
        """
        recs: List[Recommendation] = []

        # --- 1. Minimize TTFT under SLO (DistServe) ---
        pt = self._best_distserve()
        if pt:
            flags = check_deviations(pt.metrics)
            rationale = (
                f"Config {self._config_summary(pt.config)} achieves the highest "
                f"goodput/GPU of {pt.metrics.goodput_rps / max(pt.config.disaggregated_gpu_count(), 1):.2f} rps/GPU "
                f"with TTFT p90={pt.metrics.ttft_p90_ms:.1f} ms "
                f"(SLO={pt.config.ttft_slo_ms:.0f} ms) and "
                f"joint SLO attainment={pt.metrics.joint_slo_attainment_pct:.1f}%.  "
                "Disaggregated prefill-decode placement reduces TTFT by removing "
                "head-of-line blocking between long prefill and decode batches."
            )
            recs.append(
                Recommendation(
                    target_key="min_ttft_under_slo",
                    target_label=OPTIMIZATION_TARGETS["min_ttft_under_slo"],
                    config=pt.config,
                    metrics=pt.metrics,
                    rationale=rationale,
                    deviation_flags=flags,
                )
            )

        # --- 2. Maximize decode throughput (SARATHI) ---
        pt = self._best_sarathi()
        if pt:
            flags = check_deviations(pt.metrics)
            rationale = (
                f"Config {self._config_summary(pt.config)} achieves a decode "
                f"speedup of {pt.metrics.decode_speedup_vs_baseline:.2f}× over "
                "baseline with "
                f"e2e throughput={pt.metrics.end_to_end_throughput_rps:.2f} rps.  "
                f"Chunk size={pt.config.chunk_size} tokens limits prefill "
                "interference with ongoing decode microbatches, reducing "
                f"pipeline bubble ratio to {pt.metrics.pipeline_bubble_ratio:.3f}."
            )
            recs.append(
                Recommendation(
                    target_key="max_decode_throughput",
                    target_label=OPTIMIZATION_TARGETS["max_decode_throughput"],
                    config=pt.config,
                    metrics=pt.metrics,
                    rationale=rationale,
                    deviation_flags=flags,
                )
            )

        # --- 3. Maximize offline throughput (Seesaw) ---
        pt = self._best_seesaw()
        if pt:
            flags = check_deviations(pt.metrics)
            rationale = (
                f"Config {self._config_summary(pt.config)} achieves "
                f"throughput={pt.metrics.throughput_tps:.0f} tps in offline mode.  "
                f"Dynamic resharding ({pt.config.resharding_pair}) with "
                f"cpu_kv_buffer_gb={pt.config.cpu_kv_buffer_gb} GiB and "
                f"transition_policy={pt.config.transition_policy} balances "
                "prefill parallelism with decode throughput."
            )
            recs.append(
                Recommendation(
                    target_key="max_offline_throughput",
                    target_label=OPTIMIZATION_TARGETS["max_offline_throughput"],
                    config=pt.config,
                    metrics=pt.metrics,
                    rationale=rationale,
                    deviation_flags=flags,
                )
            )

        # --- 4. Minimize cost per query (general) ---
        pt = self._best_general()
        if pt:
            flags = check_deviations(pt.metrics)
            tokens_per_dollar = pt.metrics.throughput_tps  # proxy for cost
            rationale = (
                f"Config {self._config_summary(pt.config)} achieves "
                f"{tokens_per_dollar:.0f} tokens/s (proxy for cost efficiency) "
                f"with GPU memory usage of {pt.metrics.gpu_mem_used_gb:.1f} GiB.  "
                f"TP={pt.config.tp} × PP={pt.config.pp} parallelism is the "
                "most cost-efficient for this model/workload combination."
            )
            recs.append(
                Recommendation(
                    target_key="min_cost_per_query",
                    target_label=OPTIMIZATION_TARGETS["min_cost_per_query"],
                    config=pt.config,
                    metrics=pt.metrics,
                    rationale=rationale,
                    deviation_flags=flags,
                )
            )

        return recs

    def render_report(self) -> str:
        """Render a human-readable text report of all recommendations.

        Returns:
            Multi-line string suitable for logging or writing to a file.
        """
        recs = self.recommend()
        lines = [
            "=" * 72,
            "  vLLM Parallelism & Serving Config — Recommendation Report",
            "=" * 72,
            "",
        ]

        if not recs:
            lines.append("No results available.  Run the benchmark pipeline first.")
            return "\n".join(lines)

        for rec in recs:
            lines += [
                f"[{rec.target_label}]",
                "-" * 60,
                textwrap.fill(rec.rationale, width=72),
                "",
            ]
            if rec.deviation_flags:
                lines.append("  Cross-validation deviations from paper:")
                for flag in rec.deviation_flags:
                    lines.append(f"    {flag}")
                lines.append("")

        lines += [
            "=" * 72,
            "End of report",
        ]
        return "\n".join(lines)
