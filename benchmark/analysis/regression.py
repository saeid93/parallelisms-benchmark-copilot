"""
Regression detector — Stage 11.

Compares a set of *current* benchmark results against a *baseline* (a
previously saved run set or a set of reference metrics).  Flags any metric
that has regressed beyond a configurable tolerance.

Supports:
  - Absolute and relative regression thresholds
  - Per-metric direction (higher-is-better vs lower-is-better)
  - Aggregated regression score per config
  - Baseline serialisation to / from JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric direction & tolerance catalogue
# ---------------------------------------------------------------------------

# (higher_is_better, default_relative_tolerance_pct)
_METRIC_SPEC: Dict[str, Tuple[bool, float]] = {
    "throughput_tps": (True, 5.0),
    "end_to_end_throughput_rps": (True, 5.0),
    "goodput_rps": (True, 5.0),
    "ttft_p50_ms": (False, 10.0),
    "ttft_p90_ms": (False, 10.0),
    "ttft_p99_ms": (False, 15.0),
    "tpot_p50_ms": (False, 10.0),
    "tpot_p90_ms": (False, 10.0),
    "tpot_p99_ms": (False, 15.0),
    "e2e_latency_p50_ms": (False, 10.0),
    "e2e_latency_p90_ms": (False, 10.0),
    "e2e_latency_p99_ms": (False, 15.0),
    "joint_slo_attainment_pct": (True, 3.0),
    "ttft_slo_attainment_pct": (True, 3.0),
    "tpot_slo_attainment_pct": (True, 3.0),
    "gpu_mem_used_gb": (False, 5.0),
    "kv_cache_hit_rate": (True, 5.0),
    "preemption_rate": (False, 10.0),
    "pipeline_bubble_ratio": (False, 5.0),
    "decode_speedup_vs_baseline": (True, 5.0),
}


# ---------------------------------------------------------------------------
# Single-metric regression result
# ---------------------------------------------------------------------------

@dataclass
class MetricRegression:
    """A single metric that has regressed."""

    metric_name: str
    baseline_value: float
    current_value: float
    tolerance_pct: float
    higher_is_better: bool

    @property
    def delta_pct(self) -> float:
        """Percentage change from baseline (positive = improvement)."""
        if self.baseline_value == 0.0:
            return 0.0
        raw = (self.current_value - self.baseline_value) / abs(self.baseline_value) * 100.0
        return raw if self.higher_is_better else -raw

    @property
    def is_regression(self) -> bool:
        """True when the change exceeds the allowed tolerance in the wrong direction."""
        return self.delta_pct < -self.tolerance_pct

    def __str__(self) -> str:
        direction = "↑ better" if self.higher_is_better else "↓ lower is better"
        change = f"{self.delta_pct:+.2f}%"
        label = "REGRESSION" if self.is_regression else "ok"
        return (
            f"[{label}] {self.metric_name}: "
            f"baseline={self.baseline_value:.4g} → current={self.current_value:.4g} "
            f"({change}, tol={self.tolerance_pct:.1f}%, {direction})"
        )


# ---------------------------------------------------------------------------
# Per-run comparison result
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    """Comparison result for a single config against its baseline."""

    config_key: str
    regressions: List[MetricRegression] = field(default_factory=list)
    improvements: List[MetricRegression] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return any(m.is_regression for m in self.regressions)

    @property
    def regression_score(self) -> float:
        """Aggregate regression severity: sum of |delta_pct| for regressions."""
        return sum(abs(m.delta_pct) for m in self.regressions if m.is_regression)

    def summary(self) -> str:
        lines = [f"Config: {self.config_key}"]
        if not self.regressions and not self.improvements:
            lines.append("  No changes vs baseline.")
        for m in self.regressions:
            lines.append(f"  {m}")
        for m in self.improvements:
            lines.append(f"  {m}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RegressionDetector
# ---------------------------------------------------------------------------

class RegressionDetector:
    """Compares current benchmark results to a saved baseline.

    Args:
        tolerance_overrides: Dict mapping metric name to a custom tolerance
            percentage, overriding the defaults in _METRIC_SPEC.
    """

    def __init__(
        self,
        tolerance_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._tol_overrides = tolerance_overrides or {}

    # ------------------------------------------------------------------
    # Config key generation
    # ------------------------------------------------------------------

    @staticmethod
    def config_key(cfg: ConfigPoint) -> str:
        """Generate a stable string key for a ConfigPoint.

        Used to match current results to baseline entries.

        Args:
            cfg: Configuration point.

        Returns:
            Compact colon-separated key string.
        """
        parts = [
            f"suite={cfg.benchmark_suite}",
            f"tp={cfg.tp}",
            f"pp={cfg.pp}",
            f"dp={cfg.dp}",
            f"dataset={cfg.dataset}",
            f"disagg={cfg.disaggregation_mode}",
            f"chunk={cfg.chunk_size}",
            f"batching={cfg.batching_scheme}",
        ]
        return ":".join(parts)

    # ------------------------------------------------------------------
    # Baseline I/O
    # ------------------------------------------------------------------

    def save_baseline(
        self,
        results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
        path: str,
    ) -> None:
        """Serialise a result set to JSON for later comparison.

        Args:
            results: List of (ConfigPoint, BenchmarkMetrics) tuples.
            path: File path to write the JSON baseline.
        """
        baseline: Dict[str, Dict] = {}
        for cfg, metrics in results:
            key = self.config_key(cfg)
            baseline[key] = metrics.model_dump()
        with open(path, "w") as fh:
            json.dump(baseline, fh, indent=2)
        logger.info("Saved baseline with %d entries to %s", len(baseline), path)

    def load_baseline(self, path: str) -> Dict[str, BenchmarkMetrics]:
        """Load a baseline from JSON.

        Args:
            path: File path of the JSON baseline.

        Returns:
            Dict mapping config key to BenchmarkMetrics.
        """
        with open(path) as fh:
            raw = json.load(fh)
        return {
            key: BenchmarkMetrics.model_validate(val)
            for key, val in raw.items()
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _compare_metrics(
        self,
        baseline: BenchmarkMetrics,
        current: BenchmarkMetrics,
    ) -> Tuple[List[MetricRegression], List[MetricRegression]]:
        regressions: List[MetricRegression] = []
        improvements: List[MetricRegression] = []

        for metric_name, (higher_is_better, default_tol) in _METRIC_SPEC.items():
            tol = self._tol_overrides.get(metric_name, default_tol)
            baseline_val = getattr(baseline, metric_name, 0.0) or 0.0
            current_val = getattr(current, metric_name, 0.0) or 0.0

            if baseline_val == 0.0 and current_val == 0.0:
                continue

            mr = MetricRegression(
                metric_name=metric_name,
                baseline_value=baseline_val,
                current_value=current_val,
                tolerance_pct=tol,
                higher_is_better=higher_is_better,
            )
            if mr.is_regression:
                regressions.append(mr)
            elif mr.delta_pct > tol:
                improvements.append(mr)

        return regressions, improvements

    def compare(
        self,
        current_results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
        baseline: Dict[str, BenchmarkMetrics],
    ) -> List[RegressionReport]:
        """Compare current results against a baseline dict.

        Args:
            current_results: List of (ConfigPoint, BenchmarkMetrics) from the
                current benchmark run.
            baseline: Dict mapping config key to baseline BenchmarkMetrics,
                as returned by load_baseline().

        Returns:
            List of RegressionReport, one per matched config.  Configs with
            no baseline entry are skipped.
        """
        reports: List[RegressionReport] = []
        for cfg, metrics in current_results:
            key = self.config_key(cfg)
            if key not in baseline:
                logger.debug("No baseline entry for %s; skipping", key)
                continue
            regressions, improvements = self._compare_metrics(
                baseline[key], metrics
            )
            report = RegressionReport(
                config_key=key,
                regressions=regressions,
                improvements=improvements,
            )
            reports.append(report)
        return reports

    def compare_from_file(
        self,
        current_results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
        baseline_path: str,
    ) -> List[RegressionReport]:
        """Load baseline from disk and compare against current results.

        Args:
            current_results: List of (ConfigPoint, BenchmarkMetrics) tuples.
            baseline_path: Path to the JSON baseline file.

        Returns:
            List of RegressionReport instances.
        """
        baseline = self.load_baseline(baseline_path)
        return self.compare(current_results, baseline)

    def any_regression(self, reports: List[RegressionReport]) -> bool:
        """Return True if any report contains a regression.

        Args:
            reports: List of RegressionReport instances.

        Returns:
            True if at least one regression was found.
        """
        return any(r.has_regression for r in reports)

    def render_report(self, reports: List[RegressionReport]) -> str:
        """Render a human-readable regression report.

        Args:
            reports: List of RegressionReport instances.

        Returns:
            Multi-line string.
        """
        lines = [
            "=" * 72,
            "  Regression Report",
            "=" * 72,
            "",
        ]
        flagged = [r for r in reports if r.has_regression]
        if not flagged:
            lines.append("No regressions detected across all compared configs.")
        else:
            lines.append(
                f"REGRESSIONS DETECTED in {len(flagged)} / {len(reports)} configs:"
            )
            lines.append("")
            for report in sorted(flagged, key=lambda r: -r.regression_score):
                lines.append(report.summary())
                lines.append("")
        lines += ["=" * 72, "End of regression report"]
        return "\n".join(lines)
