"""Comparative benchmark reporter for side-by-side analysis.

Inspired by:
* MLPerf Inference reporting format
* "A Fair Comparison of Graph Neural Networks for Graph Classification"
* LLM serving benchmark comparison dashboards
* Database benchmark TPC-style comparison reports

Generates side-by-side configuration comparisons, delta analysis,
radar chart data, automated insight generation, and executive
summaries with natural language explanations.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class MetricDelta:
    """Change in a metric between two configurations."""

    metric_name: str
    value_a: float
    value_b: float
    absolute_delta: float
    relative_delta_pct: float
    is_improvement: bool  # based on metric direction
    significance: str = ""  # "significant", "marginal", "negligible"


@dataclass
class ConfigComparison:
    """Side-by-side comparison of two configurations."""

    config_a_label: str
    config_b_label: str
    config_a: Dict[str, Any] = field(default_factory=dict)
    config_b: Dict[str, Any] = field(default_factory=dict)
    deltas: List[MetricDelta] = field(default_factory=list)
    config_diffs: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    winner: str = ""
    win_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Comparison: {self.config_a_label} vs {self.config_b_label}",
            f"  Winner: {self.winner}",
        ]
        if self.config_diffs:
            lines.append("  Config differences:")
            for param, (va, vb) in self.config_diffs.items():
                lines.append(f"    {param}: {va} → {vb}")
        lines.append("  Metric deltas:")
        for d in self.deltas:
            arrow = "↑" if d.is_improvement else "↓"
            lines.append(
                f"    {d.metric_name}: {d.value_a:.2f} → {d.value_b:.2f} "
                f"({d.relative_delta_pct:+.1f}% {arrow}) [{d.significance}]"
            )
        return "\n".join(lines)


@dataclass
class RadarChartData:
    """Data for radar/spider chart visualization."""

    labels: List[str] = field(default_factory=list)
    datasets: List[Dict] = field(default_factory=list)  # [{label, values, color}]


@dataclass
class InsightSummary:
    """Auto-generated insight about the benchmark results."""

    category: str  # "performance", "cost", "scalability", etc.
    title: str
    description: str
    severity: str = "info"  # "info", "warning", "critical"
    evidence: List[str] = field(default_factory=list)


@dataclass
class ExecutiveSummary:
    """Natural language executive summary of benchmark results."""

    title: str = ""
    overview: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    best_config_description: str = ""
    comparisons: List[ConfigComparison] = field(default_factory=list)
    insights: List[InsightSummary] = field(default_factory=list)
    radar_data: Optional[RadarChartData] = None

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.overview, ""]

        if self.key_findings:
            lines.append("## Key Findings")
            for i, f in enumerate(self.key_findings, 1):
                lines.append(f"{i}. {f}")
            lines.append("")

        if self.best_config_description:
            lines.append("## Best Configuration")
            lines.append(self.best_config_description)
            lines.append("")

        if self.comparisons:
            lines.append("## Configuration Comparisons")
            for comp in self.comparisons:
                lines.append(f"\n### {comp.config_a_label} vs {comp.config_b_label}")
                lines.append(f"**Winner: {comp.winner}**")
                if comp.config_diffs:
                    lines.append("\n| Parameter | Config A | Config B |")
                    lines.append("|-----------|----------|----------|")
                    for param, (va, vb) in comp.config_diffs.items():
                        lines.append(f"| {param} | {va} | {vb} |")
                lines.append("\n| Metric | Config A | Config B | Delta |")
                lines.append("|--------|----------|----------|-------|")
                for d in comp.deltas:
                    lines.append(
                        f"| {d.metric_name} | {d.value_a:.2f} | {d.value_b:.2f} | "
                        f"{d.relative_delta_pct:+.1f}% |"
                    )
            lines.append("")

        if self.insights:
            lines.append("## Insights")
            for ins in self.insights:
                lines.append(f"\n### {ins.title}")
                lines.append(ins.description)
                if ins.evidence:
                    for ev in ins.evidence:
                        lines.append(f"- {ev}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for i, r in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {r}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric direction: True = higher is better
# ---------------------------------------------------------------------------

_METRIC_DIRECTION: Dict[str, bool] = {
    "throughput_tps": True,
    "end_to_end_throughput_rps": True,
    "goodput_rps": True,
    "ttft_p50_ms": False,
    "ttft_p90_ms": False,
    "ttft_p99_ms": False,
    "tpot_p50_ms": False,
    "tpot_p90_ms": False,
    "tpot_p99_ms": False,
    "e2e_latency_p50_ms": False,
    "e2e_latency_p90_ms": False,
    "e2e_latency_p99_ms": False,
    "ttft_slo_attainment_pct": True,
    "tpot_slo_attainment_pct": True,
    "joint_slo_attainment_pct": True,
    "gpu_mem_used_gb": False,
    "kv_cache_hit_rate": True,
    "preemption_rate": False,
    "bubble_ratio": False,
    "pipeline_bubble_ratio": False,
}

_COMPARISON_METRICS = [
    "throughput_tps",
    "goodput_rps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "e2e_latency_p99_ms",
    "joint_slo_attainment_pct",
    "gpu_mem_used_gb",
    "kv_cache_hit_rate",
]

_RADAR_METRICS = [
    "throughput_tps",
    "goodput_rps",
    "joint_slo_attainment_pct",
    "kv_cache_hit_rate",
]

_RADAR_INVERSE_METRICS = [
    "ttft_p99_ms",
    "tpot_p99_ms",
    "gpu_mem_used_gb",
]


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _extract(metrics: BenchmarkMetrics, name: str) -> float:
    val = getattr(metrics, name, None)
    return float(val) if val is not None else 0.0


def _cfg_dict(cfg: ConfigPoint) -> Dict[str, Any]:
    params = [
        "tp", "pp", "dp", "max_batched_tokens", "max_num_seqs",
        "dtype", "quantization", "attention_backend", "kv_dtype",
        "gpu_mem_util", "block_size", "disaggregation_mode",
    ]
    return {p: getattr(cfg, p, None) for p in params if getattr(cfg, p, None) is not None}


class ComparativeReporter:
    """Generates comparative benchmark reports.

    Usage::

        reporter = ComparativeReporter()
        reporter.add_result("config-a", cfg_a, metrics_a)
        reporter.add_result("config-b", cfg_b, metrics_b)
        summary = reporter.generate_report()
        print(summary.to_markdown())
    """

    def __init__(self) -> None:
        self._results: Dict[str, Tuple[ConfigPoint, BenchmarkMetrics]] = {}

    def add_result(
        self, label: str, cfg: ConfigPoint, metrics: BenchmarkMetrics
    ) -> None:
        self._results[label] = (cfg, metrics)

    def compare(
        self, label_a: str, label_b: str
    ) -> Optional[ConfigComparison]:
        """Compare two specific configurations."""
        if label_a not in self._results or label_b not in self._results:
            return None

        cfg_a, met_a = self._results[label_a]
        cfg_b, met_b = self._results[label_b]

        dict_a = _cfg_dict(cfg_a)
        dict_b = _cfg_dict(cfg_b)

        # Config diffs
        diffs: Dict[str, Tuple[Any, Any]] = {}
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        for k in all_keys:
            va = dict_a.get(k)
            vb = dict_b.get(k)
            if va != vb:
                diffs[k] = (va, vb)

        # Metric deltas
        deltas: List[MetricDelta] = []
        a_wins = 0
        b_wins = 0

        for metric in _COMPARISON_METRICS:
            va = _extract(met_a, metric)
            vb = _extract(met_b, metric)
            abs_delta = vb - va
            rel_delta = 100.0 * abs_delta / va if va != 0 else 0.0
            higher_better = _METRIC_DIRECTION.get(metric, True)
            is_improvement = (abs_delta > 0) == higher_better

            if abs(rel_delta) < 2.0:
                sig = "negligible"
            elif abs(rel_delta) < 10.0:
                sig = "marginal"
            else:
                sig = "significant"

            if sig == "significant" and is_improvement:
                b_wins += 1
            elif sig == "significant" and not is_improvement:
                a_wins += 1

            deltas.append(
                MetricDelta(
                    metric_name=metric,
                    value_a=va,
                    value_b=vb,
                    absolute_delta=abs_delta,
                    relative_delta_pct=rel_delta,
                    is_improvement=is_improvement,
                    significance=sig,
                )
            )

        winner = label_b if b_wins > a_wins else label_a if a_wins > b_wins else "tie"
        reasons = [
            f"{d.metric_name}: {d.relative_delta_pct:+.1f}%"
            for d in deltas
            if d.significance == "significant"
        ]

        return ConfigComparison(
            config_a_label=label_a,
            config_b_label=label_b,
            config_a=dict_a,
            config_b=dict_b,
            deltas=deltas,
            config_diffs=diffs,
            winner=winner,
            win_reasons=reasons,
        )

    def _generate_radar_data(self) -> RadarChartData:
        """Generate radar chart data for all configs."""
        labels = [m.replace("_", " ") for m in _RADAR_METRICS + _RADAR_INVERSE_METRICS]
        datasets: List[Dict] = []

        # Normalize to 0-100 scale
        all_values: Dict[str, List[float]] = {}
        for label, (cfg, metrics) in self._results.items():
            for metric in _RADAR_METRICS + _RADAR_INVERSE_METRICS:
                all_values.setdefault(metric, []).append(_extract(metrics, metric))

        for label, (cfg, metrics) in self._results.items():
            values: List[float] = []
            for metric in _RADAR_METRICS:
                v = _extract(metrics, metric)
                max_v = max(all_values.get(metric, [1]))
                values.append(100.0 * v / max_v if max_v > 0 else 0.0)
            for metric in _RADAR_INVERSE_METRICS:
                v = _extract(metrics, metric)
                max_v = max(all_values.get(metric, [1]))
                # Invert: lower is better → higher score
                values.append(100.0 * (1.0 - v / max_v) if max_v > 0 else 0.0)
            datasets.append({"label": label, "values": values})

        return RadarChartData(labels=labels, datasets=datasets)

    def _generate_insights(self) -> List[InsightSummary]:
        """Auto-generate insights from results."""
        insights: List[InsightSummary] = []

        if len(self._results) < 2:
            return insights

        # Throughput insight
        throughputs = {
            label: _extract(met, "throughput_tps")
            for label, (cfg, met) in self._results.items()
        }
        if throughputs:
            best = max(throughputs.items(), key=lambda x: x[1])
            worst = min(throughputs.items(), key=lambda x: x[1])
            if worst[1] > 0:
                ratio = best[1] / worst[1]
                if ratio > 1.5:
                    insights.append(
                        InsightSummary(
                            category="performance",
                            title="Significant throughput variation",
                            description=(
                                f"Best config ({best[0]}) achieves {ratio:.1f}x the throughput "
                                f"of worst config ({worst[0]})."
                            ),
                            severity="warning" if ratio > 3.0 else "info",
                            evidence=[
                                f"{best[0]}: {best[1]:.1f} tok/s",
                                f"{worst[0]}: {worst[1]:.1f} tok/s",
                            ],
                        )
                    )

        # SLO insight
        slo_vals = {
            label: _extract(met, "joint_slo_attainment_pct")
            for label, (cfg, met) in self._results.items()
        }
        failing = [(l, v) for l, v in slo_vals.items() if 0 < v < 90]
        if failing:
            insights.append(
                InsightSummary(
                    category="slo",
                    title="SLO attainment below target",
                    description=f"{len(failing)} config(s) fail to meet 90% SLO target.",
                    severity="critical",
                    evidence=[f"{l}: {v:.1f}%" for l, v in failing],
                )
            )

        return insights

    def generate_report(self, title: str = "Benchmark Report") -> ExecutiveSummary:
        """Generate a full comparative report."""
        if not self._results:
            return ExecutiveSummary(title=title, overview="No results to compare.")

        # Pairwise comparisons
        labels = list(self._results.keys())
        comparisons: List[ConfigComparison] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                comp = self.compare(labels[i], labels[j])
                if comp:
                    comparisons.append(comp)

        # Best config
        best_label = max(
            self._results.items(),
            key=lambda x: _extract(x[1][1], "throughput_tps"),
        )[0]
        best_cfg, best_met = self._results[best_label]

        # Key findings
        findings: List[str] = []
        findings.append(
            f"Best throughput: **{best_label}** at {_extract(best_met, 'throughput_tps'):.1f} tok/s."
        )
        if len(self._results) > 1:
            all_tps = [_extract(m, "throughput_tps") for _, (_, m) in self._results.items()]
            findings.append(
                f"Throughput ranges from {min(all_tps):.1f} to {max(all_tps):.1f} tok/s "
                f"across {len(self._results)} configurations."
            )

        # Recommendations
        recs: List[str] = []
        if comparisons:
            for comp in comparisons:
                if comp.winner != "tie":
                    sig_deltas = [d for d in comp.deltas if d.significance == "significant"]
                    if sig_deltas:
                        recs.append(
                            f"Prefer **{comp.winner}** over the alternative "
                            f"(significant improvements in {len(sig_deltas)} metrics)."
                        )

        # Radar chart
        radar = self._generate_radar_data()

        # Insights
        insights = self._generate_insights()

        overview = (
            f"Comparative analysis of {len(self._results)} configurations "
            f"across {len(_COMPARISON_METRICS)} metrics."
        )

        return ExecutiveSummary(
            title=title,
            overview=overview,
            key_findings=findings,
            recommendations=recs,
            best_config_description=f"Config: {best_label}\n{_cfg_dict(best_cfg)}",
            comparisons=comparisons,
            insights=insights,
            radar_data=radar,
        )

    def reset(self) -> None:
        self._results.clear()
