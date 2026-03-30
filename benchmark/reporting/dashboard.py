"""Dashboard data generator for interactive visualisation.

Generates structured JSON data for web dashboards (Grafana,
custom React/Vue apps, Jupyter notebooks). Provides time-series,
heatmaps, correlation matrices, and histogram data.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesPoint:
    """A single point in a time-series."""

    timestamp: float  # seconds
    value: float
    label: str = ""


@dataclass
class DashboardPanel:
    """A single dashboard panel."""

    panel_id: str
    title: str
    panel_type: str  # "timeseries", "heatmap", "histogram", "correlation", "table", "stat", "scatter"
    data: Any = None
    description: str = ""
    unit: str = ""


@dataclass
class DashboardData:
    """Complete dashboard export."""

    title: str = "Benchmark Dashboard"
    panels: List[DashboardPanel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), indent=2, default=str)

    def _to_dict(self) -> Dict:
        return {
            "title": self.title,
            "metadata": self.metadata,
            "panels": [self._panel_dict(p) for p in self.panels],
        }

    @staticmethod
    def _panel_dict(panel: DashboardPanel) -> Dict:
        return {
            "id": panel.panel_id,
            "title": panel.title,
            "type": panel.panel_type,
            "description": panel.description,
            "unit": panel.unit,
            "data": panel.data,
        }


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

_KEY_METRICS = [
    "throughput_tps", "goodput_rps", "ttft_p50_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p99_ms", "e2e_latency_p50_ms", "e2e_latency_p99_ms",
    "joint_slo_attainment_pct", "gpu_mem_used_gb", "kv_cache_hit_rate",
    "preemption_rate", "bubble_ratio",
]

_CONFIG_PARAMS = [
    "tp", "pp", "dp", "max_batched_tokens", "max_num_seqs",
    "gpu_mem_util", "block_size", "dtype", "quantization", "kv_dtype",
]


def _extract(obj: Any, name: str) -> float:
    val = getattr(obj, name, None)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _pearson_corr(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _compute_histogram(values: List[float], num_bins: int = 20) -> List[Dict]:
    """Compute histogram bins from values."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mn == mx:
        return [{"bin_start": mn, "bin_end": mx, "count": len(values), "density": 1.0}]
    bin_width = (mx - mn) / num_bins
    bins: List[Dict] = []
    for i in range(num_bins):
        start = mn + i * bin_width
        end = start + bin_width
        count = sum(1 for v in values if start <= v < end)
        if i == num_bins - 1:
            count = sum(1 for v in values if start <= v <= end)
        density = count / (len(values) * bin_width) if bin_width > 0 else 0.0
        bins.append({
            "bin_start": round(start, 4),
            "bin_end": round(end, 4),
            "count": count,
            "density": round(density, 6),
        })
    return bins


# ---------------------------------------------------------------------------
# Dashboard Generator
# ---------------------------------------------------------------------------


class DashboardGenerator:
    """Generates structured dashboard data from benchmark results.

    Usage::

        gen = DashboardGenerator()
        gen.add_result("run-1", cfg, metrics)
        gen.add_timeseries("gpu_util", [(t, val), ...])
        dashboard = gen.generate()
        print(dashboard.to_json())
    """

    def __init__(self) -> None:
        self._results: List[Tuple[str, ConfigPoint, BenchmarkMetrics]] = []
        self._timeseries: Dict[str, List[TimeSeriesPoint]] = {}

    def add_result(
        self, label: str, cfg: ConfigPoint, metrics: BenchmarkMetrics
    ) -> None:
        self._results.append((label, cfg, metrics))

    def add_timeseries(
        self, metric_name: str, points: List[Tuple[float, float]]
    ) -> None:
        """Add time-series data for a metric."""
        self._timeseries.setdefault(metric_name, []).extend(
            TimeSeriesPoint(timestamp=t, value=v) for t, v in points
        )

    def _throughput_latency_panel(self) -> DashboardPanel:
        """Throughput vs latency scatter plot data."""
        data = []
        for label, cfg, metrics in self._results:
            data.append({
                "label": label,
                "throughput_tps": _extract(metrics, "throughput_tps"),
                "ttft_p99_ms": _extract(metrics, "ttft_p99_ms"),
                "tpot_p99_ms": _extract(metrics, "tpot_p99_ms"),
                "config": {
                    "tp": getattr(cfg, "tp", 0),
                    "pp": getattr(cfg, "pp", 0),
                    "dp": getattr(cfg, "dp", 0),
                },
            })
        return DashboardPanel(
            panel_id="throughput_vs_latency",
            title="Throughput vs Latency",
            panel_type="scatter",
            data=data,
            description="Each point is a config. X=throughput, Y=p99 latency.",
        )

    def _config_heatmap_panel(self) -> DashboardPanel:
        """Configuration parameter heatmap (TP x PP -> throughput)."""
        cells: List[Dict] = []
        for label, cfg, metrics in self._results:
            tp = getattr(cfg, "tp", 1)
            pp = getattr(cfg, "pp", 1)
            cells.append({
                "x": f"TP={tp}",
                "y": f"PP={pp}",
                "value": _extract(metrics, "throughput_tps"),
                "label": label,
            })
        return DashboardPanel(
            panel_id="config_heatmap",
            title="TP x PP Throughput Heatmap",
            panel_type="heatmap",
            data=cells,
            unit="tok/s",
        )

    def _metric_distribution_panel(self) -> DashboardPanel:
        """Distribution of key metrics across configs."""
        distributions: Dict[str, Dict] = {}
        for metric in _KEY_METRICS:
            values = [_extract(m, metric) for _, _, m in self._results if _extract(m, metric) > 0]
            if not values:
                continue
            distributions[metric] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "std": round(statistics.stdev(values) if len(values) > 1 else 0, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "histogram": _compute_histogram(values),
            }
        return DashboardPanel(
            panel_id="metric_distributions",
            title="Metric Distributions",
            panel_type="histogram",
            data=distributions,
        )

    def _correlation_matrix_panel(self) -> DashboardPanel:
        """Correlation matrix between metrics."""
        metric_values: Dict[str, List[float]] = {}
        for metric in _KEY_METRICS:
            vals = [_extract(m, metric) for _, _, m in self._results]
            if any(v > 0 for v in vals):
                metric_values[metric] = vals

        correlations: List[Dict] = []
        metrics_list = list(metric_values.keys())
        for i, ma in enumerate(metrics_list):
            for j, mb in enumerate(metrics_list):
                if i <= j:
                    corr = _pearson_corr(metric_values[ma], metric_values[mb])
                    correlations.append({
                        "metric_a": ma,
                        "metric_b": mb,
                        "correlation": round(corr, 3),
                    })

        return DashboardPanel(
            panel_id="correlation_matrix",
            title="Metric Correlation Matrix",
            panel_type="correlation",
            data=correlations,
        )

    def _summary_stats_panel(self) -> DashboardPanel:
        """Key summary statistics."""
        stats: Dict[str, Any] = {
            "total_configs": len(self._results),
        }
        if self._results:
            tps = [_extract(m, "throughput_tps") for _, _, m in self._results]
            best_idx = tps.index(max(tps))
            stats["best_throughput_tps"] = round(max(tps), 2)
            stats["best_config"] = self._results[best_idx][0]
            stats["mean_throughput_tps"] = round(statistics.mean(tps), 2)
            slos = [_extract(m, "joint_slo_attainment_pct") for _, _, m in self._results]
            stats["mean_slo_attainment"] = round(statistics.mean(slos), 2) if slos else 0
            stats["configs_meeting_slo"] = sum(1 for s in slos if s >= 90)

        return DashboardPanel(
            panel_id="summary_stats",
            title="Summary Statistics",
            panel_type="stat",
            data=stats,
        )

    def _timeseries_panels(self) -> List[DashboardPanel]:
        """Generate panels for recorded time-series data."""
        panels: List[DashboardPanel] = []
        for metric_name, points in self._timeseries.items():
            data = [{"timestamp": p.timestamp, "value": p.value, "label": p.label} for p in points]
            panels.append(
                DashboardPanel(
                    panel_id=f"ts_{metric_name}",
                    title=f"Time Series: {metric_name}",
                    panel_type="timeseries",
                    data=data,
                )
            )
        return panels

    def _results_table_panel(self) -> DashboardPanel:
        """Full results table."""
        rows: List[Dict] = []
        for label, cfg, metrics in self._results:
            row: Dict[str, Any] = {"label": label}
            for p in _CONFIG_PARAMS:
                row[p] = getattr(cfg, p, "")
            for m in _KEY_METRICS:
                row[m] = round(_extract(metrics, m), 3)
            rows.append(row)

        return DashboardPanel(
            panel_id="results_table",
            title="All Results",
            panel_type="table",
            data=rows,
        )

    def _slo_attainment_panel(self) -> DashboardPanel:
        """SLO attainment bar chart data."""
        data = []
        for label, cfg, metrics in self._results:
            data.append({
                "label": label,
                "ttft_slo_pct": _extract(metrics, "ttft_slo_attainment_pct"),
                "tpot_slo_pct": _extract(metrics, "tpot_slo_attainment_pct"),
                "joint_slo_pct": _extract(metrics, "joint_slo_attainment_pct"),
            })
        return DashboardPanel(
            panel_id="slo_attainment",
            title="SLO Attainment",
            panel_type="bar",
            data=data,
            unit="%",
        )

    def _memory_panel(self) -> DashboardPanel:
        """GPU memory usage comparison."""
        data = []
        for label, cfg, metrics in self._results:
            data.append({
                "label": label,
                "gpu_mem_used_gb": _extract(metrics, "gpu_mem_used_gb"),
                "kv_cache_hit_rate": _extract(metrics, "kv_cache_hit_rate"),
                "preemption_rate": _extract(metrics, "preemption_rate"),
            })
        return DashboardPanel(
            panel_id="memory_usage",
            title="Memory & Cache",
            panel_type="bar",
            data=data,
        )

    def generate(self, title: str = "Benchmark Dashboard") -> DashboardData:
        """Generate complete dashboard data."""
        panels: List[DashboardPanel] = []

        panels.append(self._summary_stats_panel())
        panels.append(self._throughput_latency_panel())
        panels.append(self._slo_attainment_panel())
        panels.append(self._config_heatmap_panel())
        panels.append(self._metric_distribution_panel())
        panels.append(self._memory_panel())
        panels.append(self._correlation_matrix_panel())
        panels.append(self._results_table_panel())
        panels.extend(self._timeseries_panels())

        metadata = {
            "generated_by": "parallelisms-benchmark-copilot",
            "num_configs": len(self._results),
            "num_timeseries": len(self._timeseries),
        }

        return DashboardData(
            title=title,
            panels=panels,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._results.clear()
        self._timeseries.clear()
