"""
Report exporter — reporting stage.

Exports benchmark results to multiple output formats:
  - JSON       : Machine-readable full result set
  - CSV        : Flat table for spreadsheet analysis
  - Markdown   : Human-readable table for GitHub / documentation
  - HTML       : Self-contained dashboard with sortable tables and
                 inline Chart.js plots for throughput, latency, and SLO
                 attainment across all configs

The exporter operates on a list of (ConfigPoint, BenchmarkMetrics) tuples
and an optional RecommendationSynthesiser report string.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Field selectors (subset of fields written to each format)
# ---------------------------------------------------------------------------

# Config fields to include in exports
_CONFIG_FIELDS = [
    "benchmark_suite", "tp", "pp", "dp",
    "disaggregation_mode", "dataset",
    "chunked_prefill", "chunk_size", "batching_scheme",
    "prefill_tp", "prefill_pp", "decode_tp", "decode_pp",
    "kv_dtype", "gpu_mem_util", "dtype", "quantization",
    "attention_backend", "request_rate_rps", "num_requests",
    "ttft_slo_ms", "tpot_slo_ms",
]

# Metrics fields to include in exports
_METRICS_FIELDS = [
    "throughput_tps", "end_to_end_throughput_rps", "goodput_rps",
    "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
    "e2e_latency_p50_ms", "e2e_latency_p90_ms", "e2e_latency_p99_ms",
    "ttft_slo_attainment_pct", "tpot_slo_attainment_pct", "joint_slo_attainment_pct",
    "gpu_mem_used_gb", "kv_cache_hit_rate", "preemption_rate",
    "prefill_phase_time_pct", "decode_phase_time_pct", "transmission_time_pct",
    "pipeline_bubble_ratio", "decode_speedup_vs_baseline",
]


def _cfg_dict(cfg: ConfigPoint) -> Dict[str, Any]:
    """Extract selected config fields as a flat dict."""
    return {f: getattr(cfg, f, None) for f in _CONFIG_FIELDS}


def _metrics_dict(m: BenchmarkMetrics) -> Dict[str, Any]:
    """Extract selected metrics fields as a flat dict."""
    return {f: getattr(m, f, None) for f in _METRICS_FIELDS}


def _row(cfg: ConfigPoint, m: BenchmarkMetrics) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row.update(_cfg_dict(cfg))
    row.update(_metrics_dict(m))
    return row


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    report_text: Optional[str] = None,
) -> str:
    """Serialise results to a JSON string.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        report_text: Optional recommendation report text to embed.

    Returns:
        Compact JSON string.
    """
    records = [_row(cfg, m) for cfg, m in results]
    payload: Dict[str, Any] = {
        "num_results": len(records),
        "results": records,
    }
    if report_text is not None:
        payload["recommendation_report"] = report_text
    return json.dumps(payload, indent=2, default=str)


def export_json_file(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    path: str,
    report_text: Optional[str] = None,
) -> None:
    """Write JSON export to *path*.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        path: Output file path.
        report_text: Optional recommendation report text.
    """
    with open(path, "w") as fh:
        fh.write(export_json(results, report_text))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
) -> str:
    """Serialise results to CSV text.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.

    Returns:
        CSV text with header row.
    """
    if not results:
        return ""

    fieldnames = _CONFIG_FIELDS + _METRICS_FIELDS
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for cfg, m in results:
        writer.writerow(_row(cfg, m))
    return buf.getvalue()


def export_csv_file(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    path: str,
) -> None:
    """Write CSV export to *path*.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        path: Output file path.
    """
    with open(path, "w", newline="") as fh:
        fh.write(export_csv(results))


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a Markdown table."""
    sep = " | ".join("---" for _ in headers)
    header_row = " | ".join(headers)
    lines = [f"| {header_row} |", f"| {sep} |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def export_markdown(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    report_text: Optional[str] = None,
) -> str:
    """Render results as a Markdown document.

    Includes:
    - Summary table (key metrics per result)
    - Recommendation report (if provided)

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        report_text: Optional text block to append.

    Returns:
        Markdown string.
    """
    lines = [
        "# Benchmark Results",
        "",
        f"**Total configs evaluated:** {len(results)}",
        "",
        "## Results Table",
        "",
    ]

    summary_headers = [
        "suite", "TP", "PP", "DP", "dataset",
        "throughput_tps", "goodput_rps",
        "ttft_p90_ms", "tpot_p90_ms",
        "joint_slo_pct", "gpu_mem_gb",
    ]
    rows = []
    for cfg, m in results:
        rows.append([
            cfg.benchmark_suite,
            str(cfg.tp), str(cfg.pp), str(cfg.dp),
            cfg.dataset,
            f"{m.throughput_tps:.1f}",
            f"{m.goodput_rps:.2f}",
            f"{m.ttft_p90_ms:.1f}",
            f"{m.tpot_p90_ms:.1f}",
            f"{m.joint_slo_attainment_pct:.1f}",
            f"{m.gpu_mem_used_gb:.1f}",
        ])

    lines.append(_md_table(summary_headers, rows))
    lines.append("")

    if report_text:
        lines += [
            "## Recommendation Report",
            "",
            "```",
            report_text,
            "```",
            "",
        ]

    return "\n".join(lines)


def export_markdown_file(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    path: str,
    report_text: Optional[str] = None,
) -> None:
    """Write Markdown export to *path*.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        path: Output file path.
        report_text: Optional recommendation report text.
    """
    with open(path, "w") as fh:
        fh.write(export_markdown(results, report_text))


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Parallelisms Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 24px; background: #f5f5f5; color: #222; }}
    h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); border-radius: 6px;
             overflow: hidden; margin-bottom: 32px; font-size: 13px; }}
    th {{ background: #0f3460; color: #fff; padding: 8px 10px; text-align: left;
          cursor: pointer; user-select: none; }}
    th:hover {{ background: #1a4a7a; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #e8f4ff; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 11px; font-weight: 600; }}
    .badge-ok {{ background: #d4edda; color: #155724; }}
    .badge-warn {{ background: #fff3cd; color: #856404; }}
    .badge-bad {{ background: #f8d7da; color: #721c24; }}
    .chart-wrap {{ background: #fff; padding: 16px; border-radius: 6px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 32px; }}
    pre.report {{ background: #1a1a2e; color: #e0e0e0; padding: 20px;
                  border-radius: 6px; overflow-x: auto; font-size: 12px; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 16px; margin-bottom: 32px; }}
    .stat-card {{ background: #fff; border-radius: 6px; padding: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: center; }}
    .stat-val {{ font-size: 28px; font-weight: 700; color: #0f3460; }}
    .stat-lbl {{ font-size: 12px; color: #666; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>&#x1F9EE; Parallelisms Benchmark Dashboard</h1>
  <p>{num_results} configurations evaluated across {num_suites} suite(s):
  <strong>{suite_names}</strong></p>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-val">{num_results}</div>
      <div class="stat-lbl">Configs evaluated</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{best_tps:.0f}</div>
      <div class="stat-lbl">Best throughput (tok/s)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{best_goodput:.2f}</div>
      <div class="stat-lbl">Best goodput (req/s)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{best_slo:.1f}%</div>
      <div class="stat-lbl">Best joint SLO attainment</div>
    </div>
  </div>

  <h2>Throughput vs TTFT P90</h2>
  <div class="chart-wrap" style="height:320px">
    <canvas id="tpsvsTtft"></canvas>
  </div>

  <h2>SLO Attainment Overview</h2>
  <div class="chart-wrap" style="height:300px">
    <canvas id="sloChart"></canvas>
  </div>

  <h2>All Results</h2>
  {results_table}

  {report_section}

  <script>
  // Sort table on header click
  document.querySelectorAll('th[data-col]').forEach(th => {{
    th.addEventListener('click', () => {{
      const col = +th.dataset.col;
      const tbody = th.closest('table').querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const asc = th.dataset.asc !== '1';
      rows.sort((a, b) => {{
        const av = a.cells[col].innerText.replace(/[^0-9.\-]/g,'') || '0';
        const bv = b.cells[col].innerText.replace(/[^0-9.\-]/g,'') || '0';
        return asc ? +av - +bv : +bv - +av;
      }});
      rows.forEach(r => tbody.appendChild(r));
      th.dataset.asc = asc ? '1' : '0';
    }});
  }});

  // Scatter: throughput vs TTFT p90
  new Chart(document.getElementById('tpsvsTtft'), {{
    type: 'scatter',
    data: {{
      datasets: [{scatter_datasets}]
    }},
    options: {{
      plugins: {{ legend: {{ display: true }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Throughput (tok/s)' }} }},
        y: {{ title: {{ display: true, text: 'TTFT p90 (ms)' }} }},
      }},
      responsive: true, maintainAspectRatio: false,
    }}
  }});

  // Bar: SLO attainment
  new Chart(document.getElementById('sloChart'), {{
    type: 'bar',
    data: {{
      labels: {slo_labels},
      datasets: [{{
        label: 'Joint SLO attainment (%)',
        data: {slo_values},
        backgroundColor: {slo_colors},
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ min: 0, max: 100, title: {{ display: true, text: 'Attainment (%)' }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }},
      }},
      responsive: true, maintainAspectRatio: false,
    }}
  }});
  </script>
</body>
</html>
"""

_SUITE_COLORS = {
    "vllm_parallelism": "rgba(15,52,96,0.7)",
    "distserve": "rgba(232,77,69,0.7)",
    "sarathi": "rgba(39,174,96,0.7)",
    "seesaw": "rgba(156,89,182,0.7)",
}


def export_html(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    report_text: Optional[str] = None,
) -> str:
    """Render a self-contained HTML dashboard.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        report_text: Optional recommendation report text.

    Returns:
        Full HTML string.
    """
    if not results:
        return "<html><body><p>No results.</p></body></html>"

    suites = sorted({cfg.benchmark_suite for cfg, _ in results})
    best_tps = max((m.throughput_tps for _, m in results), default=0.0)
    best_goodput = max((m.goodput_rps for _, m in results), default=0.0)
    best_slo = max((m.joint_slo_attainment_pct for _, m in results), default=0.0)

    # Results table
    headers = [
        "Suite", "TP", "PP", "DP", "Dataset",
        "TPS", "Goodput", "TTFT p90 (ms)", "TPOT p90 (ms)",
        "Joint SLO %", "GPU mem (GB)",
    ]
    header_html = "".join(
        f'<th data-col="{i}">{h}</th>'
        for i, h in enumerate(headers)
    )

    rows_html = ""
    for cfg, m in results:
        slo = m.joint_slo_attainment_pct
        slo_cls = "badge-ok" if slo >= 90 else ("badge-warn" if slo >= 70 else "badge-bad")
        rows_html += (
            "<tr>"
            f"<td>{cfg.benchmark_suite}</td>"
            f"<td>{cfg.tp}</td><td>{cfg.pp}</td><td>{cfg.dp}</td>"
            f"<td>{cfg.dataset}</td>"
            f"<td>{m.throughput_tps:.1f}</td>"
            f"<td>{m.goodput_rps:.2f}</td>"
            f"<td>{m.ttft_p90_ms:.1f}</td>"
            f"<td>{m.tpot_p90_ms:.1f}</td>"
            f"<td><span class='badge {slo_cls}'>{slo:.1f}%</span></td>"
            f"<td>{m.gpu_mem_used_gb:.1f}</td>"
            "</tr>\n"
        )

    results_table = (
        f"<table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )

    # Scatter chart datasets per suite
    scatter_ds_parts = []
    for suite in suites:
        pts = [
            f'{{"x":{m.throughput_tps:.2f},"y":{m.ttft_p90_ms:.2f}}}'
            for cfg, m in results
            if cfg.benchmark_suite == suite
        ]
        color = _SUITE_COLORS.get(suite, "rgba(100,100,100,0.7)")
        scatter_ds_parts.append(
            f'{{"label":"{suite}","data":[{",".join(pts)}],'
            f'"backgroundColor":"{color}","pointRadius":5}}'
        )
    scatter_datasets = ",".join(scatter_ds_parts)

    # SLO bar chart
    slo_labels = json.dumps([
        f"{cfg.benchmark_suite[:4]}-TP{cfg.tp}PP{cfg.pp}-{cfg.dataset[:6]}"
        for cfg, _ in results[:30]  # cap for readability
    ])
    slo_values = json.dumps([
        round(m.joint_slo_attainment_pct, 1) for _, m in results[:30]
    ])
    slo_colors_list = [
        ('"rgba(39,174,96,0.7)"' if m.joint_slo_attainment_pct >= 90
         else ('"rgba(255,165,0,0.7)"' if m.joint_slo_attainment_pct >= 70
               else '"rgba(220,53,69,0.7)"'))
        for _, m in results[:30]
    ]
    slo_colors = f"[{','.join(slo_colors_list)}]"

    # Report section
    if report_text:
        report_section = (
            "<h2>Recommendation Report</h2>"
            f"<pre class='report'>{report_text}</pre>"
        )
    else:
        report_section = ""

    return _HTML_TEMPLATE.format(
        num_results=len(results),
        num_suites=len(suites),
        suite_names=", ".join(suites),
        best_tps=best_tps,
        best_goodput=best_goodput,
        best_slo=best_slo,
        results_table=results_table,
        scatter_datasets=scatter_datasets,
        slo_labels=slo_labels,
        slo_values=slo_values,
        slo_colors=slo_colors,
        report_section=report_section,
    )


def export_html_file(
    results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    path: str,
    report_text: Optional[str] = None,
) -> None:
    """Write HTML dashboard to *path*.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        path: Output file path (conventionally ending in ``.html``).
        report_text: Optional recommendation report text.
    """
    with open(path, "w") as fh:
        fh.write(export_html(results, report_text))


# ---------------------------------------------------------------------------
# BenchmarkExporter — convenience wrapper
# ---------------------------------------------------------------------------

class BenchmarkExporter:
    """Convenience wrapper that exports results to all supported formats.

    Args:
        results: List of (ConfigPoint, BenchmarkMetrics) tuples.
        report_text: Optional recommendation report text.
    """

    def __init__(
        self,
        results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
        report_text: Optional[str] = None,
    ) -> None:
        self.results = results
        self.report_text = report_text

    def to_json(self) -> str:
        return export_json(self.results, self.report_text)

    def to_csv(self) -> str:
        return export_csv(self.results)

    def to_markdown(self) -> str:
        return export_markdown(self.results, self.report_text)

    def to_html(self) -> str:
        return export_html(self.results, self.report_text)

    def write_all(self, output_dir: str, prefix: str = "benchmark") -> Dict[str, str]:
        """Write all formats to *output_dir*.

        Args:
            output_dir: Directory to write files into.
            prefix: Filename prefix (default: "benchmark").

        Returns:
            Dict mapping format name to written file path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        for fmt, fn, ext in [
            ("json", export_json_file, ".json"),
            ("csv", export_csv_file, ".csv"),
            ("markdown", export_markdown_file, ".md"),
            ("html", export_html_file, ".html"),
        ]:
            path = os.path.join(output_dir, f"{prefix}{ext}")
            if fmt in ("json", "markdown", "html"):
                fn(self.results, path, self.report_text)  # type: ignore[call-arg]
            else:
                fn(self.results, path)  # type: ignore[call-arg]
            paths[fmt] = path

        return paths
