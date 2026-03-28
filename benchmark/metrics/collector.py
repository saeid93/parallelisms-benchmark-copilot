"""
Stage 4 — Metrics collector.

Scrapes the vLLM /metrics endpoint (Prometheus format) and vLLM stats API
per run, recording per-request TTFT, TPOT, and e2e latency for SLO
attainment computation.  Also records per-phase and per-op timing for
SARATHI-style instrumentation, and KV transmission CDFs for DistServe.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import requests as http_requests
except ImportError:  # pragma: no cover
    http_requests = None  # type: ignore[assignment]

from benchmark.config.schema import BenchmarkMetrics, PerOpBreakdownMs

logger = logging.getLogger(__name__)

# Prometheus metric name patterns for vLLM ≥ 0.4
_METRIC_PATTERNS: Dict[str, str] = {
    "throughput_tps": r'vllm:tokens_total\{[^}]*\}\s+([\d.]+)',
    "gpu_mem_used_gb": r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.]+)',
    "kv_cache_hit_rate": r'vllm:cpu_prefix_cache_hit_rate\{[^}]*\}\s+([\d.]+)',
    "preemption_rate": r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.]+)',
}


# ---------------------------------------------------------------------------
# Per-request timing record
# ---------------------------------------------------------------------------

@dataclass
class RequestTiming:
    """Per-request timing breakdown."""

    request_id: str
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    prefill_exec_ms: float = 0.0
    prefill_queue_ms: float = 0.0
    decode_exec_ms: float = 0.0
    decode_queue_ms: float = 0.0
    kv_transmission_ms: float = 0.0


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    """Return the p-th percentile of a sorted list.

    Args:
        data: List of float values (need not be sorted).
        pct: Percentile in [0, 100].

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_data = sorted(data)
    idx = (pct / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def _latency_percentiles(values: List[float]) -> Tuple[float, float, float]:
    """Return (p50, p90, p99) for a list of latency values."""
    if not values:
        return 0.0, 0.0, 0.0
    return (
        _percentile(values, 50),
        _percentile(values, 90),
        _percentile(values, 99),
    )


# ---------------------------------------------------------------------------
# Prometheus scraper
# ---------------------------------------------------------------------------

def scrape_prometheus(url: str, timeout_s: float = 5.0) -> str:
    """Fetch raw Prometheus exposition text from a vLLM /metrics endpoint.

    Args:
        url: Full URL of the /metrics endpoint.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Raw Prometheus text, or empty string on error.
    """
    if http_requests is None:
        logger.warning("requests library not available; skipping Prometheus scrape")
        return ""
    try:
        resp = http_requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return ""


def parse_prometheus_float(text: str, pattern: str) -> Optional[float]:
    """Extract a single float value matching *pattern* from Prometheus text.

    Args:
        text: Raw Prometheus exposition text.
        pattern: Regular expression with one capture group for the value.

    Returns:
        Parsed float, or None if pattern did not match.
    """
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects and aggregates metrics for a single benchmark run.

    Args:
        prometheus_url: Base URL of the vLLM instance (e.g., http://localhost:8000).
        benchmark_suite: Suite name used to enable suite-specific collection.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:8000",
        benchmark_suite: str = "vllm_parallelism",
    ) -> None:
        self.prometheus_url = prometheus_url
        self.benchmark_suite = benchmark_suite
        self._request_timings: List[RequestTiming] = []

    # ------------------------------------------------------------------
    # Per-request recording
    # ------------------------------------------------------------------

    def record_request(self, timing: RequestTiming) -> None:
        """Record timing for a completed request."""
        self._request_timings.append(timing)

    def record_requests(self, timings: List[RequestTiming]) -> None:
        """Record timing for multiple completed requests."""
        self._request_timings.extend(timings)

    # ------------------------------------------------------------------
    # Prometheus scraping
    # ------------------------------------------------------------------

    def _scrape(self) -> str:
        url = f"{self.prometheus_url}/metrics"
        return scrape_prometheus(url)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def collect(
        self,
        total_time_s: float,
        num_output_tokens: int,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
    ) -> BenchmarkMetrics:
        """Aggregate all collected timings into a BenchmarkMetrics record.

        Args:
            total_time_s: Wall-clock duration of the benchmark run in seconds.
            num_output_tokens: Total output tokens generated (for TPS calc).
            ttft_slo_ms: TTFT SLO target in milliseconds.
            tpot_slo_ms: TPOT SLO target in milliseconds.

        Returns:
            Populated BenchmarkMetrics instance.
        """
        metrics = BenchmarkMetrics()
        n = len(self._request_timings)

        # --- Throughput ---
        if total_time_s > 0:
            metrics.throughput_tps = num_output_tokens / total_time_s
            metrics.end_to_end_throughput_rps = n / total_time_s

        # --- Latency percentiles ---
        ttft_vals = [t.ttft_ms for t in self._request_timings]
        tpot_vals = [t.tpot_ms for t in self._request_timings]
        e2e_vals = [t.e2e_latency_ms for t in self._request_timings]

        metrics.ttft_p50_ms, metrics.ttft_p90_ms, metrics.ttft_p99_ms = (
            _latency_percentiles(ttft_vals)
        )
        metrics.tpot_p50_ms, metrics.tpot_p90_ms, metrics.tpot_p99_ms = (
            _latency_percentiles(tpot_vals)
        )
        metrics.e2e_latency_p50_ms, metrics.e2e_latency_p90_ms, metrics.e2e_latency_p99_ms = (
            _latency_percentiles(e2e_vals)
        )

        # --- SLO attainment ---
        if n > 0:
            metrics.ttft_slo_attainment_pct = (
                sum(1 for v in ttft_vals if v <= ttft_slo_ms) / n * 100.0
            )
            metrics.tpot_slo_attainment_pct = (
                sum(1 for v in tpot_vals if v <= tpot_slo_ms) / n * 100.0
            )
            metrics.joint_slo_attainment_pct = (
                sum(
                    1
                    for t in self._request_timings
                    if t.ttft_ms <= ttft_slo_ms and t.tpot_ms <= tpot_slo_ms
                )
                / n
                * 100.0
            )

        # --- Phase timing (DistServe + SARATHI) ---
        if self._request_timings:
            metrics.prefill_exec_time_ms = statistics.mean(
                t.prefill_exec_ms for t in self._request_timings
            )
            metrics.prefill_queuing_time_ms = statistics.mean(
                t.prefill_queue_ms for t in self._request_timings
            )
            metrics.decode_exec_time_ms = statistics.mean(
                t.decode_exec_ms for t in self._request_timings
            )
            metrics.decode_queuing_time_ms = statistics.mean(
                t.decode_queue_ms for t in self._request_timings
            )
            metrics.kv_transmission_time_ms = statistics.mean(
                t.kv_transmission_ms for t in self._request_timings
            )

        # --- Phase time percentages ---
        total_phase_ms = (
            metrics.prefill_exec_time_ms
            + metrics.decode_exec_time_ms
            + metrics.kv_transmission_time_ms
        )
        if total_phase_ms > 0:
            metrics.prefill_phase_time_pct = (
                metrics.prefill_exec_time_ms / total_phase_ms * 100.0
            )
            metrics.decode_phase_time_pct = (
                metrics.decode_exec_time_ms / total_phase_ms * 100.0
            )
            metrics.transmission_time_pct = (
                metrics.kv_transmission_time_ms / total_phase_ms * 100.0
            )

        # --- Prometheus metrics (best-effort) ---
        prom_text = self._scrape()
        if prom_text:
            for attr, pattern in _METRIC_PATTERNS.items():
                val = parse_prometheus_float(prom_text, pattern)
                if val is not None:
                    setattr(metrics, attr, val)

        return metrics

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._request_timings = []
