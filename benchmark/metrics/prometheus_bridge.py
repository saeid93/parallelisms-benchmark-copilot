"""
Prometheus metrics bridge for Kubernetes-mode benchmark runs.

Provides two capabilities for Kubernetes execution mode:

1. **ServiceMonitor-aware scraping** — Discovers vLLM pods by label selector
   within the cluster and scrapes their ``/metrics`` endpoints via the
   Kubernetes Service DNS.

2. **Pushgateway support** — Pushes aggregated per-run metrics to a
   Prometheus Pushgateway so that they survive pod termination and can be
   scraped by a long-lived Prometheus instance.

When running locally (``execution_mode="local"``) the bridge falls back to
direct ``http://localhost:8000/metrics`` scraping, matching the behaviour
of ``MetricsCollector``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import requests as http_requests
except ImportError:  # pragma: no cover
    http_requests = None  # type: ignore[assignment]

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    _HAS_PROM_CLIENT = True
except ImportError:  # pragma: no cover
    _HAS_PROM_CLIENT = False

from benchmark.config.schema import BenchmarkMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Additional vLLM Prometheus metric patterns (beyond collector.py)
# ---------------------------------------------------------------------------

VLLM_METRIC_PATTERNS: Dict[str, str] = {
    # Core throughput / request metrics
    "tokens_total": r"vllm:tokens_total\{[^}]*\}\s+([\d.]+)",
    "generation_tokens_total": r"vllm:generation_tokens_total\{[^}]*\}\s+([\d.]+)",
    "prompt_tokens_total": r"vllm:prompt_tokens_total\{[^}]*\}\s+([\d.]+)",
    "request_success_total": r"vllm:request_success_total\{[^}]*\}\s+([\d.]+)",
    # Latency histograms (sum / count for averages)
    "e2e_request_latency_seconds_sum": (
        r"vllm:e2e_request_latency_seconds_sum\{[^}]*\}\s+([\d.]+)"
    ),
    "e2e_request_latency_seconds_count": (
        r"vllm:e2e_request_latency_seconds_count\{[^}]*\}\s+([\d.]+)"
    ),
    "time_to_first_token_seconds_sum": (
        r"vllm:time_to_first_token_seconds_sum\{[^}]*\}\s+([\d.]+)"
    ),
    "time_to_first_token_seconds_count": (
        r"vllm:time_to_first_token_seconds_count\{[^}]*\}\s+([\d.]+)"
    ),
    "time_per_output_token_seconds_sum": (
        r"vllm:time_per_output_token_seconds_sum\{[^}]*\}\s+([\d.]+)"
    ),
    "time_per_output_token_seconds_count": (
        r"vllm:time_per_output_token_seconds_count\{[^}]*\}\s+([\d.]+)"
    ),
    # Cache / scheduler
    "gpu_cache_usage_perc": r"vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.]+)",
    "cpu_prefix_cache_hit_rate": (
        r"vllm:cpu_prefix_cache_hit_rate\{[^}]*\}\s+([\d.]+)"
    ),
    "num_preemptions_total": (
        r"vllm:num_preemptions_total\{[^}]*\}\s+([\d.]+)"
    ),
    # Scheduler state
    "num_requests_running": (
        r"vllm:num_requests_running\{[^}]*\}\s+([\d.]+)"
    ),
    "num_requests_waiting": (
        r"vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)"
    ),
    "num_requests_swapped": (
        r"vllm:num_requests_swapped\{[^}]*\}\s+([\d.]+)"
    ),
}


def _parse_float(text: str, pattern: str) -> Optional[float]:
    """Extract a float from *text* using a regex pattern."""
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            return None
    return None


@dataclass
class PrometheusSnapshot:
    """Parsed snapshot of vLLM Prometheus metrics."""

    raw_text: str = ""
    values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str) -> "PrometheusSnapshot":
        """Parse raw Prometheus exposition text into a snapshot."""
        values: Dict[str, float] = {}
        for name, pattern in VLLM_METRIC_PATTERNS.items():
            val = _parse_float(text, pattern)
            if val is not None:
                values[name] = val
        return cls(raw_text=text, values=values)


# ---------------------------------------------------------------------------
# PrometheusBridge
# ---------------------------------------------------------------------------

class PrometheusBridge:
    """Bridges vLLM Prometheus metrics with the benchmark pipeline.

    In Kubernetes mode, discovers the vLLM service endpoint by constructing
    the in-cluster DNS name from namespace and service name.  In local mode,
    falls back to a direct URL.

    Args:
        execution_mode: ``"local"`` or ``"kubernetes"``.
        prometheus_url: Direct URL for local-mode scraping.
        namespace: Kubernetes namespace where vLLM pods run.
        service_name: Kubernetes Service name fronting vLLM pods.
        service_port: Port on the K8s Service (typically 8000).
        pushgateway_url: Optional Prometheus Pushgateway URL for push mode.
        scrape_timeout_s: HTTP timeout for metric scraping.
    """

    def __init__(
        self,
        execution_mode: str = "local",
        prometheus_url: str = "http://localhost:8000",
        namespace: str = "benchmark",
        service_name: str = "vllm-benchmark",
        service_port: int = 8000,
        pushgateway_url: Optional[str] = None,
        scrape_timeout_s: float = 5.0,
    ) -> None:
        self.execution_mode = execution_mode
        self.prometheus_url = prometheus_url
        self.namespace = namespace
        self.service_name = service_name
        self.service_port = service_port
        self.pushgateway_url = pushgateway_url
        self.scrape_timeout_s = scrape_timeout_s

    # ------------------------------------------------------------------
    # URL resolution
    # ------------------------------------------------------------------

    def _metrics_url(self, run_id: Optional[str] = None) -> str:
        """Build the metrics endpoint URL based on execution mode.

        In Kubernetes mode, uses in-cluster DNS:
        ``http://<service_name>.<namespace>.svc.cluster.local:<port>/metrics``

        Args:
            run_id: Optional run ID (unused currently; reserved for per-pod
                scraping via label selector in future).

        Returns:
            Full URL string.
        """
        if self.execution_mode == "kubernetes":
            host = (
                f"{self.service_name}.{self.namespace}"
                f".svc.cluster.local"
            )
            return f"http://{host}:{self.service_port}/metrics"
        return f"{self.prometheus_url}/metrics"

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape(self, run_id: Optional[str] = None) -> PrometheusSnapshot:
        """Scrape the vLLM /metrics endpoint and return a parsed snapshot.

        Args:
            run_id: Optional run ID for per-pod resolution.

        Returns:
            PrometheusSnapshot with all parsed vLLM metrics.
        """
        url = self._metrics_url(run_id)
        if http_requests is None:
            logger.warning("requests library not available; skipping scrape")
            return PrometheusSnapshot()
        try:
            resp = http_requests.get(url, timeout=self.scrape_timeout_s)
            resp.raise_for_status()
            return PrometheusSnapshot.from_text(resp.text)
        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", url, exc)
            return PrometheusSnapshot()

    # ------------------------------------------------------------------
    # Push to Pushgateway
    # ------------------------------------------------------------------

    def push_metrics(
        self,
        run_id: str,
        metrics: BenchmarkMetrics,
        job_name: str = "benchmark",
    ) -> bool:
        """Push aggregated benchmark metrics to a Prometheus Pushgateway.

        Args:
            run_id: Unique run identifier (used as grouping key).
            metrics: Aggregated BenchmarkMetrics to push.
            job_name: Prometheus job name for the push.

        Returns:
            True on success, False on failure.
        """
        if not self.pushgateway_url:
            logger.debug("No pushgateway_url configured; skipping push")
            return False

        if not _HAS_PROM_CLIENT:
            logger.warning(
                "prometheus_client not available; cannot push metrics"
            )
            return False

        registry = CollectorRegistry()

        # Create gauges for key metrics
        _metric_fields = {
            "throughput_tps": "Throughput in tokens per second",
            "end_to_end_throughput_rps": "End-to-end requests per second",
            "goodput_rps": "Goodput in requests per second",
            "ttft_p50_ms": "TTFT P50 in milliseconds",
            "ttft_p90_ms": "TTFT P90 in milliseconds",
            "ttft_p99_ms": "TTFT P99 in milliseconds",
            "tpot_p50_ms": "TPOT P50 in milliseconds",
            "tpot_p90_ms": "TPOT P90 in milliseconds",
            "tpot_p99_ms": "TPOT P99 in milliseconds",
            "e2e_latency_p50_ms": "E2E latency P50 in milliseconds",
            "e2e_latency_p90_ms": "E2E latency P90 in milliseconds",
            "e2e_latency_p99_ms": "E2E latency P99 in milliseconds",
            "joint_slo_attainment_pct": "Joint SLO attainment percentage",
            "gpu_mem_used_gb": "GPU memory used in GB",
            "kv_cache_hit_rate": "KV cache hit rate",
            "preemption_rate": "Preemption rate",
        }

        for field_name, description in _metric_fields.items():
            gauge = Gauge(
                f"benchmark_{field_name}",
                description,
                labelnames=["run_id"],
                registry=registry,
            )
            gauge.labels(run_id=run_id).set(getattr(metrics, field_name, 0.0))

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=job_name,
                registry=registry,
                grouping_key={"run_id": run_id},
            )
            logger.info(
                "Pushed metrics for run_id=%s to %s",
                run_id[:8],
                self.pushgateway_url,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to push metrics to %s: %s",
                self.pushgateway_url,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Enrich BenchmarkMetrics with Prometheus data
    # ------------------------------------------------------------------

    def enrich_metrics(
        self,
        metrics: BenchmarkMetrics,
        snapshot: PrometheusSnapshot,
    ) -> BenchmarkMetrics:
        """Enrich a BenchmarkMetrics instance with scraped Prometheus values.

        Updates gpu_mem_used_gb, kv_cache_hit_rate, and preemption_rate from
        the Prometheus snapshot if available.

        Args:
            metrics: Metrics to enrich (copied, not mutated in place).
            snapshot: PrometheusSnapshot from a scrape.

        Returns:
            A new BenchmarkMetrics with Prometheus values applied.
        """
        enriched = metrics.model_copy()

        if "gpu_cache_usage_perc" in snapshot.values:
            # Matches existing collector.py mapping: vllm:gpu_cache_usage_perc
            # is stored as gpu_mem_used_gb for backward compatibility.
            enriched.gpu_mem_used_gb = snapshot.values["gpu_cache_usage_perc"]
        if "cpu_prefix_cache_hit_rate" in snapshot.values:
            enriched.kv_cache_hit_rate = snapshot.values[
                "cpu_prefix_cache_hit_rate"
            ]
        if "num_preemptions_total" in snapshot.values:
            enriched.preemption_rate = snapshot.values[
                "num_preemptions_total"
            ]

        return enriched
