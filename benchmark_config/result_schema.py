"""Structured output schema for benchmark results (Section 2).

Each benchmark run emits one ``BenchmarkResult`` record that captures
the full configuration snapshot alongside all measured metrics.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from benchmark_config.schema import BenchmarkRunConfig


@dataclass
class LatencyMetrics:
    """Latency percentiles in milliseconds."""

    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    p50_e2e_latency_ms: float = 0.0
    p90_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0


@dataclass
class ThroughputMetrics:
    """Throughput and goodput measurements."""

    total_tokens_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0
    requests_per_sec: float = 0.0
    goodput_requests_per_sec: float = 0.0
    slo_attainment_pct: float = 0.0


@dataclass
class ResourceMetrics:
    """GPU / memory utilisation during the run."""

    peak_gpu_memory_gb: float = 0.0
    avg_gpu_utilization_pct: float = 0.0
    kv_cache_utilization_pct: float = 0.0
    num_gpus_used: int = 0


@dataclass
class BenchmarkResult:
    """Complete result record for one benchmark run.

    This is the canonical output artefact; every run persists exactly one
    of these as a JSON record.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    status: str = "pending"

    config: Optional[BenchmarkRunConfig] = None
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)

    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    VALID_STATUSES: tuple[str, ...] = (
        "pending", "running", "completed", "failed", "skipped",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full result to a JSON-compatible dict."""
        d = asdict(self)
        # Remove class-level constants from serialised output
        for section in (d, d.get("config", {})):
            if not isinstance(section, dict):
                continue
            keys_to_drop = [
                k for k in section
                if k.startswith("VALID_") or k.isupper()
            ]
            for k in keys_to_drop:
                section.pop(k, None)
            # Also clean nested sub-configs
            for v in list(section.values()):
                if isinstance(v, dict):
                    nested_drop = [
                        nk for nk in v
                        if nk.startswith("VALID_") or nk.isupper()
                    ]
                    for nk in nested_drop:
                        v.pop(nk, None)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
