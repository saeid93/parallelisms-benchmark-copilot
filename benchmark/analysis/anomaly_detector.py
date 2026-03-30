"""Performance anomaly detector for LLM serving benchmarks.

Inspired by:
* AIOps / intelligent monitoring literature
* "Unsupervised Anomaly Detection for Self-Driving Cars" (adapted to ML systems)
* Statistical Process Control (SPC) methods
* "Robust Anomaly Detection for Multivariate Time Series" (Su et al. 2019)

Detects performance anomalies in metrics time-series: latency spikes,
throughput drops, OOM patterns, using Z-score, IQR, and temporal
pattern detection with automated root cause correlation.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AnomalyType(str, Enum):
    """Types of performance anomalies."""

    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    OOM_EVENT = "oom_event"
    MEMORY_LEAK = "memory_leak"
    PREEMPTION_BURST = "preemption_burst"
    QUEUE_BUILDUP = "queue_buildup"
    GPU_THROTTLING = "gpu_throttling"
    COMMUNICATION_STALL = "communication_stall"
    KV_CACHE_PRESSURE = "kv_cache_pressure"
    TAIL_LATENCY_REGRESSION = "tail_latency_regression"


class AnomalySeverity(str, Enum):
    """Severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """How the anomaly was detected."""

    Z_SCORE = "z_score"
    IQR = "iqr"
    PERCENTAGE_CHANGE = "percentage_change"
    THRESHOLD = "threshold"
    TREND = "trend"
    TEMPORAL_PATTERN = "temporal_pattern"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class MetricSample:
    """A single metric observation."""

    metric_name: str
    timestamp_s: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected anomaly."""

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp_s: float
    metric_name: str
    observed_value: float
    expected_value: float
    deviation: float  # how far from normal (in std devs or %)
    detection_method: DetectionMethod
    description: str
    possible_causes: List[str] = field(default_factory=list)
    correlated_metrics: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "timestamp_s": self.timestamp_s,
            "metric": self.metric_name,
            "observed": self.observed_value,
            "expected": self.expected_value,
            "deviation": self.deviation,
            "method": self.detection_method.value,
            "description": self.description,
            "causes": self.possible_causes,
            "correlated": self.correlated_metrics,
            "recommendation": self.recommendation,
        }


@dataclass
class AnomalyReport:
    """Aggregated anomaly detection results."""

    total_samples: int = 0
    total_anomalies: int = 0
    anomalies: List[Anomaly] = field(default_factory=list)
    anomaly_rate: float = 0.0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    metrics_health: Dict[str, str] = field(default_factory=dict)  # metric -> "healthy"/"degraded"/"critical"

    def summary(self) -> str:
        lines = [
            "Anomaly Detection Report",
            f"  Samples analysed  : {self.total_samples}",
            f"  Anomalies found   : {self.total_anomalies}",
            f"  Anomaly rate      : {self.anomaly_rate:.2%}",
        ]
        if self.by_severity:
            lines.append("  By severity:")
            for sev, count in sorted(self.by_severity.items()):
                lines.append(f"    {sev}: {count}")
        if self.by_type:
            lines.append("  By type:")
            for typ, count in sorted(self.by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    {typ}: {count}")
        if self.metrics_health:
            lines.append("  Metric health:")
            for metric, health in sorted(self.metrics_health.items()):
                lines.append(f"    {metric}: {health}")
        if self.anomalies:
            lines.append("  Top anomalies:")
            severity_order = {
                AnomalySeverity.CRITICAL: 0,
                AnomalySeverity.HIGH: 1,
                AnomalySeverity.MEDIUM: 2,
                AnomalySeverity.LOW: 3,
            }
            top = sorted(self.anomalies, key=lambda a: severity_order.get(a.severity, 4))[:5]
            for a in top:
                lines.append(f"    [{a.severity.value}] {a.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

# Map anomaly types to possible root causes
_CAUSE_MAP: Dict[AnomalyType, List[str]] = {
    AnomalyType.LATENCY_SPIKE: [
        "KV cache eviction causing recomputation",
        "Request preemption and re-scheduling",
        "Long prompt causing prefill bottleneck",
        "GPU thermal throttling",
    ],
    AnomalyType.THROUGHPUT_DROP: [
        "Decreased batch size due to memory pressure",
        "Network communication stall",
        "Pipeline bubble overhead increase",
        "Resource contention from co-located workloads",
    ],
    AnomalyType.OOM_EVENT: [
        "KV cache exhausted",
        "Large batch of long sequences",
        "Memory fragmentation",
        "Unexpected model weight loading",
    ],
    AnomalyType.MEMORY_LEAK: [
        "Unreleased GPU tensors",
        "Growing KV cache without eviction",
        "Python garbage collection delay",
    ],
    AnomalyType.GPU_THROTTLING: [
        "Thermal throttling (high temperature)",
        "Power limit exceeded",
        "Insufficient cooling",
    ],
}


class AnomalyDetector:
    """Detects performance anomalies in metric time-series.

    Usage::

        ad = AnomalyDetector(z_threshold=3.0)
        ad.add_samples([MetricSample("latency_ms", t, val) for t, val in data])
        report = ad.detect()
        print(report.summary())
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        pct_change_threshold: float = 50.0,
        min_samples: int = 10,
    ) -> None:
        self._z_threshold = z_threshold
        self._iqr_multiplier = iqr_multiplier
        self._pct_threshold = pct_change_threshold
        self._min_samples = min_samples
        self._samples: Dict[str, List[MetricSample]] = {}

    def add_sample(self, sample: MetricSample) -> None:
        self._samples.setdefault(sample.metric_name, []).append(sample)

    def add_samples(self, samples: List[MetricSample]) -> None:
        for s in samples:
            self.add_sample(s)

    def _z_score_detect(
        self, metric_name: str, samples: List[MetricSample]
    ) -> List[Anomaly]:
        """Detect anomalies using Z-score method."""
        anomalies: List[Anomaly] = []
        values = [s.value for s in samples]
        if len(values) < self._min_samples:
            return anomalies

        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        if stdev == 0:
            return anomalies

        for s in samples:
            z = abs(s.value - mean) / stdev
            if z > self._z_threshold:
                is_spike = s.value > mean
                atype = (
                    AnomalyType.LATENCY_SPIKE
                    if "latency" in metric_name.lower() and is_spike
                    else AnomalyType.THROUGHPUT_DROP
                    if "throughput" in metric_name.lower() and not is_spike
                    else AnomalyType.LATENCY_SPIKE
                )
                severity = self._classify_severity(z)
                anomalies.append(
                    Anomaly(
                        anomaly_type=atype,
                        severity=severity,
                        timestamp_s=s.timestamp_s,
                        metric_name=metric_name,
                        observed_value=s.value,
                        expected_value=mean,
                        deviation=z,
                        detection_method=DetectionMethod.Z_SCORE,
                        description=(
                            f"{metric_name} = {s.value:.2f} (Z={z:.1f}σ from mean {mean:.2f})"
                        ),
                        possible_causes=_CAUSE_MAP.get(atype, []),
                    )
                )
        return anomalies

    def _iqr_detect(
        self, metric_name: str, samples: List[MetricSample]
    ) -> List[Anomaly]:
        """Detect anomalies using IQR method (robust to non-normal distributions)."""
        anomalies: List[Anomaly] = []
        values = sorted(s.value for s in samples)
        if len(values) < self._min_samples:
            return anomalies

        q1_idx = len(values) // 4
        q3_idx = 3 * len(values) // 4
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1
        if iqr == 0:
            return anomalies

        lower = q1 - self._iqr_multiplier * iqr
        upper = q3 + self._iqr_multiplier * iqr

        for s in samples:
            if s.value < lower or s.value > upper:
                deviation = abs(s.value - (q1 + q3) / 2) / iqr
                severity = self._classify_severity(deviation)
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.LATENCY_SPIKE if s.value > upper else AnomalyType.THROUGHPUT_DROP,
                        severity=severity,
                        timestamp_s=s.timestamp_s,
                        metric_name=metric_name,
                        observed_value=s.value,
                        expected_value=(q1 + q3) / 2,
                        deviation=deviation,
                        detection_method=DetectionMethod.IQR,
                        description=(
                            f"{metric_name} = {s.value:.2f} outside IQR [{lower:.2f}, {upper:.2f}]"
                        ),
                    )
                )
        return anomalies

    def _trend_detect(
        self, metric_name: str, samples: List[MetricSample]
    ) -> List[Anomaly]:
        """Detect monotonic trends that indicate degradation (e.g. memory leaks)."""
        anomalies: List[Anomaly] = []
        if len(samples) < self._min_samples:
            return anomalies

        # Sort by time
        sorted_samples = sorted(samples, key=lambda s: s.timestamp_s)
        values = [s.value for s in sorted_samples]

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(values)
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        # Normalize slope by mean
        rel_slope = slope / y_mean if y_mean != 0 else 0.0

        if "memory" in metric_name.lower() and rel_slope > 0.01:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.MEMORY_LEAK,
                    severity=AnomalySeverity.HIGH if rel_slope > 0.05 else AnomalySeverity.MEDIUM,
                    timestamp_s=sorted_samples[-1].timestamp_s,
                    metric_name=metric_name,
                    observed_value=values[-1],
                    expected_value=values[0],
                    deviation=rel_slope * 100,
                    detection_method=DetectionMethod.TREND,
                    description=(
                        f"{metric_name} shows {rel_slope * 100:.1f}% upward trend "
                        f"({values[0]:.1f} → {values[-1]:.1f})"
                    ),
                    possible_causes=_CAUSE_MAP.get(AnomalyType.MEMORY_LEAK, []),
                    recommendation="Check for unreleased GPU tensors or growing KV cache.",
                )
            )

        return anomalies

    def _classify_severity(self, deviation: float) -> AnomalySeverity:
        if deviation > 5.0:
            return AnomalySeverity.CRITICAL
        if deviation > 3.5:
            return AnomalySeverity.HIGH
        if deviation > 2.5:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def detect(self) -> AnomalyReport:
        """Run anomaly detection across all recorded metrics."""
        all_anomalies: List[Anomaly] = []
        total_samples = sum(len(v) for v in self._samples.values())
        health: Dict[str, str] = {}

        for metric_name, samples in self._samples.items():
            z_anomalies = self._z_score_detect(metric_name, samples)
            iqr_anomalies = self._iqr_detect(metric_name, samples)
            trend_anomalies = self._trend_detect(metric_name, samples)

            # Deduplicate: keep unique by timestamp (prefer higher severity)
            seen_ts: Dict[float, Anomaly] = {}
            for a in z_anomalies + iqr_anomalies + trend_anomalies:
                key = a.timestamp_s
                if key not in seen_ts or _severity_rank(a.severity) < _severity_rank(seen_ts[key].severity):
                    seen_ts[key] = a

            metric_anomalies = list(seen_ts.values())
            all_anomalies.extend(metric_anomalies)

            # Metric health
            if not metric_anomalies:
                health[metric_name] = "healthy"
            elif any(a.severity in (AnomalySeverity.CRITICAL, AnomalySeverity.HIGH) for a in metric_anomalies):
                health[metric_name] = "critical"
            else:
                health[metric_name] = "degraded"

        # Cross-metric correlation
        self._correlate_anomalies(all_anomalies)

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for a in all_anomalies:
            by_type[a.anomaly_type.value] = by_type.get(a.anomaly_type.value, 0) + 1
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1

        return AnomalyReport(
            total_samples=total_samples,
            total_anomalies=len(all_anomalies),
            anomalies=all_anomalies,
            anomaly_rate=len(all_anomalies) / total_samples if total_samples > 0 else 0.0,
            by_type=by_type,
            by_severity=by_severity,
            metrics_health=health,
        )

    def _correlate_anomalies(self, anomalies: List[Anomaly]) -> None:
        """Find temporally correlated anomalies across metrics."""
        WINDOW_S = 5.0  # correlation window
        for i, a in enumerate(anomalies):
            for j, b in enumerate(anomalies):
                if i == j or a.metric_name == b.metric_name:
                    continue
                if abs(a.timestamp_s - b.timestamp_s) < WINDOW_S:
                    if b.metric_name not in a.correlated_metrics:
                        a.correlated_metrics.append(b.metric_name)

    def reset(self) -> None:
        self._samples.clear()


def _severity_rank(severity: AnomalySeverity) -> int:
    return {
        AnomalySeverity.CRITICAL: 0,
        AnomalySeverity.HIGH: 1,
        AnomalySeverity.MEDIUM: 2,
        AnomalySeverity.LOW: 3,
    }.get(severity, 4)
