"""
Bottleneck analyser — Stage 9.

Classifies each benchmark run into one or more bottleneck categories:

  - COMPUTE_BOUND     : GPU SMs are saturated (high utilisation, low BW)
  - MEMORY_BW_BOUND   : HBM bandwidth is saturating (decode-heavy workloads)
  - LATENCY_BOUND     : Small batch sizes prevent hiding memory latency
  - PIPELINE_BUBBLE   : PP > 1 introduces significant pipeline bubbles
  - KV_CACHE_PRESSURE : KV cache nearly full, high preemption rate
  - NETWORK_BOUND     : KV transmission dominates (disaggregated setups)
  - PREFILL_DOMINATED : Prefill time ≫ decode time (SARATHI target)
  - DECODE_DOMINATED  : Decode time ≫ prefill time (batch inference target)
  - SLO_CONSTRAINED   : Request rate limited by SLO, not by hardware

Each category carries a severity score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Bottleneck categories
# ---------------------------------------------------------------------------

class BottleneckKind(str, Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BW_BOUND = "memory_bw_bound"
    LATENCY_BOUND = "latency_bound"
    PIPELINE_BUBBLE = "pipeline_bubble"
    KV_CACHE_PRESSURE = "kv_cache_pressure"
    NETWORK_BOUND = "network_bound"
    PREFILL_DOMINATED = "prefill_dominated"
    DECODE_DOMINATED = "decode_dominated"
    SLO_CONSTRAINED = "slo_constrained"


# ---------------------------------------------------------------------------
# Detection thresholds (tunable)
# ---------------------------------------------------------------------------

# Fraction of decode-phase time that must come from KV transmission for
# NETWORK_BOUND classification.
NETWORK_BOUND_THRESHOLD = 0.20

# Minimum pipeline bubble ratio to classify as PIPELINE_BUBBLE.
PIPELINE_BUBBLE_THRESHOLD = 0.05

# Minimum preemption rate (fraction) for KV_CACHE_PRESSURE.
KV_CACHE_PRESSURE_THRESHOLD = 0.05

# Ratio of prefill% to decode% above which PREFILL_DOMINATED is flagged.
PREFILL_DOMINATED_RATIO = 2.0

# Ratio of decode% to prefill% above which DECODE_DOMINATED is flagged.
DECODE_DOMINATED_RATIO = 2.0

# SLO attainment below this threshold → SLO_CONSTRAINED.
SLO_CONSTRAINED_THRESHOLD = 95.0

# Batch size below which LATENCY_BOUND is flagged.
LATENCY_BOUND_BATCH_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BottleneckResult:
    """Bottleneck classification for a single run."""

    config: ConfigPoint
    metrics: BenchmarkMetrics
    bottlenecks: List[Dict] = field(default_factory=list)

    def add(self, kind: BottleneckKind, severity: float, reason: str) -> None:
        """Record a detected bottleneck.

        Args:
            kind: The bottleneck category.
            severity: Score in [0, 1]; 1 = maximum severity.
            reason: Human-readable explanation.
        """
        self.bottlenecks.append(
            {
                "kind": kind.value,
                "severity": round(min(max(severity, 0.0), 1.0), 4),
                "reason": reason,
            }
        )

    @property
    def primary(self) -> Optional[Dict]:
        """Return the most severe bottleneck, or None."""
        if not self.bottlenecks:
            return None
        return max(self.bottlenecks, key=lambda b: b["severity"])

    def summary(self) -> str:
        if not self.bottlenecks:
            return "No bottleneck detected"
        lines = ["Bottleneck analysis:"]
        for b in sorted(self.bottlenecks, key=lambda x: -x["severity"]):
            lines.append(
                f"  [{b['kind']}] severity={b['severity']:.2f}  — {b['reason']}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BottleneckAnalyser
# ---------------------------------------------------------------------------

class BottleneckAnalyser:
    """Identifies bottlenecks for one or more benchmark results.

    Detection rules are heuristic and intentionally conservative: a category
    is only flagged when the evidence is strong enough to guide action.
    """

    def analyse(
        self,
        cfg: ConfigPoint,
        metrics: BenchmarkMetrics,
    ) -> BottleneckResult:
        """Classify the bottlenecks for a single (config, metrics) pair.

        Args:
            cfg: Configuration point describing parallelism and batching.
            metrics: Collected benchmark metrics for this run.

        Returns:
            BottleneckResult with all detected bottleneck categories.
        """
        result = BottleneckResult(config=cfg, metrics=metrics)

        # --- NETWORK_BOUND ---
        if metrics.transmission_time_pct > NETWORK_BOUND_THRESHOLD * 100.0:
            severity = metrics.transmission_time_pct / 100.0
            result.add(
                BottleneckKind.NETWORK_BOUND,
                severity,
                f"KV transmission consumes {metrics.transmission_time_pct:.1f}% of "
                "total phase time; consider reducing TP or co-locating prefill/decode.",
            )

        # --- PIPELINE_BUBBLE ---
        bubble = metrics.pipeline_bubble_ratio or metrics.bubble_ratio
        if cfg.pp > 1 and bubble > PIPELINE_BUBBLE_THRESHOLD:
            severity = min(1.0, bubble / 0.5)
            result.add(
                BottleneckKind.PIPELINE_BUBBLE,
                severity,
                f"Pipeline bubble ratio={bubble:.3f} with PP={cfg.pp}; "
                "try smaller micro-batch size or a different batching scheme.",
            )

        # --- KV_CACHE_PRESSURE ---
        if metrics.preemption_rate > KV_CACHE_PRESSURE_THRESHOLD:
            severity = min(1.0, metrics.preemption_rate)
            result.add(
                BottleneckKind.KV_CACHE_PRESSURE,
                severity,
                f"Preemption rate={metrics.preemption_rate:.3f}; KV cache is "
                "under pressure.  Consider increasing gpu_mem_util, reducing "
                "max_num_seqs, or enabling prefix_caching.",
            )

        # --- PREFILL_DOMINATED ---
        total_pct = metrics.prefill_phase_time_pct + metrics.decode_phase_time_pct
        if total_pct > 0:
            prefill_ratio = metrics.prefill_phase_time_pct / total_pct
            decode_ratio = metrics.decode_phase_time_pct / total_pct

            if (
                decode_ratio > 0
                and prefill_ratio / decode_ratio > PREFILL_DOMINATED_RATIO
            ):
                severity = min(1.0, prefill_ratio)
                result.add(
                    BottleneckKind.PREFILL_DOMINATED,
                    severity,
                    f"Prefill consumes {metrics.prefill_phase_time_pct:.1f}% vs "
                    f"decode {metrics.decode_phase_time_pct:.1f}%.  "
                    "Enable chunked prefill or increase PP to hide prefill latency.",
                )

            # --- DECODE_DOMINATED ---
            elif (
                prefill_ratio > 0
                and decode_ratio / prefill_ratio > DECODE_DOMINATED_RATIO
            ):
                severity = min(1.0, decode_ratio)
                result.add(
                    BottleneckKind.DECODE_DOMINATED,
                    severity,
                    f"Decode consumes {metrics.decode_phase_time_pct:.1f}% vs "
                    f"prefill {metrics.prefill_phase_time_pct:.1f}%.  "
                    "Increase TP to reduce per-token decode latency.",
                )

        # --- MEMORY_BW_BOUND ---
        # Proxy: high decode time per token AND low prefill time suggests
        # memory-BW limitation rather than compute.
        if (
            metrics.decode_time_per_token_ms > 0
            and metrics.prefill_phase_time_pct < 10.0
            and metrics.decode_phase_time_pct > 60.0
        ):
            severity = min(1.0, metrics.decode_time_per_token_ms / 10.0)
            result.add(
                BottleneckKind.MEMORY_BW_BOUND,
                severity,
                f"High decode time per token ({metrics.decode_time_per_token_ms:.2f} ms) "
                "with low prefill fraction suggests HBM bandwidth saturation.  "
                "Use FP8 quantization or increase TP to spread BW across GPUs.",
            )

        # --- LATENCY_BOUND ---
        if metrics.avg_decode_batch_size > 0 and metrics.avg_decode_batch_size < LATENCY_BOUND_BATCH_THRESHOLD:
            severity = max(0.0, 1.0 - metrics.avg_decode_batch_size / LATENCY_BOUND_BATCH_THRESHOLD)
            result.add(
                BottleneckKind.LATENCY_BOUND,
                severity,
                f"Average decode batch size={metrics.avg_decode_batch_size:.1f} is very small; "
                "GPU SMs are under-utilised.  Increase max_num_seqs or request_rate_rps.",
            )

        # --- SLO_CONSTRAINED ---
        if (
            metrics.joint_slo_attainment_pct > 0
            and metrics.joint_slo_attainment_pct < SLO_CONSTRAINED_THRESHOLD
        ):
            severity = 1.0 - metrics.joint_slo_attainment_pct / 100.0
            result.add(
                BottleneckKind.SLO_CONSTRAINED,
                severity,
                f"Joint SLO attainment={metrics.joint_slo_attainment_pct:.1f}% < "
                f"{SLO_CONSTRAINED_THRESHOLD:.0f}% target.  "
                "Lower request_rate_rps or relax the SLO thresholds.",
            )

        return result

    def analyse_batch(
        self,
        results: List[tuple],
    ) -> List[BottleneckResult]:
        """Classify bottlenecks for a list of (ConfigPoint, BenchmarkMetrics).

        Args:
            results: List of (ConfigPoint, BenchmarkMetrics) tuples.

        Returns:
            List of BottleneckResult, one per input pair.
        """
        return [self.analyse(cfg, metrics) for cfg, metrics in results]

    def top_bottlenecks(
        self,
        results: List[tuple],
        top_n: int = 3,
    ) -> List[BottleneckResult]:
        """Return the *top_n* most severely bottlenecked runs.

        Args:
            results: List of (ConfigPoint, BenchmarkMetrics) tuples.
            top_n: Number of worst results to return.

        Returns:
            Up to *top_n* BottleneckResult instances sorted by primary severity.
        """
        analysed = self.analyse_batch(results)
        scored = [
            (r, r.primary["severity"] if r.primary else 0.0)
            for r in analysed
        ]
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[:top_n]]
