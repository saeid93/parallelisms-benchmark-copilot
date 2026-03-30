"""Attention mechanism profiler for LLM serving.

Inspired by:
* "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al. 2022)
* "FlashAttention-2: Faster Attention with Better Parallelism" (Dao 2023)
* "PagedAttention" (Kwon et al. 2023, vLLM)
* "Efficient Transformers: A Survey" (Tay et al. 2022)

Profiles attention kernel execution, memory access patterns,
compute vs memory bound classification per head/layer, and
attention score sparsity analysis.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AttentionBackend(str, Enum):
    """Attention backend implementations."""

    FLASH_ATTN_V2 = "flash_attn_v2"
    FLASH_ATTN_V3 = "flash_attn_v3"
    FLASH_ATTN_V4 = "flash_attn_v4"
    FLASHINFER = "flashinfer"
    TRITON = "triton"
    XFORMERS = "xformers"
    PAGED_ATTENTION = "paged_attention"
    NAIVE = "naive"


class AttentionPhase(str, Enum):
    """Attention computation phases."""

    QK_MATMUL = "qk_matmul"
    SOFTMAX = "softmax"
    SCORE_V_MATMUL = "score_v_matmul"
    MASK_APPLY = "mask_apply"
    ROPE = "rope"
    KV_CACHE_READ = "kv_cache_read"
    KV_CACHE_WRITE = "kv_cache_write"
    OUTPUT_PROJ = "output_proj"
    FULL_ATTENTION = "full_attention"


class BoundClassification(str, Enum):
    """Whether an operation is compute or memory bound."""

    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    BALANCED = "balanced"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class AttentionKernelEvent:
    """A single attention kernel execution record."""

    layer_index: int
    head_index: int
    phase: AttentionPhase
    backend: AttentionBackend
    start_us: float
    duration_us: float
    seq_len: int = 0
    kv_len: int = 0
    head_dim: int = 128
    num_heads: int = 32
    batch_size: int = 1
    flops: float = 0.0
    bytes_accessed: float = 0.0
    is_prefill: bool = True
    request_id: str = ""

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP/byte ratio — key metric for roofline analysis."""
        if self.bytes_accessed <= 0:
            return 0.0
        return self.flops / self.bytes_accessed

    @property
    def tflops(self) -> float:
        dur_s = self.duration_us / 1e6
        if dur_s <= 0:
            return 0.0
        return self.flops / (dur_s * 1e12)

    @property
    def bandwidth_gbps(self) -> float:
        dur_s = self.duration_us / 1e6
        if dur_s <= 0:
            return 0.0
        return self.bytes_accessed / (dur_s * 1e9)


@dataclass
class SparsityAnalysis:
    """Attention score sparsity for a layer/head."""

    layer_index: int
    head_index: int
    total_elements: int = 0
    near_zero_elements: int = 0
    sparsity_ratio: float = 0.0
    top_k_concentration: float = 0.0  # fraction of attention in top-k positions
    top_k: int = 32
    entropy: float = 0.0
    is_local_pattern: bool = False
    effective_context_length: int = 0


@dataclass
class AttentionLayerStats:
    """Per-layer attention statistics."""

    layer_index: int
    num_events: int = 0
    total_time_us: float = 0.0
    mean_time_us: float = 0.0
    prefill_time_us: float = 0.0
    decode_time_us: float = 0.0
    bound_classification: BoundClassification = BoundClassification.BALANCED
    mean_arithmetic_intensity: float = 0.0
    mean_tflops: float = 0.0
    mean_bandwidth_gbps: float = 0.0
    phase_breakdown: Dict[str, float] = field(default_factory=dict)
    per_head_sparsity: List[SparsityAnalysis] = field(default_factory=list)


@dataclass
class AttentionProfile:
    """Full attention profiling results."""

    total_attention_time_us: float = 0.0
    total_prefill_attn_us: float = 0.0
    total_decode_attn_us: float = 0.0
    per_layer_stats: List[AttentionLayerStats] = field(default_factory=list)
    backend: str = ""
    overall_bound: BoundClassification = BoundClassification.BALANCED
    mean_arithmetic_intensity: float = 0.0
    mean_sparsity: float = 0.0
    kv_cache_read_time_us: float = 0.0
    kv_cache_write_time_us: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Attention Profile",
            f"  Backend              : {self.backend}",
            f"  Total attn time      : {self.total_attention_time_us / 1e3:.1f} ms",
            f"  Prefill attn time    : {self.total_prefill_attn_us / 1e3:.1f} ms",
            f"  Decode attn time     : {self.total_decode_attn_us / 1e3:.1f} ms",
            f"  KV cache read time   : {self.kv_cache_read_time_us / 1e3:.1f} ms",
            f"  KV cache write time  : {self.kv_cache_write_time_us / 1e3:.1f} ms",
            f"  Overall bound        : {self.overall_bound.value}",
            f"  Arith. intensity     : {self.mean_arithmetic_intensity:.2f} FLOP/byte",
            f"  Mean sparsity        : {self.mean_sparsity:.1%}",
            f"  Layers profiled      : {len(self.per_layer_stats)}",
        ]
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# Roofline thresholds for common GPUs (TFLOPS, TB/s)
GPU_ROOFLINES = {
    "A100": {"peak_tflops_fp16": 312.0, "peak_bw_tbs": 2.0},
    "H100": {"peak_tflops_fp16": 989.0, "peak_bw_tbs": 3.35},
    "L40S": {"peak_tflops_fp16": 362.0, "peak_bw_tbs": 0.864},
    "A10G": {"peak_tflops_fp16": 125.0, "peak_bw_tbs": 0.6},
}


class AttentionProfiler:
    """Profiles attention kernel execution and memory patterns.

    Usage::

        ap = AttentionProfiler(gpu_type="H100")
        ap.record(AttentionKernelEvent(...))
        ap.record_sparsity(SparsityAnalysis(...))
        profile = ap.analyse()
        print(profile.summary())
    """

    def __init__(self, gpu_type: str = "A100", num_layers: int = 32) -> None:
        self._gpu_type = gpu_type
        self._num_layers = num_layers
        self._events: List[AttentionKernelEvent] = []
        self._sparsity: List[SparsityAnalysis] = []
        self._roofline = GPU_ROOFLINES.get(gpu_type, GPU_ROOFLINES["A100"])

    def record(self, event: AttentionKernelEvent) -> None:
        """Record an attention kernel event."""
        self._events.append(event)

    def record_batch(self, events: List[AttentionKernelEvent]) -> None:
        """Record multiple events."""
        self._events.extend(events)

    def record_sparsity(self, analysis: SparsityAnalysis) -> None:
        """Record sparsity analysis for a layer/head."""
        self._sparsity.append(analysis)

    def analyse(self) -> AttentionProfile:
        """Analyse all recorded attention events."""
        if not self._events:
            return AttentionProfile()

        # Group by layer
        by_layer: Dict[int, List[AttentionKernelEvent]] = {}
        for e in self._events:
            by_layer.setdefault(e.layer_index, []).append(e)

        total_time = sum(e.duration_us for e in self._events)
        prefill_time = sum(e.duration_us for e in self._events if e.is_prefill)
        decode_time = sum(e.duration_us for e in self._events if not e.is_prefill)
        kv_read = sum(e.duration_us for e in self._events if e.phase == AttentionPhase.KV_CACHE_READ)
        kv_write = sum(e.duration_us for e in self._events if e.phase == AttentionPhase.KV_CACHE_WRITE)

        layer_stats: List[AttentionLayerStats] = []
        all_ai: List[float] = []

        for layer_idx in sorted(by_layer.keys()):
            events = by_layer[layer_idx]
            durations = [e.duration_us for e in events]
            ais = [e.arithmetic_intensity for e in events if e.arithmetic_intensity > 0]
            tfl = [e.tflops for e in events if e.tflops > 0]
            bws = [e.bandwidth_gbps for e in events if e.bandwidth_gbps > 0]

            all_ai.extend(ais)

            # Phase breakdown
            phase_times: Dict[str, float] = {}
            for e in events:
                phase_times[e.phase.value] = phase_times.get(e.phase.value, 0.0) + e.duration_us

            # Bound classification
            mean_ai = statistics.mean(ais) if ais else 0.0
            ridge_point = self._roofline["peak_tflops_fp16"] / self._roofline["peak_bw_tbs"]
            if mean_ai > 0 and mean_ai < ridge_point * 0.5:
                bound = BoundClassification.MEMORY_BOUND
            elif mean_ai > ridge_point * 1.5:
                bound = BoundClassification.COMPUTE_BOUND
            else:
                bound = BoundClassification.BALANCED

            # Sparsity for this layer
            layer_sparsity = [s for s in self._sparsity if s.layer_index == layer_idx]

            layer_stat = AttentionLayerStats(
                layer_index=layer_idx,
                num_events=len(events),
                total_time_us=sum(durations),
                mean_time_us=statistics.mean(durations) if durations else 0.0,
                prefill_time_us=sum(e.duration_us for e in events if e.is_prefill),
                decode_time_us=sum(e.duration_us for e in events if not e.is_prefill),
                bound_classification=bound,
                mean_arithmetic_intensity=mean_ai,
                mean_tflops=statistics.mean(tfl) if tfl else 0.0,
                mean_bandwidth_gbps=statistics.mean(bws) if bws else 0.0,
                phase_breakdown=phase_times,
                per_head_sparsity=layer_sparsity,
            )
            layer_stats.append(layer_stat)

        # Overall bound
        overall_ai = statistics.mean(all_ai) if all_ai else 0.0
        ridge = self._roofline["peak_tflops_fp16"] / self._roofline["peak_bw_tbs"]
        if overall_ai > 0 and overall_ai < ridge * 0.5:
            overall_bound = BoundClassification.MEMORY_BOUND
        elif overall_ai > ridge * 1.5:
            overall_bound = BoundClassification.COMPUTE_BOUND
        else:
            overall_bound = BoundClassification.BALANCED

        # Sparsity
        mean_sparsity = (
            statistics.mean([s.sparsity_ratio for s in self._sparsity])
            if self._sparsity
            else 0.0
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            layer_stats, overall_bound, mean_sparsity
        )

        backend_name = ""
        if self._events:
            backend_name = self._events[0].backend.value

        return AttentionProfile(
            total_attention_time_us=total_time,
            total_prefill_attn_us=prefill_time,
            total_decode_attn_us=decode_time,
            per_layer_stats=layer_stats,
            backend=backend_name,
            overall_bound=overall_bound,
            mean_arithmetic_intensity=overall_ai,
            mean_sparsity=mean_sparsity,
            kv_cache_read_time_us=kv_read,
            kv_cache_write_time_us=kv_write,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        layer_stats: List[AttentionLayerStats],
        overall_bound: BoundClassification,
        mean_sparsity: float,
    ) -> List[str]:
        recs: List[str] = []

        if overall_bound == BoundClassification.MEMORY_BOUND:
            recs.append(
                "Attention is memory-bound. Consider FlashAttention to reduce HBM reads "
                "or increase batch size to improve arithmetic intensity."
            )
        elif overall_bound == BoundClassification.COMPUTE_BOUND:
            recs.append(
                "Attention is compute-bound. Consider FP8 attention or quantized KV cache "
                "to reduce FLOP count."
            )

        if mean_sparsity > 0.8:
            recs.append(
                f"High attention sparsity ({mean_sparsity:.0%}). Sparse attention patterns "
                "like local + global or top-k attention could save significant compute."
            )

        # Check for layer imbalance
        if len(layer_stats) >= 2:
            times = [ls.total_time_us for ls in layer_stats]
            if times:
                mean_t = statistics.mean(times)
                max_t = max(times)
                if mean_t > 0 and max_t > 2 * mean_t:
                    slow_layer = layer_stats[times.index(max_t)].layer_index
                    recs.append(
                        f"Layer {slow_layer} is {max_t / mean_t:.1f}x slower than average. "
                        "Investigate if GQA or different head configuration would help."
                    )

        return recs

    def export_roofline_data(self) -> List[Dict]:
        """Export data points for roofline model plotting."""
        return [
            {
                "layer": e.layer_index,
                "head": e.head_index,
                "phase": e.phase.value,
                "arithmetic_intensity": e.arithmetic_intensity,
                "achieved_tflops": e.tflops,
                "is_prefill": e.is_prefill,
                "seq_len": e.seq_len,
            }
            for e in self._events
            if e.arithmetic_intensity > 0
        ]

    def reset(self) -> None:
        """Clear all recorded data."""
        self._events.clear()
        self._sparsity.clear()

    def __len__(self) -> int:
        return len(self._events)
