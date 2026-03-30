"""Roofline model analyzer for LLM serving workloads.

Inspired by:
* Berkeley Roofline Model (Williams et al. 2009)
* "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
* Empirical Roofline Toolkit (ERT)
* NVIDIA NSight Compute roofline analysis

Maps operations to the roofline model to classify whether each
kernel / phase is compute-bound or memory-bound, and quantifies
the gap to peak performance.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OperationType(str, Enum):
    """Types of operations in LLM inference."""

    GEMM = "gemm"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    ALLREDUCE = "allreduce"
    KV_CACHE = "kv_cache"
    ROPE = "rope"
    FFN = "ffn"
    OTHER = "other"


class BoundType(str, Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"


# ---------------------------------------------------------------------------
# GPU Specs
# ---------------------------------------------------------------------------


@dataclass
class GPUSpec:
    """Hardware specifications for roofline analysis."""

    name: str
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    peak_int8_tops: float
    hbm_bandwidth_tbs: float
    l2_cache_bandwidth_tbs: float = 0.0
    l1_cache_bandwidth_tbs: float = 0.0
    nvlink_bandwidth_gbs: float = 0.0
    pcie_bandwidth_gbs: float = 0.0
    hbm_size_gb: float = 0.0

    @property
    def ridge_point_fp16(self) -> float:
        """Arithmetic intensity at which we transition from mem to compute bound."""
        if self.hbm_bandwidth_tbs <= 0:
            return 0.0
        return self.peak_fp16_tflops / self.hbm_bandwidth_tbs


GPU_SPECS = {
    "A100_80GB": GPUSpec(
        name="A100 80GB SXM",
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        peak_int8_tops=624.0,
        hbm_bandwidth_tbs=2.039,
        l2_cache_bandwidth_tbs=8.0,
        nvlink_bandwidth_gbs=600.0,
        pcie_bandwidth_gbs=64.0,
        hbm_size_gb=80.0,
    ),
    "H100_SXM": GPUSpec(
        name="H100 SXM",
        peak_fp16_tflops=989.0,
        peak_fp32_tflops=67.0,
        peak_int8_tops=1979.0,
        hbm_bandwidth_tbs=3.35,
        l2_cache_bandwidth_tbs=12.0,
        nvlink_bandwidth_gbs=900.0,
        pcie_bandwidth_gbs=128.0,
        hbm_size_gb=80.0,
    ),
    "L40S": GPUSpec(
        name="L40S",
        peak_fp16_tflops=362.0,
        peak_fp32_tflops=91.6,
        peak_int8_tops=724.0,
        hbm_bandwidth_tbs=0.864,
        hbm_size_gb=48.0,
    ),
    "A10G": GPUSpec(
        name="A10G",
        peak_fp16_tflops=125.0,
        peak_fp32_tflops=31.2,
        peak_int8_tops=250.0,
        hbm_bandwidth_tbs=0.6,
        hbm_size_gb=24.0,
    ),
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class RooflineDataPoint:
    """A single operation mapped to the roofline model."""

    operation: OperationType
    label: str = ""
    flops: float = 0.0
    bytes_accessed: float = 0.0
    duration_s: float = 0.0
    layer_index: int = 0
    is_prefill: bool = True
    batch_size: int = 1
    seq_len: int = 0

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_accessed <= 0:
            return 0.0
        return self.flops / self.bytes_accessed

    @property
    def achieved_tflops(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.flops / (self.duration_s * 1e12)

    @property
    def achieved_bandwidth_tbs(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.bytes_accessed / (self.duration_s * 1e12)


@dataclass
class RooflineResult:
    """Classification result for a single data point."""

    point: RooflineDataPoint
    bound_type: BoundType = BoundType.MEMORY_BOUND
    efficiency_pct: float = 0.0  # fraction of peak achieved
    gap_to_peak_pct: float = 0.0
    attainable_tflops: float = 0.0
    bottleneck_resource: str = ""


@dataclass
class RooflineProfile:
    """Full roofline analysis results."""

    gpu_name: str = ""
    ridge_point: float = 0.0
    total_operations: int = 0
    compute_bound_count: int = 0
    memory_bound_count: int = 0
    mean_efficiency_pct: float = 0.0
    per_op_type_efficiency: Dict[str, float] = field(default_factory=dict)
    results: List[RooflineResult] = field(default_factory=list)
    prefill_vs_decode_bound: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Roofline Analysis ({self.gpu_name})",
            f"  Ridge point         : {self.ridge_point:.1f} FLOP/byte",
            f"  Total operations    : {self.total_operations}",
            f"  Compute-bound       : {self.compute_bound_count}",
            f"  Memory-bound        : {self.memory_bound_count}",
            f"  Mean efficiency     : {self.mean_efficiency_pct:.1f}%",
        ]
        if self.per_op_type_efficiency:
            lines.append("  Per-op efficiency:")
            for op, eff in sorted(self.per_op_type_efficiency.items()):
                lines.append(f"    {op}: {eff:.1f}%")
        if self.prefill_vs_decode_bound:
            lines.append("  Phase classification:")
            for phase, bound in self.prefill_vs_decode_bound.items():
                lines.append(f"    {phase}: {bound}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class RooflineAnalyzer:
    """Maps operations to roofline model and identifies bottlenecks.

    Usage::

        ra = RooflineAnalyzer(gpu="H100_SXM")
        ra.add_point(RooflineDataPoint(operation=OperationType.GEMM, ...))
        profile = ra.analyse()
        print(profile.summary())
    """

    def __init__(self, gpu: str = "A100_80GB") -> None:
        self._spec = GPU_SPECS.get(gpu, GPU_SPECS["A100_80GB"])
        self._points: List[RooflineDataPoint] = []

    def add_point(self, point: RooflineDataPoint) -> None:
        self._points.append(point)

    def add_points(self, points: List[RooflineDataPoint]) -> None:
        self._points.extend(points)

    def classify_point(self, point: RooflineDataPoint) -> RooflineResult:
        """Classify a single point on the roofline model."""
        ai = point.arithmetic_intensity
        achieved = point.achieved_tflops
        ridge = self._spec.ridge_point_fp16

        # attainable performance = min(peak_compute, peak_bw * AI)
        attainable = min(
            self._spec.peak_fp16_tflops,
            self._spec.hbm_bandwidth_tbs * ai if ai > 0 else 0.0,
        )

        if ai < ridge:
            bound = BoundType.MEMORY_BOUND
            bottleneck = "HBM bandwidth"
        else:
            bound = BoundType.COMPUTE_BOUND
            bottleneck = "Compute (FP16 TFLOPS)"

        eff = 100.0 * achieved / attainable if attainable > 0 else 0.0
        gap = 100.0 - eff

        return RooflineResult(
            point=point,
            bound_type=bound,
            efficiency_pct=min(100.0, eff),
            gap_to_peak_pct=max(0.0, gap),
            attainable_tflops=attainable,
            bottleneck_resource=bottleneck,
        )

    def analyse(self) -> RooflineProfile:
        """Analyse all recorded data points."""
        if not self._points:
            return RooflineProfile(
                gpu_name=self._spec.name,
                ridge_point=self._spec.ridge_point_fp16,
            )

        results = [self.classify_point(p) for p in self._points]
        cb = sum(1 for r in results if r.bound_type == BoundType.COMPUTE_BOUND)
        mb = sum(1 for r in results if r.bound_type == BoundType.MEMORY_BOUND)
        effs = [r.efficiency_pct for r in results]

        # Per-op efficiency
        by_op: Dict[str, List[float]] = {}
        for r in results:
            by_op.setdefault(r.point.operation.value, []).append(r.efficiency_pct)
        per_op_eff = {op: statistics.mean(vals) for op, vals in by_op.items()}

        # Prefill vs decode
        prefill_results = [r for r in results if r.point.is_prefill]
        decode_results = [r for r in results if not r.point.is_prefill]
        pvd: Dict[str, str] = {}
        if prefill_results:
            p_cb = sum(1 for r in prefill_results if r.bound_type == BoundType.COMPUTE_BOUND)
            pvd["prefill"] = (
                "compute_bound" if p_cb > len(prefill_results) / 2 else "memory_bound"
            )
        if decode_results:
            d_cb = sum(1 for r in decode_results if r.bound_type == BoundType.COMPUTE_BOUND)
            pvd["decode"] = (
                "compute_bound" if d_cb > len(decode_results) / 2 else "memory_bound"
            )

        recs = self._generate_recommendations(results, per_op_eff, pvd)

        return RooflineProfile(
            gpu_name=self._spec.name,
            ridge_point=self._spec.ridge_point_fp16,
            total_operations=len(results),
            compute_bound_count=cb,
            memory_bound_count=mb,
            mean_efficiency_pct=statistics.mean(effs) if effs else 0.0,
            per_op_type_efficiency=per_op_eff,
            results=results,
            prefill_vs_decode_bound=pvd,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        results: List[RooflineResult],
        per_op_eff: Dict[str, float],
        pvd: Dict[str, str],
    ) -> List[str]:
        recs: List[str] = []

        # Low overall efficiency
        effs = [r.efficiency_pct for r in results]
        mean_eff = statistics.mean(effs) if effs else 0.0
        if mean_eff < 30.0:
            recs.append(
                f"Overall efficiency is low ({mean_eff:.0f}%). "
                "Consider operator fusion, larger batch sizes, or quantization."
            )

        # Memory-bound GEMMs
        gemm_eff = per_op_eff.get(OperationType.GEMM.value, 0)
        if gemm_eff > 0 and gemm_eff < 50:
            recs.append(
                f"GEMM efficiency is only {gemm_eff:.0f}%. "
                "Increase batch size to improve arithmetic intensity."
            )

        # Decode phase mostly memory bound
        if pvd.get("decode") == "memory_bound":
            recs.append(
                "Decode phase is memory-bound (expected for autoregressive decoding). "
                "Consider speculative decoding or continuous batching to improve utilization."
            )

        # Attention low efficiency
        attn_eff = per_op_eff.get(OperationType.ATTENTION.value, 0)
        if attn_eff > 0 and attn_eff < 40:
            recs.append(
                f"Attention efficiency is {attn_eff:.0f}%. "
                "Ensure FlashAttention is enabled and batch queries where possible."
            )

        return recs

    def export_plot_data(self) -> Dict:
        """Export data for roofline plot generation."""
        return {
            "gpu_name": self._spec.name,
            "peak_fp16_tflops": self._spec.peak_fp16_tflops,
            "hbm_bandwidth_tbs": self._spec.hbm_bandwidth_tbs,
            "ridge_point": self._spec.ridge_point_fp16,
            "points": [
                {
                    "label": r.point.label or r.point.operation.value,
                    "arithmetic_intensity": r.point.arithmetic_intensity,
                    "achieved_tflops": r.point.achieved_tflops,
                    "bound_type": r.bound_type.value,
                    "efficiency_pct": r.efficiency_pct,
                    "is_prefill": r.point.is_prefill,
                    "operation": r.point.operation.value,
                }
                for r in [self.classify_point(p) for p in self._points]
            ],
        }

    def reset(self) -> None:
        self._points.clear()
