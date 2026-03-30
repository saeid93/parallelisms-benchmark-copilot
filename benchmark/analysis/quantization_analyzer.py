"""Quantization impact analyzer for LLM serving.

Inspired by:
* GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers (Frantar et al. 2022)
* AWQ: Activation-aware Weight Quantization (Lin et al. 2023)
* SmoothQuant: Accurate and Efficient Post-Training Quantization (Xiao et al. 2023)
* FP8 Formats for Deep Learning (Micikevicius et al. 2022)
* "A Survey of Quantization Methods for Efficient Neural Network Inference"

Analyses quality degradation vs speedup tradeoffs, estimates
perplexity impact, identifies layer-wise quantization sensitivity,
and recommends mixed-precision strategies.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class QuantMethod(str, Enum):
    """Quantization methods."""

    NONE = "none"
    GPTQ = "gptq"
    AWQ = "awq"
    SMOOTHQUANT = "smoothquant"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"
    BITSANDBYTES_NF4 = "bitsandbytes_nf4"
    GPTQ_MARLIN = "gptq_marlin"
    AWQ_MARLIN = "awq_marlin"
    COMPRESSED_TENSORS = "compressed_tensors"


class QuantGranularity(str, Enum):
    """Quantization granularity."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"
    PER_TOKEN = "per_token"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class QuantizationResult:
    """Results for a single quantization configuration."""

    method: QuantMethod
    bits: int = 16
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    group_size: int = 128
    throughput_tps: float = 0.0
    baseline_throughput_tps: float = 0.0
    latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    perplexity: float = 0.0
    baseline_perplexity: float = 0.0
    model_size_gb: float = 0.0
    baseline_model_size_gb: float = 0.0
    memory_savings_pct: float = 0.0
    accuracy_metric: float = 0.0
    baseline_accuracy: float = 0.0

    @property
    def speedup(self) -> float:
        if self.baseline_throughput_tps <= 0:
            return 0.0
        return self.throughput_tps / self.baseline_throughput_tps

    @property
    def latency_reduction_pct(self) -> float:
        if self.baseline_latency_ms <= 0:
            return 0.0
        return 100.0 * (1.0 - self.latency_ms / self.baseline_latency_ms)

    @property
    def perplexity_increase_pct(self) -> float:
        if self.baseline_perplexity <= 0:
            return 0.0
        return 100.0 * (self.perplexity - self.baseline_perplexity) / self.baseline_perplexity

    @property
    def accuracy_drop_pct(self) -> float:
        if self.baseline_accuracy <= 0:
            return 0.0
        return 100.0 * (self.baseline_accuracy - self.accuracy_metric) / self.baseline_accuracy

    @property
    def compression_ratio(self) -> float:
        if self.model_size_gb <= 0:
            return 0.0
        return self.baseline_model_size_gb / self.model_size_gb


@dataclass
class LayerSensitivity:
    """Quantization sensitivity analysis per layer."""

    layer_index: int
    layer_name: str = ""
    weight_range: float = 0.0
    activation_range: float = 0.0
    sensitivity_score: float = 0.0  # higher = more sensitive
    recommended_bits: int = 16
    perplexity_delta: float = 0.0  # increase when this layer is quantized


@dataclass
class MixedPrecisionPlan:
    """Recommended mixed-precision quantization plan."""

    layer_configs: Dict[int, int] = field(default_factory=dict)  # layer_idx -> bits
    estimated_speedup: float = 0.0
    estimated_perplexity_increase: float = 0.0
    estimated_memory_savings_pct: float = 0.0
    sensitive_layers: List[int] = field(default_factory=list)
    aggressive_layers: List[int] = field(default_factory=list)


@dataclass
class QuantizationProfile:
    """Full quantization impact analysis."""

    results: List[QuantizationResult] = field(default_factory=list)
    layer_sensitivities: List[LayerSensitivity] = field(default_factory=list)
    mixed_precision_plan: Optional[MixedPrecisionPlan] = None
    best_quality_method: Optional[QuantMethod] = None
    best_speed_method: Optional[QuantMethod] = None
    best_balanced_method: Optional[QuantMethod] = None
    pareto_frontier: List[QuantizationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Quantization Impact Analysis",
            f"  Methods compared    : {len(self.results)}",
        ]
        for r in self.results:
            lines.append(
                f"  {r.method.value} ({r.bits}b): "
                f"speedup={r.speedup:.2f}x, "
                f"ppl+={r.perplexity_increase_pct:.1f}%, "
                f"mem-={r.memory_savings_pct:.0f}%"
            )
        if self.best_quality_method:
            lines.append(f"  Best quality   : {self.best_quality_method.value}")
        if self.best_speed_method:
            lines.append(f"  Best speed     : {self.best_speed_method.value}")
        if self.best_balanced_method:
            lines.append(f"  Best balanced  : {self.best_balanced_method.value}")
        if self.layer_sensitivities:
            top = sorted(self.layer_sensitivities, key=lambda l: l.sensitivity_score, reverse=True)[:3]
            lines.append("  Most sensitive layers:")
            for ls in top:
                lines.append(f"    Layer {ls.layer_index}: score={ls.sensitivity_score:.3f}, rec={ls.recommended_bits}b")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class QuantizationAnalyzer:
    """Analyses quantization impact on quality, speed, and memory.

    Usage::

        qa = QuantizationAnalyzer()
        qa.add_result(QuantizationResult(method=QuantMethod.NONE, ...))
        qa.add_result(QuantizationResult(method=QuantMethod.AWQ, ...))
        qa.add_layer_sensitivity(LayerSensitivity(layer_index=0, ...))
        profile = qa.analyse()
        print(profile.summary())
    """

    def __init__(
        self,
        perplexity_threshold: float = 5.0,
        min_speedup: float = 1.1,
    ) -> None:
        self._ppl_threshold = perplexity_threshold  # max acceptable ppl increase %
        self._min_speedup = min_speedup
        self._results: List[QuantizationResult] = []
        self._sensitivities: List[LayerSensitivity] = []

    def add_result(self, result: QuantizationResult) -> None:
        self._results.append(result)

    def add_layer_sensitivity(self, sensitivity: LayerSensitivity) -> None:
        self._sensitivities.append(sensitivity)

    def analyse(self) -> QuantizationProfile:
        if not self._results:
            return QuantizationProfile()

        # Find Pareto frontier (maximize speedup, minimize perplexity increase)
        pareto = self._compute_pareto()

        # Best by category
        valid = [r for r in self._results if r.method != QuantMethod.NONE]

        best_quality = None
        best_speed = None
        best_balanced = None

        if valid:
            # Best quality: lowest perplexity increase among those with speedup
            quality_sorted = sorted(valid, key=lambda r: r.perplexity_increase_pct)
            best_quality = quality_sorted[0].method

            # Best speed: highest speedup
            speed_sorted = sorted(valid, key=lambda r: r.speedup, reverse=True)
            best_speed = speed_sorted[0].method

            # Best balanced: best (speedup * (1 - ppl_increase/100))
            def balance_score(r: QuantizationResult) -> float:
                ppl_factor = max(0.0, 1.0 - r.perplexity_increase_pct / 100.0)
                return r.speedup * ppl_factor

            balanced_sorted = sorted(valid, key=balance_score, reverse=True)
            best_balanced = balanced_sorted[0].method

        # Mixed precision plan
        mixed = self._compute_mixed_precision()

        recs = self._generate_recommendations(pareto, best_balanced)

        return QuantizationProfile(
            results=self._results,
            layer_sensitivities=self._sensitivities,
            mixed_precision_plan=mixed,
            best_quality_method=best_quality,
            best_speed_method=best_speed,
            best_balanced_method=best_balanced,
            pareto_frontier=pareto,
            recommendations=recs,
        )

    def _compute_pareto(self) -> List[QuantizationResult]:
        """Find Pareto-optimal quantization configurations."""
        pareto: List[QuantizationResult] = []
        for r in self._results:
            dominated = False
            for other in self._results:
                if other is r:
                    continue
                # other dominates r if better speedup AND lower ppl increase
                if (
                    other.speedup >= r.speedup
                    and other.perplexity_increase_pct <= r.perplexity_increase_pct
                    and (other.speedup > r.speedup or other.perplexity_increase_pct < r.perplexity_increase_pct)
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto

    def _compute_mixed_precision(self) -> Optional[MixedPrecisionPlan]:
        """Recommend mixed-precision plan based on layer sensitivity."""
        if not self._sensitivities:
            return None

        sorted_sens = sorted(self._sensitivities, key=lambda s: s.sensitivity_score, reverse=True)
        top_20_pct = max(1, len(sorted_sens) // 5)

        sensitive = sorted_sens[:top_20_pct]
        aggressive = sorted_sens[top_20_pct:]

        layer_configs: Dict[int, int] = {}
        for ls in sensitive:
            layer_configs[ls.layer_index] = 16  # keep FP16
        for ls in aggressive:
            layer_configs[ls.layer_index] = ls.recommended_bits

        # Estimate impact
        ppl_increase = sum(ls.perplexity_delta for ls in aggressive) if aggressive else 0.0
        savings = sum(1 for ls in aggressive if ls.recommended_bits < 16) / len(sorted_sens) * 50.0

        return MixedPrecisionPlan(
            layer_configs=layer_configs,
            estimated_perplexity_increase=ppl_increase,
            estimated_memory_savings_pct=savings,
            sensitive_layers=[ls.layer_index for ls in sensitive],
            aggressive_layers=[ls.layer_index for ls in aggressive],
        )

    def _generate_recommendations(
        self,
        pareto: List[QuantizationResult],
        best_balanced: Optional[QuantMethod],
    ) -> List[str]:
        recs: List[str] = []

        if best_balanced:
            recs.append(
                f"Recommended method: {best_balanced.value} provides the best "
                "balance of speed and quality."
            )

        # Check if FP8 is in Pareto
        fp8_in_pareto = any(
            r.method in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2)
            for r in pareto
        )
        if fp8_in_pareto:
            recs.append(
                "FP8 quantization is Pareto-optimal for this model. "
                "Use FP8 E4M3 for weights and E5M2 for activations."
            )

        # Check high perplexity increase
        for r in self._results:
            if r.perplexity_increase_pct > 10.0:
                recs.append(
                    f"{r.method.value} causes {r.perplexity_increase_pct:.1f}% perplexity increase. "
                    "Consider using a calibration dataset or mixed precision."
                )
                break

        if self._sensitivities:
            top_sensitive = sorted(
                self._sensitivities, key=lambda s: s.sensitivity_score, reverse=True
            )[:3]
            sensitive_names = [f"layer {s.layer_index}" for s in top_sensitive]
            recs.append(
                f"Most quantization-sensitive layers: {', '.join(sensitive_names)}. "
                "Keep these in higher precision."
            )

        return recs

    def reset(self) -> None:
        self._results.clear()
        self._sensitivities.clear()
