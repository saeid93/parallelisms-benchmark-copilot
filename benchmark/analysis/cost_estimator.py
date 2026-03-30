"""
Cost estimator — Stage 10.

Translates benchmark results into estimated cloud GPU costs.  Supports
common GPU instance types (A100, H100, A10G) across major cloud providers.
For each benchmark config, computes:
  - cost per 1 M tokens generated
  - cost per hour at measured goodput
  - GPU efficiency ratio (goodput tokens / theoretical GPU peak)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# GPU instance pricing catalogue (USD per GPU per hour, on-demand)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUInstanceType:
    """Specification for a cloud GPU instance."""

    name: str
    provider: str
    gpu_model: str
    price_per_gpu_hour_usd: float
    peak_tflops_bf16: float      # Peak BF16 TFLOP/s per GPU
    memory_bandwidth_gbs: float  # Peak HBM bandwidth GB/s per GPU
    gpu_memory_gb: float         # HBM capacity per GPU

    def cost_per_hour(self, num_gpus: int) -> float:
        """Total instance cost per hour for *num_gpus* GPUs."""
        return self.price_per_gpu_hour_usd * num_gpus


# Catalogue of representative GPU instances (prices as of 2025 Q1).
GPU_CATALOGUE: Dict[str, GPUInstanceType] = {
    "a100_sxm4_80gb": GPUInstanceType(
        name="a100_sxm4_80gb",
        provider="AWS/GCP/Azure",
        gpu_model="NVIDIA A100 SXM4 80GB",
        price_per_gpu_hour_usd=3.21,
        peak_tflops_bf16=312.0,
        memory_bandwidth_gbs=2000.0,
        gpu_memory_gb=80.0,
    ),
    "h100_sxm5_80gb": GPUInstanceType(
        name="h100_sxm5_80gb",
        provider="AWS/GCP/Azure",
        gpu_model="NVIDIA H100 SXM5 80GB",
        price_per_gpu_hour_usd=8.00,
        peak_tflops_bf16=989.0,
        memory_bandwidth_gbs=3350.0,
        gpu_memory_gb=80.0,
    ),
    "a10g_24gb": GPUInstanceType(
        name="a10g_24gb",
        provider="AWS",
        gpu_model="NVIDIA A10G 24GB",
        price_per_gpu_hour_usd=1.006,
        peak_tflops_bf16=125.0,
        memory_bandwidth_gbs=600.0,
        gpu_memory_gb=24.0,
    ),
    "l4_24gb": GPUInstanceType(
        name="l4_24gb",
        provider="GCP",
        gpu_model="NVIDIA L4 24GB",
        price_per_gpu_hour_usd=0.54,
        peak_tflops_bf16=242.0,
        memory_bandwidth_gbs=300.0,
        gpu_memory_gb=24.0,
    ),
    "l40s_48gb": GPUInstanceType(
        name="l40s_48gb",
        provider="GCP/Azure",
        gpu_model="NVIDIA L40S 48GB",
        price_per_gpu_hour_usd=2.00,
        peak_tflops_bf16=362.0,
        memory_bandwidth_gbs=864.0,
        gpu_memory_gb=48.0,
    ),
}

DEFAULT_GPU_INSTANCE = "a100_sxm4_80gb"


# ---------------------------------------------------------------------------
# Cost estimate result
# ---------------------------------------------------------------------------

@dataclass
class CostEstimate:
    """Cost estimate for a single benchmark run."""

    config: ConfigPoint
    metrics: BenchmarkMetrics
    gpu_instance: GPUInstanceType
    num_gpus: int

    # Derived costs
    cost_per_hour_usd: float = 0.0
    cost_per_million_tokens_usd: float = 0.0
    cost_per_million_requests_usd: float = 0.0

    # Efficiency
    gpu_compute_efficiency_pct: float = 0.0
    gpu_memory_bw_efficiency_pct: float = 0.0

    def __post_init__(self) -> None:
        self._compute()

    def _compute(self) -> None:
        self.cost_per_hour_usd = self.gpu_instance.cost_per_hour(self.num_gpus)

        # Cost per 1 M tokens generated
        if self.metrics.throughput_tps > 0:
            tokens_per_hour = self.metrics.throughput_tps * 3600.0
            self.cost_per_million_tokens_usd = (
                self.cost_per_hour_usd / tokens_per_hour * 1_000_000.0
            )

        # Cost per 1 M requests (using goodput)
        rps = self.metrics.goodput_rps or self.metrics.end_to_end_throughput_rps
        if rps > 0:
            requests_per_hour = rps * 3600.0
            self.cost_per_million_requests_usd = (
                self.cost_per_hour_usd / requests_per_hour * 1_000_000.0
            )

        # GPU compute efficiency: actual TPS vs theoretical peak
        # Assume ~2 FLOPs per token per parameter (rough estimate for matmuls)
        # and model_params is not directly available here, so we use a proxy:
        # efficiency = actual_tps / (peak_tflops * 1e12 / (2 * model_size_in_params))
        # We skip the model-size dependency and report a relative metric instead.
        # Approximation: token throughput efficiency vs GPU memory BW limit.
        if self.gpu_instance.memory_bandwidth_gbs > 0 and self.metrics.throughput_tps > 0:
            # KV cache and weight movement dominates memory BW in decode.
            # Rough upper bound: each token requires ~2 bytes per parameter read.
            # We use throughput_tps / (peak_memory_bw / bytes_per_token) as proxy.
            # bytes_per_token ≈ 2 * model_size_gb * 1024 (bfloat16 params per GPU)
            # Simplified: cap at 100%.
            bw_bound_tps = self.gpu_instance.memory_bandwidth_gbs * 1e9 / (2 * 1e6)
            self.gpu_memory_bw_efficiency_pct = min(
                100.0,
                self.metrics.throughput_tps / (bw_bound_tps * self.num_gpus) * 100.0,
            )

    def summary(self) -> str:
        return (
            f"GPU: {self.gpu_instance.gpu_model} × {self.num_gpus}  "
            f"| ${self.cost_per_hour_usd:.2f}/hr  "
            f"| ${self.cost_per_million_tokens_usd:.4f}/M tokens  "
            f"| ${self.cost_per_million_requests_usd:.4f}/M req"
        )


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------

class CostEstimator:
    """Estimates GPU cloud costs for a set of benchmark results.

    Args:
        gpu_instance_name: Key in GPU_CATALOGUE to use for pricing.
            Defaults to A100 SXM4 80GB.
    """

    def __init__(
        self,
        gpu_instance_name: str = DEFAULT_GPU_INSTANCE,
    ) -> None:
        if gpu_instance_name not in GPU_CATALOGUE:
            raise ValueError(
                f"Unknown GPU instance {gpu_instance_name!r}. "
                f"Available: {sorted(GPU_CATALOGUE)}"
            )
        self.gpu_instance = GPU_CATALOGUE[gpu_instance_name]

    def estimate(
        self,
        cfg: ConfigPoint,
        metrics: BenchmarkMetrics,
    ) -> CostEstimate:
        """Compute cost estimate for a single (config, metrics) pair.

        Args:
            cfg: ConfigPoint describing the parallelism setup.
            metrics: Collected BenchmarkMetrics for this run.

        Returns:
            CostEstimate with cost and efficiency fields populated.
        """
        if cfg.disaggregation_mode in ("distserve", "seesaw_resharding"):
            num_gpus = cfg.disaggregated_gpu_count()
        else:
            num_gpus = cfg.gpu_count()
        num_gpus = max(num_gpus, 1)

        return CostEstimate(
            config=cfg,
            metrics=metrics,
            gpu_instance=self.gpu_instance,
            num_gpus=num_gpus,
        )

    def estimate_batch(
        self,
        results: List[Tuple[ConfigPoint, BenchmarkMetrics]],
    ) -> List[CostEstimate]:
        """Compute cost estimates for a list of (config, metrics) pairs.

        Args:
            results: List of (ConfigPoint, BenchmarkMetrics) tuples.

        Returns:
            List of CostEstimate instances, one per result.
        """
        return [self.estimate(cfg, metrics) for cfg, metrics in results]

    def cheapest(
        self,
        estimates: List[CostEstimate],
    ) -> Optional[CostEstimate]:
        """Return the estimate with the lowest cost per million tokens.

        Args:
            estimates: List of CostEstimate instances.

        Returns:
            Cheapest CostEstimate, or None if the list is empty.
        """
        valid = [e for e in estimates if e.cost_per_million_tokens_usd > 0]
        if not valid:
            return None
        return min(valid, key=lambda e: e.cost_per_million_tokens_usd)

    def most_efficient(
        self,
        estimates: List[CostEstimate],
    ) -> Optional[CostEstimate]:
        """Return the estimate with the highest GPU memory BW efficiency.

        Args:
            estimates: List of CostEstimate instances.

        Returns:
            Most efficient CostEstimate, or None if the list is empty.
        """
        if not estimates:
            return None
        return max(estimates, key=lambda e: e.gpu_memory_bw_efficiency_pct)
