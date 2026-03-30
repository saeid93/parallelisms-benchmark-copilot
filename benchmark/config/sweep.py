"""
Stage 1 — Config space generator.

Emits the full sweep matrix from Section 1 of the problem statement and
prunes infeasible combinations before they are dispatched to benchmark
runners.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelVariant:
    """Metadata for a single model variant used in sweep generation.

    Attributes:
        model_id: HuggingFace model identifier (e.g. "meta-llama/Llama-2-13b-hf").
        params_gb: Approximate model parameter memory footprint in GiB.
        default_max_model_len: Default max sequence length for this model.
    """

    model_id: str
    params_gb: float
    default_max_model_len: int = 8192


# Curated registry of common model variants.
MODEL_VARIANTS: Dict[str, ModelVariant] = {
    "llama-2-7b": ModelVariant(
        model_id="meta-llama/Llama-2-7b-hf",
        params_gb=14.0,
        default_max_model_len=4096,
    ),
    "llama-2-13b": ModelVariant(
        model_id="meta-llama/Llama-2-13b-hf",
        params_gb=26.0,
        default_max_model_len=4096,
    ),
    "llama-2-70b": ModelVariant(
        model_id="meta-llama/Llama-2-70b-hf",
        params_gb=140.0,
        default_max_model_len=4096,
    ),
    "llama-3-8b": ModelVariant(
        model_id="meta-llama/Meta-Llama-3-8B",
        params_gb=16.0,
        default_max_model_len=8192,
    ),
    "llama-3-70b": ModelVariant(
        model_id="meta-llama/Meta-Llama-3-70B",
        params_gb=140.0,
        default_max_model_len=8192,
    ),
    "mistral-7b": ModelVariant(
        model_id="mistralai/Mistral-7B-v0.1",
        params_gb=14.0,
        default_max_model_len=8192,
    ),
    "mixtral-8x7b": ModelVariant(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        params_gb=93.0,
        default_max_model_len=32768,
    ),
    "qwen-2-7b": ModelVariant(
        model_id="Qwen/Qwen2-7B",
        params_gb=14.0,
        default_max_model_len=32768,
    ),
    "qwen-2-72b": ModelVariant(
        model_id="Qwen/Qwen2-72B",
        params_gb=144.0,
        default_max_model_len=32768,
    ),
}


def get_model_variant(name: str) -> ModelVariant:
    """Look up a model variant by short name.

    Args:
        name: Short name key in MODEL_VARIANTS (e.g. "llama-2-7b").

    Returns:
        The corresponding ModelVariant.

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in MODEL_VARIANTS:
        raise KeyError(
            f"Unknown model variant {name!r}. "
            f"Available: {sorted(MODEL_VARIANTS)}"
        )
    return MODEL_VARIANTS[name]


# Fallback variant used when no explicit model_variants list is supplied.
_DEFAULT_MODEL_ID = "meta-llama/Llama-2-13b-hf"


# ---------------------------------------------------------------------------
# Sweep dimension definitions (Section 1)
# ---------------------------------------------------------------------------

# 1A — Parallelism
TP_SIZES = [1, 2, 4, 8]
PP_SIZES = [1, 2, 4, 8]
DP_REPLICAS = [1, 2, 4]
DISAGGREGATION_MODES = ["none", "distserve", "seesaw_resharding"]
DISTRIBUTED_BACKENDS = ["mp", "ray", "external_launcher"]
EP_ALL2ALL_BACKENDS = [
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_nvlink_one_sided",
    "naive",
]
CPU_KV_BUFFER_GB_VALUES = [20, 40, 80, 160]
KV_CACHE_LAYOUTS = ["NHD", "HND"]
TRANSITION_POLICIES = [
    "prefill_prioritizing",
    "decode_prioritizing",
    "transition_minimizing",
]

# 1B — Batching
MAX_BATCHED_TOKENS_VALUES = [512, 1024, 2048, 4096, 8192]
MAX_NUM_SEQS_VALUES = [16, 32, 64, 128, 256]
MAX_NUM_PARTIAL_PREFILLS_VALUES = [1, 2, 4]
MAX_LONG_PARTIAL_PREFILLS_VALUES = [1, 2]
LONG_PREFILL_THRESHOLD_VALUES = [256, 512, 1024]
CHUNK_SIZES = [64, 128, 192, 256, 320, 384, 448, 512]
BATCHING_SCHEMES = [
    "decode_maximal",
    "prefill_only",
    "decode_only",
    "orca_iteration_level",
]
PD_RATIOS = [1, 5, 10, 14, 20, 28, 42, 50, 84, 100, 128, 200]

# 1C — KV Cache
KV_CACHE_DTYPES = ["auto", "fp16", "fp8", "fp8_e4m3", "fp8_e5m2", "int8"]
GPU_MEM_UTILS = [0.80, 0.85, 0.90, 0.95]
BLOCK_SIZES = [8, 16, 32, 64, 128]
SWAP_SPACE_GB_VALUES = [0, 4, 8, 16]
CPU_OFFLOAD_GB_VALUES = [0, 4, 8, 16]
KV_OFFLOADING_BACKENDS = ["native", "lmcache"]
MAX_MODEL_LEN_VALUES = [2048, 4096, 8192, 16384]

# 1D — Compute & Precision
DTYPES = ["float16", "bfloat16", "float32"]
QUANTIZATIONS = [
    "none",
    "awq",
    "gptq",
    "fp8",
    "gptq_marlin",
    "awq_marlin",
    "bitsandbytes",
    "compressed_tensors",
]
MAX_SEQ_LEN_TO_CAPTURE_VALUES = [2048, 4096, 8192]

# 1E — Attention Backend
ATTENTION_BACKENDS = ["flash_attn", "flashinfer", "triton", "auto"]
FLASH_ATTN_VERSIONS = [2, 3, 4]

# 1F — Workload
WORKLOAD_DATASETS = [
    "sharegpt",
    "humaneval",
    "longbench",
    "arxiv_summarization",
    "synthetic_uniform",
    "synthetic_zipf",
]
ARRIVAL_PROCESSES = ["offline", "poisson"]
REQUEST_RATE_RPS_VALUES = [0.5, 1.0, 1.6, 2.0, 3.0, 4.0, 5.6, 7.0, 10.0]
NUM_REQUESTS_VALUES = [500, 1000, 2000, 10000]
INPUT_LENGTH_TOKENS_VALUES = [128, 256, 512, 1024, 2048, 3000, 4096]
OUTPUT_LENGTH_TOKENS_VALUES = [64, 128, 256, 512, 1024]

# 1F — SLO (DistServe)
TTFT_SLO_MS_VALUES = [125, 250, 2500, 4000, 15000]
TPOT_SLO_MS_VALUES = [100, 150, 200]
SLO_ATTAINMENT_TARGET_PCT_VALUES = [90, 99]
SLO_SCALE_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Config point dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfigPoint:
    """A single point in the sweep configuration space."""

    # model
    model_id: str = "meta-llama/Llama-2-13b-hf"

    # parallelism
    tp: int = 1
    pp: int = 1
    dp: int = 1
    prefill_tp: Optional[int] = None
    prefill_pp: Optional[int] = None
    decode_tp: Optional[int] = None
    decode_pp: Optional[int] = None
    disaggregation_mode: str = "none"
    resharding_pair: Optional[str] = None
    expert_parallel: bool = False
    ep_all2all_backend: str = "auto"
    distributed_backend: str = "mp"

    # batching
    max_batched_tokens: int = 2048
    max_num_seqs: int = 64
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_threshold: int = 512
    chunked_prefill: bool = True
    chunk_size: int = 256
    batching_scheme: str = "decode_maximal"
    pd_ratio: Optional[float] = None

    # kv cache
    kv_dtype: str = "auto"
    calculate_kv_scales: bool = False
    gpu_mem_util: float = 0.90
    kv_cache_memory_bytes: Optional[int] = None
    block_size: int = 16
    prefix_caching: bool = False
    swap_space_gb: int = 4
    cpu_offload_gb: int = 0
    cpu_kv_buffer_gb: Optional[float] = None
    kv_cache_layout: str = "HND"
    transition_policy: Optional[str] = None
    kv_offloading_backend: str = "native"
    max_model_len: int = 8192

    # compute
    dtype: str = "bfloat16"
    quantization: str = "none"
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    attention_backend: str = "flash_attn"
    flash_attn_version: int = 3
    disable_sliding_window: bool = False

    # speculative decoding
    speculative: bool = False
    num_speculative_tokens: Optional[int] = None
    draft_model: Optional[str] = None
    acceptance_rate: Optional[float] = None

    # workload
    dataset: str = "sharegpt"
    arrival_process: str = "poisson"
    request_rate_rps: float = 2.0
    num_requests: int = 1000
    avg_input_tokens: int = 755
    avg_output_tokens: int = 200
    pd_ratio_actual: Optional[float] = None

    # SLO
    ttft_slo_ms: float = 250.0
    tpot_slo_ms: float = 100.0
    slo_attainment_target_pct: float = 90.0
    slo_scale: float = 1.0

    # metadata
    benchmark_suite: str = "vllm_parallelism"

    def gpu_count(self) -> int:
        """Total GPUs consumed by this config (non-disaggregated)."""
        return self.tp * self.pp * self.dp

    def disaggregated_gpu_count(self) -> int:
        """Total GPUs for a disaggregated prefill+decode config."""
        prefill_gpus = (self.prefill_tp or 1) * (self.prefill_pp or 1)
        decode_gpus = (self.decode_tp or 1) * (self.decode_pp or 1)
        return (prefill_gpus + decode_gpus) * self.dp


# ---------------------------------------------------------------------------
# Feasibility pruning (Section 4)
# ---------------------------------------------------------------------------

def _is_feasible(cfg: ConfigPoint, max_gpus: int, model_params_gb: float) -> bool:
    """Return True when the config is feasible given hardware constraints.

    Args:
        cfg: Configuration point to evaluate.
        max_gpus: Total GPUs available in the cluster.
        model_params_gb: Approximate model parameter memory in GiB (used for
            a coarse OOM check).

    Returns:
        True if the config is feasible, False otherwise.
    """
    # --- GPU count check ---
    if cfg.disaggregation_mode in ("distserve", "seesaw_resharding"):
        if cfg.disaggregated_gpu_count() > max_gpus:
            return False
    else:
        if cfg.gpu_count() > max_gpus:
            return False

    # --- chunk_size vs max_model_len ---
    if cfg.chunked_prefill and cfg.chunk_size > cfg.max_model_len:
        return False

    # --- max_seq_len_to_capture vs max_model_len ---
    if cfg.max_seq_len_to_capture > cfg.max_model_len:
        return False

    # --- Coarse OOM check: model must fit across TP shards ---
    gpus_for_model = cfg.tp * cfg.pp
    if gpus_for_model == 0:
        return False
    mem_per_gpu_gb = model_params_gb / gpus_for_model
    # Assume 80 GiB per GPU (A100/H100); leave gpu_mem_util headroom
    if mem_per_gpu_gb > 80.0 * cfg.gpu_mem_util:
        return False

    # --- pd_ratio outside realistic range (Section 1, Stage 1 note) ---
    if cfg.pd_ratio is not None and not (1 <= cfg.pd_ratio <= 200):
        return False

    return True


# ---------------------------------------------------------------------------
# Sweep generators
# ---------------------------------------------------------------------------

def generate_parallelism_sweep(
    tp_sizes: List[int] = TP_SIZES,
    pp_sizes: List[int] = PP_SIZES,
    dp_replicas: List[int] = DP_REPLICAS,
    datasets: List[str] = WORKLOAD_DATASETS,
    max_gpus: int = 8,
    model_params_gb: float = 14.0,
    model_variants: Optional[List[str]] = None,
) -> Iterator[ConfigPoint]:
    """Yield feasible configs for the general vllm_parallelism suite.

    Args:
        model_variants: Optional list of short model variant names from the
            MODEL_VARIANTS registry.  When provided the sweep iterates over
            each variant, using its ``params_gb`` for the OOM check and
            setting ``model_id`` on every emitted ConfigPoint.
    """
    if model_variants:
        variants = [get_model_variant(n) for n in model_variants]
    else:
        variants = [ModelVariant(model_id=_DEFAULT_MODEL_ID, params_gb=model_params_gb)]

    for variant in variants:
        for tp, pp, dp, dataset in itertools.product(
            tp_sizes, pp_sizes, dp_replicas, datasets
        ):
            cfg = ConfigPoint(
                model_id=variant.model_id,
                tp=tp,
                pp=pp,
                dp=dp,
                dataset=dataset,
                max_model_len=variant.default_max_model_len,
                max_seq_len_to_capture=min(8192, variant.default_max_model_len),
                benchmark_suite="vllm_parallelism",
            )
            if _is_feasible(cfg, max_gpus, variant.params_gb):
                yield cfg


def generate_distserve_sweep(
    prefill_tp_sizes: List[int] = TP_SIZES,
    prefill_pp_sizes: List[int] = [1, 2, 4],
    decode_tp_sizes: List[int] = TP_SIZES,
    decode_pp_sizes: List[int] = [1, 2, 4],
    ttft_slos: List[float] = TTFT_SLO_MS_VALUES,
    tpot_slos: List[float] = TPOT_SLO_MS_VALUES,
    slo_scales: List[float] = SLO_SCALE_SWEEP,
    datasets: List[str] = WORKLOAD_DATASETS,
    max_gpus: int = 8,
    model_params_gb: float = 14.0,
    model_variants: Optional[List[str]] = None,
) -> Iterator[ConfigPoint]:
    """Yield feasible configs for the DistServe disaggregated suite."""
    if model_variants:
        variants = [get_model_variant(n) for n in model_variants]
    else:
        variants = [ModelVariant(model_id=_DEFAULT_MODEL_ID, params_gb=model_params_gb)]

    for variant in variants:
        for (
            prefill_tp, prefill_pp, decode_tp, decode_pp,
            ttft_slo, tpot_slo, slo_scale, dataset
        ) in itertools.product(
            prefill_tp_sizes, prefill_pp_sizes,
            decode_tp_sizes, decode_pp_sizes,
            ttft_slos, tpot_slos, slo_scales, datasets,
        ):
            cfg = ConfigPoint(
                model_id=variant.model_id,
                tp=max(prefill_tp, decode_tp),
                pp=max(prefill_pp, decode_pp),
                prefill_tp=prefill_tp,
                prefill_pp=prefill_pp,
                decode_tp=decode_tp,
                decode_pp=decode_pp,
                disaggregation_mode="distserve",
                ttft_slo_ms=ttft_slo,
                tpot_slo_ms=tpot_slo,
                slo_scale=slo_scale,
                dataset=dataset,
                max_model_len=variant.default_max_model_len,
                max_seq_len_to_capture=min(8192, variant.default_max_model_len),
                benchmark_suite="distserve",
            )
            if _is_feasible(cfg, max_gpus, variant.params_gb):
                yield cfg


def generate_sarathi_sweep(
    chunk_sizes: List[int] = CHUNK_SIZES,
    batching_schemes: List[str] = BATCHING_SCHEMES,
    pd_ratios: List[int] = PD_RATIOS,
    tp_sizes: List[int] = TP_SIZES,
    pp_sizes: List[int] = PP_SIZES,
    datasets: List[str] = WORKLOAD_DATASETS,
    max_gpus: int = 8,
    model_params_gb: float = 14.0,
    model_variants: Optional[List[str]] = None,
) -> Iterator[ConfigPoint]:
    """Yield feasible configs for the SARATHI chunked-prefill suite."""
    if model_variants:
        variants = [get_model_variant(n) for n in model_variants]
    else:
        variants = [ModelVariant(model_id=_DEFAULT_MODEL_ID, params_gb=model_params_gb)]

    for variant in variants:
        for tp, pp, chunk_size, batching_scheme, pd_ratio, dataset in itertools.product(
            tp_sizes, pp_sizes, chunk_sizes, batching_schemes, pd_ratios, datasets
        ):
            cfg = ConfigPoint(
                model_id=variant.model_id,
                tp=tp,
                pp=pp,
                chunked_prefill=True,
                chunk_size=chunk_size,
                batching_scheme=batching_scheme,
                pd_ratio=float(pd_ratio),
                dataset=dataset,
                max_model_len=variant.default_max_model_len,
                max_seq_len_to_capture=min(8192, variant.default_max_model_len),
                benchmark_suite="sarathi",
            )
            if _is_feasible(cfg, max_gpus, variant.params_gb):
                yield cfg


def generate_seesaw_sweep(
    cpu_kv_buffer_gb_values: List[float] = CPU_KV_BUFFER_GB_VALUES,
    kv_cache_layouts: List[str] = KV_CACHE_LAYOUTS,
    transition_policies: List[str] = TRANSITION_POLICIES,
    tp_sizes: List[int] = TP_SIZES,
    pp_sizes: List[int] = PP_SIZES,
    datasets: List[str] = WORKLOAD_DATASETS,
    max_gpus: int = 8,
    model_params_gb: float = 14.0,
    model_variants: Optional[List[str]] = None,
) -> Iterator[ConfigPoint]:
    """Yield feasible configs for the Seesaw dynamic re-sharding suite."""
    if model_variants:
        variants = [get_model_variant(n) for n in model_variants]
    else:
        variants = [ModelVariant(model_id=_DEFAULT_MODEL_ID, params_gb=model_params_gb)]

    for variant in variants:
        for (
            tp, pp, cpu_kv_buffer_gb, kv_cache_layout, transition_policy, dataset
        ) in itertools.product(
            tp_sizes, pp_sizes,
            cpu_kv_buffer_gb_values, kv_cache_layouts, transition_policies, datasets,
        ):
            resharding_pair = f"PP{pp}->TP{tp}"
            cfg = ConfigPoint(
                model_id=variant.model_id,
                tp=tp,
                pp=pp,
                disaggregation_mode="seesaw_resharding",
                resharding_pair=resharding_pair,
                cpu_kv_buffer_gb=cpu_kv_buffer_gb,
                kv_cache_layout=kv_cache_layout,
                transition_policy=transition_policy,
                dataset=dataset,
                max_model_len=variant.default_max_model_len,
                max_seq_len_to_capture=min(8192, variant.default_max_model_len),
                benchmark_suite="seesaw",
            )
            if _is_feasible(cfg, max_gpus, variant.params_gb):
                yield cfg


def generate_full_sweep(
    max_gpus: int = 8,
    model_params_gb: float = 14.0,
    suites: Optional[List[str]] = None,
    model_variants: Optional[List[str]] = None,
) -> List[ConfigPoint]:
    """Generate the full config sweep across all benchmark suites.

    Args:
        max_gpus: Total GPUs available for hardware feasibility pruning.
        model_params_gb: Approximate model parameter footprint in GiB.
            Ignored when *model_variants* is provided (each variant
            carries its own ``params_gb``).
        suites: List of suite names to include. Defaults to all four suites.
        model_variants: Optional list of short model variant names from the
            MODEL_VARIANTS registry.  When provided the sweep iterates
            over each variant, using its ``params_gb`` for OOM checks and
            setting ``model_id`` on every emitted ConfigPoint.

    Returns:
        A list of feasible ConfigPoint instances.
    """
    enabled = set(suites or ["vllm_parallelism", "distserve", "sarathi", "seesaw"])
    configs: List[ConfigPoint] = []

    sweep_kwargs = dict(max_gpus=max_gpus, model_params_gb=model_params_gb, model_variants=model_variants)

    if "vllm_parallelism" in enabled:
        configs.extend(generate_parallelism_sweep(**sweep_kwargs))
    if "distserve" in enabled:
        configs.extend(generate_distserve_sweep(**sweep_kwargs))
    if "sarathi" in enabled:
        configs.extend(generate_sarathi_sweep(**sweep_kwargs))
    if "seesaw" in enabled:
        configs.extend(generate_seesaw_sweep(**sweep_kwargs))

    return configs
