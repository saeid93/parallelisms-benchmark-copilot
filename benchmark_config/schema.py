"""Configuration schema definitions for all sweep dimensions.

Each dataclass maps to one subsection of the config sweep specification
(Sections 1A–1F). All fields use the exact parameter names from vLLM's
CLI / engine config so that a config record can be passed directly to
the benchmark harness.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


# ── 1A. Parallelism ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParallelismConfig:
    """Unified parallelism knobs (tensor, pipeline, data, expert)."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_replicas: int = 1
    expert_parallel: bool = False
    ep_all2all_backend: str = "naive"
    distributed_executor_backend: str = "mp"

    VALID_TP: tuple[int, ...] = (1, 2, 4, 8)
    VALID_PP: tuple[int, ...] = (1, 2, 4, 8)
    VALID_DP: tuple[int, ...] = (1, 2, 4)
    VALID_EP_BACKENDS: tuple[str, ...] = (
        "deepep_high_throughput",
        "deepep_low_latency",
        "flashinfer_nvlink_one_sided",
        "naive",
    )
    VALID_EXEC_BACKENDS: tuple[str, ...] = ("mp", "ray", "external_launcher")


@dataclass(frozen=True)
class DisaggregatedConfig:
    """DistServe-style disaggregated prefill/decode placement."""

    prefill_tp: int = 1
    prefill_pp: int = 1
    decode_tp: int = 1
    decode_pp: int = 1
    disaggregation_mode: str = "none"

    VALID_PREFILL_TP: tuple[int, ...] = (1, 2, 4, 8)
    VALID_PREFILL_PP: tuple[int, ...] = (1, 2, 4)
    VALID_DECODE_TP: tuple[int, ...] = (1, 2, 4, 8)
    VALID_DECODE_PP: tuple[int, ...] = (1, 2, 4)
    VALID_MODES: tuple[str, ...] = ("none", "distserve", "seesaw_resharding")


@dataclass(frozen=True)
class DynamicReshardingConfig:
    """Seesaw-style dynamic re-sharding between prefill and decode."""

    prefill_parallelism: str = "PP4"
    decode_parallelism: str = "TP4"
    cpu_kv_buffer_gb: int = 20
    kv_cache_layout: str = "NHD"
    transition_policy: str = "prefill_prioritizing"

    VALID_CPU_KV_BUFFER_GB: tuple[int, ...] = (20, 40, 80, 160)
    VALID_KV_LAYOUTS: tuple[str, ...] = ("NHD", "HND")
    VALID_TRANSITION_POLICIES: tuple[str, ...] = (
        "prefill_prioritizing",
        "decode_prioritizing",
        "transition_minimizing",
    )


# ── 1B. Batching & Scheduling ─────────────────────────────────────────────


@dataclass(frozen=True)
class BatchingConfig:
    """Core batching and scheduling knobs."""

    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 128
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 512
    enable_chunked_prefill: bool = True

    VALID_MAX_BATCHED_TOKENS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    VALID_MAX_SEQS: tuple[int, ...] = (16, 32, 64, 128, 256)
    VALID_PARTIAL_PREFILLS: tuple[int, ...] = (1, 2, 4)
    VALID_LONG_PARTIAL: tuple[int, ...] = (1, 2)
    VALID_LONG_THRESHOLD: tuple[int, ...] = (256, 512, 1024)


@dataclass(frozen=True)
class ChunkedPrefillConfig:
    """SARATHI-style chunked prefill sweep parameters."""

    chunk_size: int = 256
    batching_scheme: str = "decode_maximal"
    pd_ratio: int = 10

    VALID_CHUNK_SIZES: tuple[int, ...] = (
        64, 128, 192, 256, 320, 384, 448, 512,
    )
    VALID_BATCHING_SCHEMES: tuple[str, ...] = (
        "decode_maximal",
        "prefill_only",
        "decode_only",
        "orca_iteration_level",
    )
    VALID_PD_RATIOS: tuple[int, ...] = (
        1, 5, 10, 14, 20, 28, 42, 50, 84, 100, 128, 200,
    )


# ── 1C. KV Cache ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KVCacheConfig:
    """KV cache sizing, dtype, and offloading knobs."""

    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    enable_prefix_caching: bool = False
    calculate_kv_scales: bool = False
    swap_space_gb: int = 0
    cpu_offload_gb: int = 0
    kv_offloading_backend: str = "native"
    max_model_len: int = 4096

    VALID_KV_DTYPES: tuple[str, ...] = (
        "auto", "fp16", "fp8", "fp8_e4m3", "fp8_e5m2", "int8",
    )
    VALID_GPU_MEM_UTIL: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
    VALID_BLOCK_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128)
    VALID_OFFLOAD_BACKENDS: tuple[str, ...] = ("native", "lmcache")
    VALID_SWAP_SPACE: tuple[int, ...] = (0, 4, 8, 16)
    VALID_CPU_OFFLOAD: tuple[int, ...] = (0, 4, 8, 16)
    VALID_MAX_MODEL_LEN: tuple[int, ...] = (2048, 4096, 8192, 16384)


# ── 1D. Compute & Precision ───────────────────────────────────────────────


@dataclass(frozen=True)
class ComputePrecisionConfig:
    """Dtype, quantization, and CUDA-graph capture settings."""

    dtype: str = "bfloat16"
    quantization: str = "none"
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192

    VALID_DTYPES: tuple[str, ...] = ("float16", "bfloat16", "float32")
    VALID_QUANTIZATIONS: tuple[str, ...] = (
        "none", "awq", "gptq", "fp8",
        "gptq_marlin", "awq_marlin",
        "bitsandbytes", "compressed_tensors",
    )
    VALID_SEQ_LEN_CAPTURE: tuple[int, ...] = (2048, 4096, 8192)


# ── 1E. Attention Backend ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AttentionBackendConfig:
    """Attention kernel selection."""

    attention_backend: str = "auto"
    flash_attn_version: int = 2
    disable_sliding_window: bool = False

    VALID_BACKENDS: tuple[str, ...] = (
        "flash_attn", "flashinfer", "triton", "auto",
    )
    VALID_FLASH_VERSIONS: tuple[int, ...] = (2, 3, 4)


# ── 1F. Workload ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkloadConfig:
    """Workload profile shared across all benchmark suites."""

    workload_dataset: str = "sharegpt"
    arrival_process: str = "offline"
    request_rate_rps: float = 1.0
    num_requests: int = 1000
    input_length_tokens: int = 512
    output_length_tokens: int = 128

    VALID_DATASETS: tuple[str, ...] = (
        "sharegpt", "humaneval", "longbench",
        "arxiv_summarization", "synthetic_uniform", "synthetic_zipf",
    )
    VALID_ARRIVAL: tuple[str, ...] = ("offline", "poisson")
    VALID_RPS: tuple[float, ...] = (
        0.5, 1.0, 1.6, 2.0, 3.0, 4.0, 5.6, 7.0, 10.0,
    )
    VALID_NUM_REQUESTS: tuple[int, ...] = (500, 1000, 2000, 10000)
    VALID_INPUT_LEN: tuple[int, ...] = (128, 256, 512, 1024, 2048, 3000, 4096)
    VALID_OUTPUT_LEN: tuple[int, ...] = (64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class SLOTargetConfig:
    """DistServe SLO targets for goodput evaluation."""

    ttft_slo_ms: int = 250
    tpot_slo_ms: int = 150
    slo_attainment_target_pct: int = 99
    slo_scale_sweep: float = 1.0

    VALID_TTFT: tuple[int, ...] = (125, 250, 2500, 4000, 15000)
    VALID_TPOT: tuple[int, ...] = (100, 150, 200)
    VALID_ATTAINMENT: tuple[int, ...] = (90, 99)
    VALID_SCALE: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


# ── Top-level composite config ────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Complete configuration for a single benchmark run.

    Aggregates all sub-configs.  The ``benchmark_suite`` field selects
    which sub-configs are active:
      - ``"unified"``   → uses ``parallelism``
      - ``"distserve"`` → uses ``disaggregated`` + ``slo_target``
      - ``"sarathi"``   → uses ``chunked_prefill``
      - ``"seesaw"``    → uses ``dynamic_resharding``
    """

    benchmark_suite: str = "unified"
    model: str = "meta-llama/Llama-2-7b-hf"

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    disaggregated: DisaggregatedConfig = field(
        default_factory=DisaggregatedConfig,
    )
    dynamic_resharding: DynamicReshardingConfig = field(
        default_factory=DynamicReshardingConfig,
    )
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    chunked_prefill: ChunkedPrefillConfig = field(
        default_factory=ChunkedPrefillConfig,
    )
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    compute: ComputePrecisionConfig = field(
        default_factory=ComputePrecisionConfig,
    )
    attention: AttentionBackendConfig = field(
        default_factory=AttentionBackendConfig,
    )
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    slo_target: SLOTargetConfig = field(default_factory=SLOTargetConfig)

    VALID_SUITES: tuple[str, ...] = (
        "unified", "distserve", "sarathi", "seesaw",
    )

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for JSON export."""
        return dataclasses.asdict(self)
