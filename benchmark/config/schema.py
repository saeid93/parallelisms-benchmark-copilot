"""
Pydantic models for benchmark run config and metrics (Section 2 schema).
"""

from __future__ import annotations

import uuid
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    # model
    model_id: str = Field(
        "meta-llama/Llama-2-13b-hf",
        description="HuggingFace model identifier to benchmark",
    )

    # parallelism
    tp: int = Field(1, description="Tensor parallel size")
    pp: int = Field(1, description="Pipeline parallel size")
    dp: int = Field(1, description="Data parallel replicas")
    prefill_tp: Optional[int] = None
    prefill_pp: Optional[int] = None
    decode_tp: Optional[int] = None
    decode_pp: Optional[int] = None
    disaggregation_mode: Literal["none", "distserve", "seesaw_resharding"] = "none"
    resharding_pair: Optional[str] = None
    expert_parallel: bool = False
    ep_all2all_backend: Literal[
        "deepep_high_throughput",
        "deepep_low_latency",
        "flashinfer_nvlink_one_sided",
        "naive",
        "auto",
    ] = "auto"
    distributed_backend: Literal["mp", "ray", "external_launcher"] = "mp"

    # batching
    max_batched_tokens: int = 2048
    max_num_seqs: int = 64
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_threshold: int = 512
    chunked_prefill: bool = True
    chunk_size: int = 256
    batching_scheme: Literal[
        "decode_maximal",
        "prefill_only",
        "decode_only",
        "orca_iteration_level",
    ] = "decode_maximal"
    pd_ratio: Optional[float] = None

    # kv cache
    kv_dtype: Literal[
        "auto", "fp16", "fp8", "fp8_e4m3", "fp8_e5m2", "int8"
    ] = "auto"
    calculate_kv_scales: bool = False
    gpu_mem_util: float = 0.90
    kv_cache_memory_bytes: Optional[int] = None
    block_size: int = 16
    prefix_caching: bool = False
    swap_space_gb: int = 4
    cpu_offload_gb: int = 0
    cpu_kv_buffer_gb: Optional[float] = None
    kv_cache_layout: Literal["NHD", "HND"] = "HND"
    transition_policy: Optional[
        Literal[
            "prefill_prioritizing",
            "decode_prioritizing",
            "transition_minimizing",
        ]
    ] = None
    kv_offloading_backend: Literal["native", "lmcache"] = "native"
    max_model_len: int = 8192

    # compute
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    quantization: Literal[
        "none",
        "awq",
        "gptq",
        "fp8",
        "gptq_marlin",
        "awq_marlin",
        "bitsandbytes",
        "compressed_tensors",
    ] = "none"
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    attention_backend: Literal["flash_attn", "flashinfer", "triton", "auto"] = (
        "flash_attn"
    )
    flash_attn_version: int = 3
    disable_sliding_window: bool = False

    # speculative decoding
    speculative: bool = False
    num_speculative_tokens: Optional[int] = None
    draft_model: Optional[str] = None
    acceptance_rate: Optional[float] = None

    # workload
    dataset: Literal[
        "sharegpt",
        "humaneval",
        "longbench",
        "arxiv_summarization",
        "synthetic_uniform",
        "synthetic_zipf",
    ] = "sharegpt"
    arrival_process: Literal["offline", "poisson"] = "poisson"
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


# ---------------------------------------------------------------------------
# Per-op timing breakdown
# ---------------------------------------------------------------------------

class PerOpBreakdownMs(BaseModel):
    preproj: float = 0.0
    attn: float = 0.0
    postproj: float = 0.0
    ffn: float = 0.0


# ---------------------------------------------------------------------------
# Metrics model
# ---------------------------------------------------------------------------

class BenchmarkMetrics(BaseModel):
    # throughput
    throughput_tps: float = 0.0
    end_to_end_throughput_rps: float = 0.0
    goodput_rps: float = 0.0

    # latency breakdown
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p90_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    e2e_latency_p50_ms: float = 0.0
    e2e_latency_p90_ms: float = 0.0
    e2e_latency_p99_ms: float = 0.0

    # SLO attainment
    ttft_slo_attainment_pct: float = 0.0
    tpot_slo_attainment_pct: float = 0.0
    joint_slo_attainment_pct: float = 0.0

    # phase timing
    prefill_exec_time_ms: float = 0.0
    prefill_queuing_time_ms: float = 0.0
    decode_exec_time_ms: float = 0.0
    decode_queuing_time_ms: float = 0.0
    kv_transmission_time_ms: float = 0.0

    # SARATHI-specific
    decode_speedup_vs_baseline: float = 0.0
    decode_time_per_token_ms: float = 0.0
    pipeline_bubble_ratio: float = 0.0
    bubble_time_per_request_ms: float = 0.0
    per_op_breakdown_ms: PerOpBreakdownMs = Field(
        default_factory=PerOpBreakdownMs
    )

    # Seesaw-specific
    resharding_overhead_ms: float = 0.0
    num_prefill_decode_transitions: int = 0
    avg_decode_batch_size: float = 0.0
    cpu_kv_utilization_pct: float = 0.0

    # memory
    gpu_mem_used_gb: float = 0.0
    kv_cache_hit_rate: float = 0.0
    preemption_rate: float = 0.0
    bubble_ratio: float = 0.0

    # parallelism efficiency
    prefill_phase_time_pct: float = 0.0
    decode_phase_time_pct: float = 0.0
    transmission_time_pct: float = 0.0
    allreduce_overhead_pct: float = 0.0
    weight_transfer_time_pct: float = 0.0

    # speculative decoding
    speculative_acceptance_rate: Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level run record
# ---------------------------------------------------------------------------

BenchmarkSuite = Literal[
    "vllm_parallelism", "distserve", "sarathi", "seesaw"
]


class BenchmarkRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_suite: BenchmarkSuite = "vllm_parallelism"
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
