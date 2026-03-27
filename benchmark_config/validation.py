"""Configuration validation logic.

``ConfigValidator`` checks that every field in a ``BenchmarkRunConfig``
falls within its declared valid-value set, and that cross-field
constraints (e.g. TP × PP ≤ available GPUs) are satisfied.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Sequence

from benchmark_config.schema import (
    AttentionBackendConfig,
    BatchingConfig,
    BenchmarkRunConfig,
    ChunkedPrefillConfig,
    ComputePrecisionConfig,
    DisaggregatedConfig,
    DynamicReshardingConfig,
    KVCacheConfig,
    ParallelismConfig,
    SLOTargetConfig,
    WorkloadConfig,
)


class ConfigValidationError(Exception):
    """Raised when one or more config fields are invalid."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"{len(errors)} validation error(s):\n" + "\n".join(errors)
        )


# Mapping from (sub-config class, field name) → VALID_* attribute name
_FIELD_VALID_MAP: dict[type, dict[str, str]] = {
    ParallelismConfig: {
        "tensor_parallel_size": "VALID_TP",
        "pipeline_parallel_size": "VALID_PP",
        "data_parallel_replicas": "VALID_DP",
        "ep_all2all_backend": "VALID_EP_BACKENDS",
        "distributed_executor_backend": "VALID_EXEC_BACKENDS",
    },
    DisaggregatedConfig: {
        "prefill_tp": "VALID_PREFILL_TP",
        "prefill_pp": "VALID_PREFILL_PP",
        "decode_tp": "VALID_DECODE_TP",
        "decode_pp": "VALID_DECODE_PP",
        "disaggregation_mode": "VALID_MODES",
    },
    DynamicReshardingConfig: {
        "cpu_kv_buffer_gb": "VALID_CPU_KV_BUFFER_GB",
        "kv_cache_layout": "VALID_KV_LAYOUTS",
        "transition_policy": "VALID_TRANSITION_POLICIES",
    },
    BatchingConfig: {
        "max_num_batched_tokens": "VALID_MAX_BATCHED_TOKENS",
        "max_num_seqs": "VALID_MAX_SEQS",
        "max_num_partial_prefills": "VALID_PARTIAL_PREFILLS",
        "max_long_partial_prefills": "VALID_LONG_PARTIAL",
        "long_prefill_token_threshold": "VALID_LONG_THRESHOLD",
    },
    ChunkedPrefillConfig: {
        "chunk_size": "VALID_CHUNK_SIZES",
        "batching_scheme": "VALID_BATCHING_SCHEMES",
        "pd_ratio": "VALID_PD_RATIOS",
    },
    KVCacheConfig: {
        "kv_cache_dtype": "VALID_KV_DTYPES",
        "gpu_memory_utilization": "VALID_GPU_MEM_UTIL",
        "block_size": "VALID_BLOCK_SIZES",
        "swap_space_gb": "VALID_SWAP_SPACE",
        "cpu_offload_gb": "VALID_CPU_OFFLOAD",
        "kv_offloading_backend": "VALID_OFFLOAD_BACKENDS",
        "max_model_len": "VALID_MAX_MODEL_LEN",
    },
    ComputePrecisionConfig: {
        "dtype": "VALID_DTYPES",
        "quantization": "VALID_QUANTIZATIONS",
        "max_seq_len_to_capture": "VALID_SEQ_LEN_CAPTURE",
    },
    AttentionBackendConfig: {
        "attention_backend": "VALID_BACKENDS",
        "flash_attn_version": "VALID_FLASH_VERSIONS",
    },
    WorkloadConfig: {
        "workload_dataset": "VALID_DATASETS",
        "arrival_process": "VALID_ARRIVAL",
        "request_rate_rps": "VALID_RPS",
        "num_requests": "VALID_NUM_REQUESTS",
        "input_length_tokens": "VALID_INPUT_LEN",
        "output_length_tokens": "VALID_OUTPUT_LEN",
    },
    SLOTargetConfig: {
        "ttft_slo_ms": "VALID_TTFT",
        "tpot_slo_ms": "VALID_TPOT",
        "slo_attainment_target_pct": "VALID_ATTAINMENT",
        "slo_scale_sweep": "VALID_SCALE",
    },
}

# Mapping from sub-config class → attribute name on BenchmarkRunConfig
_SUB_CONFIG_ATTR: dict[type, str] = {
    ParallelismConfig: "parallelism",
    DisaggregatedConfig: "disaggregated",
    DynamicReshardingConfig: "dynamic_resharding",
    BatchingConfig: "batching",
    ChunkedPrefillConfig: "chunked_prefill",
    KVCacheConfig: "kv_cache",
    ComputePrecisionConfig: "compute",
    AttentionBackendConfig: "attention",
    WorkloadConfig: "workload",
    SLOTargetConfig: "slo_target",
}


class ConfigValidator:
    """Validates a :class:`BenchmarkRunConfig` against allowed values.

    Parameters
    ----------
    max_gpus:
        Maximum number of GPUs available.  Used for cross-field checks
        (TP × PP ≤ max_gpus, etc.).
    """

    def __init__(self, max_gpus: int = 8) -> None:
        self.max_gpus = max_gpus

    # ── public API ────────────────────────────────────────────────────

    def validate(self, cfg: BenchmarkRunConfig) -> list[str]:
        """Return a list of human-readable error strings (empty = valid)."""
        errors: list[str] = []
        self._check_suite(cfg, errors)
        self._check_field_values(cfg, errors)
        self._check_cross_field(cfg, errors)
        return errors

    def validate_or_raise(self, cfg: BenchmarkRunConfig) -> None:
        """Raise :class:`ConfigValidationError` if invalid."""
        errors = self.validate(cfg)
        if errors:
            raise ConfigValidationError(errors)

    # ── internal helpers ──────────────────────────────────────────────

    def _check_suite(
        self, cfg: BenchmarkRunConfig, errors: list[str],
    ) -> None:
        if cfg.benchmark_suite not in BenchmarkRunConfig.VALID_SUITES:
            errors.append(
                f"benchmark_suite={cfg.benchmark_suite!r} not in "
                f"{BenchmarkRunConfig.VALID_SUITES}"
            )

    def _check_field_values(
        self, cfg: BenchmarkRunConfig, errors: list[str],
    ) -> None:
        for cls, field_map in _FIELD_VALID_MAP.items():
            attr_name = _SUB_CONFIG_ATTR[cls]
            sub = getattr(cfg, attr_name)
            for fname, valid_attr in field_map.items():
                value = getattr(sub, fname)
                allowed: Sequence[Any] = getattr(cls, valid_attr)
                if value not in allowed:
                    errors.append(
                        f"{attr_name}.{fname}={value!r} not in {allowed}"
                    )

    def _check_cross_field(
        self, cfg: BenchmarkRunConfig, errors: list[str],
    ) -> None:
        # TP × PP must not exceed available GPUs
        tp_pp = (
            cfg.parallelism.tensor_parallel_size
            * cfg.parallelism.pipeline_parallel_size
        )
        if tp_pp > self.max_gpus:
            errors.append(
                f"TP({cfg.parallelism.tensor_parallel_size}) × "
                f"PP({cfg.parallelism.pipeline_parallel_size}) = {tp_pp} "
                f"exceeds max_gpus={self.max_gpus}"
            )

        # Total GPUs including data-parallel replicas
        total = tp_pp * cfg.parallelism.data_parallel_replicas
        if total > self.max_gpus:
            errors.append(
                f"TP×PP×DP = {total} exceeds max_gpus={self.max_gpus}"
            )

        # Disaggregated placement check
        if cfg.benchmark_suite == "distserve":
            prefill_gpus = cfg.disaggregated.prefill_tp * cfg.disaggregated.prefill_pp
            decode_gpus = cfg.disaggregated.decode_tp * cfg.disaggregated.decode_pp
            if prefill_gpus + decode_gpus > self.max_gpus:
                errors.append(
                    f"Disaggregated prefill GPUs ({prefill_gpus}) + "
                    f"decode GPUs ({decode_gpus}) = "
                    f"{prefill_gpus + decode_gpus} exceeds "
                    f"max_gpus={self.max_gpus}"
                )

        # Chunked prefill: chunk_size ≤ max_num_batched_tokens
        if cfg.benchmark_suite == "sarathi":
            if cfg.chunked_prefill.chunk_size > cfg.batching.max_num_batched_tokens:
                errors.append(
                    f"chunk_size ({cfg.chunked_prefill.chunk_size}) > "
                    f"max_num_batched_tokens "
                    f"({cfg.batching.max_num_batched_tokens})"
                )

        # max_model_len ≥ input + output lengths
        needed = (
            cfg.workload.input_length_tokens
            + cfg.workload.output_length_tokens
        )
        if cfg.kv_cache.max_model_len < needed:
            errors.append(
                f"max_model_len ({cfg.kv_cache.max_model_len}) < "
                f"input_length ({cfg.workload.input_length_tokens}) + "
                f"output_length ({cfg.workload.output_length_tokens}) "
                f"= {needed}"
            )
