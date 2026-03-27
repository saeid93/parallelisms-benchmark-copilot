"""Sweep grid generation with cross-product enumeration.

``SweepGenerator`` produces :class:`BenchmarkRunConfig` instances by
taking the Cartesian product of selected dimensions and filtering out
infeasible combinations.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, Iterator, Optional, Sequence

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
from benchmark_config.validation import ConfigValidator


class SweepGenerator:
    """Generate benchmark configurations from sweep dimensions.

    Parameters
    ----------
    max_gpus:
        GPU budget; configs whose TP×PP (×DP) exceed this are filtered.
    base_config:
        Optional base config whose non-swept fields are preserved.
    validate:
        When *True* (the default) every emitted config passes
        :class:`ConfigValidator`.
    """

    def __init__(
        self,
        max_gpus: int = 8,
        base_config: Optional[BenchmarkRunConfig] = None,
        validate: bool = True,
    ) -> None:
        self.max_gpus = max_gpus
        self.base = base_config or BenchmarkRunConfig()
        self._validator = ConfigValidator(max_gpus=max_gpus) if validate else None

    # ── public helpers ────────────────────────────────────────────────

    def sweep_parallelism(
        self,
        tp_sizes: Sequence[int] = ParallelismConfig.VALID_TP,
        pp_sizes: Sequence[int] = ParallelismConfig.VALID_PP,
        dp_replicas: Sequence[int] = (1,),
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield configs for every feasible TP × PP × DP combination."""
        for tp, pp, dp in itertools.product(tp_sizes, pp_sizes, dp_replicas):
            if tp * pp * dp > self.max_gpus:
                continue
            cfg = replace(
                self.base,
                parallelism=replace(
                    self.base.parallelism,
                    tensor_parallel_size=tp,
                    pipeline_parallel_size=pp,
                    data_parallel_replicas=dp,
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def sweep_disaggregated(
        self,
        prefill_tp: Sequence[int] = DisaggregatedConfig.VALID_PREFILL_TP,
        prefill_pp: Sequence[int] = DisaggregatedConfig.VALID_PREFILL_PP,
        decode_tp: Sequence[int] = DisaggregatedConfig.VALID_DECODE_TP,
        decode_pp: Sequence[int] = DisaggregatedConfig.VALID_DECODE_PP,
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield DistServe disaggregated placement configs."""
        base = replace(self.base, benchmark_suite="distserve")
        for ptp, ppp, dtp, dpp in itertools.product(
            prefill_tp, prefill_pp, decode_tp, decode_pp,
        ):
            if ptp * ppp + dtp * dpp > self.max_gpus:
                continue
            cfg = replace(
                base,
                disaggregated=replace(
                    base.disaggregated,
                    prefill_tp=ptp,
                    prefill_pp=ppp,
                    decode_tp=dtp,
                    decode_pp=dpp,
                    disaggregation_mode="distserve",
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def sweep_chunked_prefill(
        self,
        chunk_sizes: Sequence[int] = ChunkedPrefillConfig.VALID_CHUNK_SIZES,
        batching_schemes: Sequence[str] = ChunkedPrefillConfig.VALID_BATCHING_SCHEMES,
        pd_ratios: Sequence[int] = ChunkedPrefillConfig.VALID_PD_RATIOS,
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield SARATHI-style chunked prefill sweep configs."""
        base = replace(self.base, benchmark_suite="sarathi")
        for cs, bs, pdr in itertools.product(
            chunk_sizes, batching_schemes, pd_ratios,
        ):
            cfg = replace(
                base,
                chunked_prefill=replace(
                    base.chunked_prefill,
                    chunk_size=cs,
                    batching_scheme=bs,
                    pd_ratio=pdr,
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def sweep_kv_cache(
        self,
        dtypes: Sequence[str] = KVCacheConfig.VALID_KV_DTYPES,
        block_sizes: Sequence[int] = KVCacheConfig.VALID_BLOCK_SIZES,
        gpu_mem_utils: Sequence[float] = KVCacheConfig.VALID_GPU_MEM_UTIL,
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield configs sweeping KV cache parameters."""
        for kdt, blk, gmu in itertools.product(
            dtypes, block_sizes, gpu_mem_utils,
        ):
            cfg = replace(
                self.base,
                kv_cache=replace(
                    self.base.kv_cache,
                    kv_cache_dtype=kdt,
                    block_size=blk,
                    gpu_memory_utilization=gmu,
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def sweep_workload(
        self,
        datasets: Sequence[str] = WorkloadConfig.VALID_DATASETS,
        rps_values: Sequence[float] = WorkloadConfig.VALID_RPS,
        input_lengths: Sequence[int] = WorkloadConfig.VALID_INPUT_LEN,
        output_lengths: Sequence[int] = WorkloadConfig.VALID_OUTPUT_LEN,
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield configs sweeping workload parameters."""
        for ds, rps, il, ol in itertools.product(
            datasets, rps_values, input_lengths, output_lengths,
        ):
            if il + ol > self.base.kv_cache.max_model_len:
                continue
            cfg = replace(
                self.base,
                workload=replace(
                    self.base.workload,
                    workload_dataset=ds,
                    request_rate_rps=rps,
                    input_length_tokens=il,
                    output_length_tokens=ol,
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def sweep_slo(
        self,
        ttft_values: Sequence[int] = SLOTargetConfig.VALID_TTFT,
        tpot_values: Sequence[int] = SLOTargetConfig.VALID_TPOT,
        scales: Sequence[float] = SLOTargetConfig.VALID_SCALE,
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield configs sweeping SLO targets (DistServe suite)."""
        base = replace(self.base, benchmark_suite="distserve")
        for ttft, tpot, scale in itertools.product(
            ttft_values, tpot_values, scales,
        ):
            cfg = replace(
                base,
                slo_target=replace(
                    base.slo_target,
                    ttft_slo_ms=ttft,
                    tpot_slo_ms=tpot,
                    slo_scale_sweep=scale,
                ),
            )
            if self._is_valid(cfg):
                yield cfg

    def custom_sweep(
        self,
        dimensions: dict[str, dict[str, Sequence[Any]]],
    ) -> Iterator[BenchmarkRunConfig]:
        """Yield configs from an arbitrary set of dimensions.

        ``dimensions`` maps sub-config attribute names (e.g.
        ``"parallelism"``) to dicts of ``{field_name: [values]}``.

        Example::

            gen.custom_sweep({
                "parallelism": {
                    "tensor_parallel_size": [1, 2, 4],
                },
                "kv_cache": {
                    "block_size": [16, 32],
                },
            })
        """
        # Build per-sub-config products, then cross them
        sub_products: list[list[tuple[str, dict[str, Any]]]] = []
        for sub_name, field_vals in dimensions.items():
            keys = list(field_vals.keys())
            value_lists = [field_vals[k] for k in keys]
            combos = [
                (sub_name, dict(zip(keys, vals)))
                for vals in itertools.product(*value_lists)
            ]
            sub_products.append(combos)

        # If any dimension has an empty value list the product is empty,
        # which is the correct behaviour (no configs to sweep).  When
        # *dimensions* itself is empty the product yields one empty tuple
        # and we emit a single base config – also intentional.
        if not sub_products:
            if self._is_valid(self.base):
                yield self.base
            return

        for combo in itertools.product(*sub_products):
            cfg = self.base
            for sub_name, overrides in combo:
                sub = getattr(cfg, sub_name)
                cfg = replace(cfg, **{sub_name: replace(sub, **overrides)})
            if self._is_valid(cfg):
                yield cfg

    # ── internals ─────────────────────────────────────────────────────

    def _is_valid(self, cfg: BenchmarkRunConfig) -> bool:
        if self._validator is None:
            return True
        return len(self._validator.validate(cfg)) == 0
