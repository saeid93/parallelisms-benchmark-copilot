"""
Config compatibility validator.

Checks each BenchmarkConfig or ConfigPoint for logical inconsistencies
and hardware-incompatible combinations that cannot be detected by Pydantic
field-level validation alone.

Validation rules cover:
  - Speculative decoding: requires draft_model when speculative=True
  - Expert parallelism: requires disaggregation_mode != 'none'
  - Disaggregated configs: require prefill_tp/prefill_pp/decode_tp/decode_pp
  - Seesaw resharding: requires cpu_kv_buffer_gb and transition_policy
  - Quantization × dtype compatibility
  - KV dtype × attention backend compatibility
  - chunked_prefill requires chunk_size > 0 and chunk_size <= max_batched_tokens
  - swap_space vs cpu_offload mutual exclusivity
  - Flash attention v3 requires bfloat16
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from benchmark.config.schema import BenchmarkConfig
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    ERROR = "error"      # Config is invalid; will fail at runtime
    WARNING = "warning"  # Config may work but produces unexpected results
    INFO = "info"        # Informational note


# ---------------------------------------------------------------------------
# Validation issue
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single validation issue found in a config."""

    severity: Severity
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] ({self.rule}) {self.message}"


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Collection of issues found during config validation."""

    issues: List[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        """True when there are no ERROR-severity issues."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        if not self.issues:
            return "Config OK — no issues found."
        lines = [f"Config validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------

def _check_speculative(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if getattr(cfg, "speculative", False):
        if not getattr(cfg, "draft_model", None):
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                rule="speculative_needs_draft_model",
                message="speculative=True requires draft_model to be set.",
            ))
        num_spec = getattr(cfg, "num_speculative_tokens", None)
        if num_spec is None or num_spec <= 0:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                rule="speculative_num_tokens",
                message=(
                    "speculative=True without a positive num_speculative_tokens; "
                    "defaulting to 5 may be sub-optimal for your workload."
                ),
            ))
    return issues


def _check_expert_parallel(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    ep = getattr(cfg, "expert_parallel", False)
    disagg = getattr(cfg, "disaggregation_mode", "none")
    if ep and disagg == "none":
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="ep_needs_disaggregation",
            message=(
                "expert_parallel=True typically requires disaggregation_mode='distserve' "
                "or 'seesaw_resharding' for MoE models; 'none' may yield suboptimal routing."
            ),
        ))
    backend = getattr(cfg, "ep_all2all_backend", "auto")
    if ep and backend == "naive":
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="ep_naive_backend",
            message=(
                "ep_all2all_backend='naive' has high latency; prefer "
                "'deepep_low_latency' for online serving."
            ),
        ))
    return issues


def _check_disaggregation(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    disagg = getattr(cfg, "disaggregation_mode", "none")

    if disagg in ("distserve", "seesaw_resharding"):
        missing = []
        for attr in ("prefill_tp", "prefill_pp", "decode_tp", "decode_pp"):
            if getattr(cfg, attr, None) is None:
                missing.append(attr)
        if missing:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                rule="disagg_missing_tp_pp",
                message=(
                    f"disaggregation_mode={disagg!r} requires {missing} to be set."
                ),
            ))

    if disagg == "seesaw_resharding":
        if getattr(cfg, "cpu_kv_buffer_gb", None) is None:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                rule="seesaw_needs_cpu_kv_buffer",
                message=(
                    "disaggregation_mode='seesaw_resharding' requires cpu_kv_buffer_gb."
                ),
            ))
        if getattr(cfg, "transition_policy", None) is None:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                rule="seesaw_needs_transition_policy",
                message=(
                    "disaggregation_mode='seesaw_resharding' without transition_policy; "
                    "defaulting to 'prefill_prioritizing'."
                ),
            ))

    return issues


def _check_quantization(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    quant = getattr(cfg, "quantization", "none")
    dtype = getattr(cfg, "dtype", "bfloat16")
    kv_dtype = getattr(cfg, "kv_dtype", "auto")
    attn = getattr(cfg, "attention_backend", "flash_attn")
    flash_ver = getattr(cfg, "flash_attn_version", 3)

    # fp8 quantization requires bfloat16 or float16 dtype
    if quant in ("fp8", "gptq_marlin", "awq_marlin") and dtype == "float32":
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            rule="quant_fp8_needs_fp16_bf16",
            message=(
                f"quantization={quant!r} is incompatible with dtype='float32'. "
                "Use 'bfloat16' or 'float16'."
            ),
        ))

    # FP8 KV cache with triton attention may not be supported
    if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2") and attn == "triton":
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="fp8_kv_triton_attn",
            message=(
                f"kv_dtype={kv_dtype!r} with attention_backend='triton' may not be "
                "supported in all vLLM versions; prefer 'flashinfer' for FP8 KV."
            ),
        ))

    # Flash attention v3 requires bfloat16
    if attn == "flash_attn" and flash_ver >= 3 and dtype != "bfloat16":
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="flash_attn_v3_bf16",
            message=(
                f"flash_attn_version=3 is optimised for bfloat16; "
                f"dtype={dtype!r} may fall back to v2 kernels."
            ),
        ))

    return issues


def _check_chunked_prefill(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not getattr(cfg, "chunked_prefill", False):
        return issues

    chunk = getattr(cfg, "chunk_size", 0)
    max_batched = getattr(cfg, "max_batched_tokens", 2048)
    max_model_len = getattr(cfg, "max_model_len", 8192)

    if chunk <= 0:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            rule="chunked_prefill_chunk_size_positive",
            message="chunked_prefill=True requires chunk_size > 0.",
        ))
    elif chunk > max_batched:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="chunk_size_exceeds_max_batched_tokens",
            message=(
                f"chunk_size={chunk} > max_batched_tokens={max_batched}; "
                "the chunk will be capped internally."
            ),
        ))

    if chunk > max_model_len:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            rule="chunk_size_exceeds_max_model_len",
            message=(
                f"chunk_size={chunk} > max_model_len={max_model_len}; "
                "this configuration is invalid."
            ),
        ))

    return issues


def _check_memory(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    swap_gb = getattr(cfg, "swap_space_gb", 0)
    offload_gb = getattr(cfg, "cpu_offload_gb", 0)

    if swap_gb > 0 and offload_gb > 0:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="swap_and_cpu_offload_combined",
            message=(
                f"swap_space_gb={swap_gb} and cpu_offload_gb={offload_gb} are both set; "
                "using both simultaneously may cause unexpected memory pressure."
            ),
        ))

    gpu_util = getattr(cfg, "gpu_mem_util", 0.90)
    if gpu_util > 0.98:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="gpu_mem_util_very_high",
            message=(
                f"gpu_mem_util={gpu_util} is very high; OOM risk increases "
                "significantly above 0.95."
            ),
        ))

    return issues


def _check_pipeline(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    pp = getattr(cfg, "pp", 1)
    enforce_eager = getattr(cfg, "enforce_eager", False)

    if pp > 1 and enforce_eager:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            rule="pp_with_eager",
            message=(
                f"pp={pp} with enforce_eager=True disables CUDA graph capturing; "
                "pipeline parallelism performs poorly without CUDA graphs."
            ),
        ))

    return issues


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """Validates BenchmarkConfig or ConfigPoint instances.

    Args:
        strict: If True, treat WARNING-level issues as errors for
            is_valid evaluation.
    """

    _RULES = [
        _check_speculative,
        _check_expert_parallel,
        _check_disaggregation,
        _check_quantization,
        _check_chunked_prefill,
        _check_memory,
        _check_pipeline,
    ]

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def validate(
        self,
        cfg: Union[BenchmarkConfig, ConfigPoint],
    ) -> ValidationResult:
        """Run all validation rules against a config.

        Args:
            cfg: BenchmarkConfig or ConfigPoint to validate.

        Returns:
            ValidationResult with all detected issues.
        """
        issues: List[ValidationIssue] = []
        for rule_fn in self._RULES:
            issues.extend(rule_fn(cfg))

        if self.strict:
            # Promote warnings to errors in strict mode
            promoted = []
            for issue in issues:
                if issue.severity == Severity.WARNING:
                    promoted.append(ValidationIssue(
                        severity=Severity.ERROR,
                        rule=issue.rule,
                        message=issue.message,
                    ))
                else:
                    promoted.append(issue)
            issues = promoted

        return ValidationResult(issues=issues)

    def validate_batch(
        self,
        configs: List[Union[BenchmarkConfig, ConfigPoint]],
    ) -> List[ValidationResult]:
        """Validate a list of configs.

        Args:
            configs: List of BenchmarkConfig or ConfigPoint instances.

        Returns:
            List of ValidationResult, one per input config.
        """
        return [self.validate(cfg) for cfg in configs]

    def filter_valid(
        self,
        configs: List[Union[BenchmarkConfig, ConfigPoint]],
    ) -> List[Union[BenchmarkConfig, ConfigPoint]]:
        """Return only the configs that pass validation.

        Args:
            configs: List of configs to filter.

        Returns:
            Subset of configs with no ERROR-level issues.
        """
        return [
            cfg
            for cfg, result in zip(configs, self.validate_batch(configs))
            if result.is_valid
        ]
