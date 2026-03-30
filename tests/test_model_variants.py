"""Tests for model variant support across config, sweep, and validation."""

import pytest

from benchmark.config.schema import BenchmarkConfig
from benchmark.config.sweep import (
    MODEL_VARIANTS,
    ModelVariant,
    ConfigPoint,
    generate_distserve_sweep,
    generate_full_sweep,
    generate_parallelism_sweep,
    generate_sarathi_sweep,
    generate_seesaw_sweep,
    get_model_variant,
)
from benchmark.config.validation import ConfigValidator, Severity


# ---------------------------------------------------------------------------
# ModelVariant registry
# ---------------------------------------------------------------------------

class TestModelVariantRegistry:
    def test_known_variant_lookup(self):
        variant = get_model_variant("llama-2-7b")
        assert variant.model_id == "meta-llama/Llama-2-7b-hf"
        assert variant.params_gb == 14.0

    def test_all_registry_entries_have_model_id(self):
        for name, variant in MODEL_VARIANTS.items():
            assert variant.model_id, f"{name} missing model_id"
            assert variant.params_gb > 0, f"{name} missing params_gb"
            assert "/" in variant.model_id, f"{name} model_id should be org/name"

    def test_unknown_variant_raises(self):
        with pytest.raises(KeyError, match="Unknown model variant"):
            get_model_variant("nonexistent-model")

    def test_registry_has_expected_variants(self):
        expected = {
            "llama-2-7b", "llama-2-13b", "llama-2-70b",
            "llama-3-8b", "llama-3-70b",
            "mistral-7b", "mixtral-8x7b",
            "qwen-2-7b", "qwen-2-72b",
        }
        assert expected == set(MODEL_VARIANTS.keys())

    def test_model_variant_is_frozen(self):
        variant = get_model_variant("llama-2-7b")
        with pytest.raises(AttributeError):
            variant.model_id = "changed"


# ---------------------------------------------------------------------------
# model_id on ConfigPoint and BenchmarkConfig
# ---------------------------------------------------------------------------

class TestModelIdField:
    def test_config_point_default_model_id(self):
        cfg = ConfigPoint()
        assert cfg.model_id == "meta-llama/Llama-2-13b-hf"

    def test_config_point_custom_model_id(self):
        cfg = ConfigPoint(model_id="mistralai/Mistral-7B-v0.1")
        assert cfg.model_id == "mistralai/Mistral-7B-v0.1"

    def test_benchmark_config_default_model_id(self):
        cfg = BenchmarkConfig()
        assert cfg.model_id == "meta-llama/Llama-2-13b-hf"

    def test_benchmark_config_custom_model_id(self):
        cfg = BenchmarkConfig(model_id="Qwen/Qwen2-7B")
        assert cfg.model_id == "Qwen/Qwen2-7B"


# ---------------------------------------------------------------------------
# Sweep generation with model_variants
# ---------------------------------------------------------------------------

class TestSweepWithModelVariants:
    def test_parallelism_sweep_single_variant(self):
        configs = list(generate_parallelism_sweep(
            tp_sizes=[1, 2],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-2-7b"],
        ))
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.model_id == "meta-llama/Llama-2-7b-hf"

    def test_parallelism_sweep_multiple_variants(self):
        configs = list(generate_parallelism_sweep(
            tp_sizes=[1],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-2-7b", "mistral-7b"],
        ))
        model_ids = {cfg.model_id for cfg in configs}
        assert "meta-llama/Llama-2-7b-hf" in model_ids
        assert "mistralai/Mistral-7B-v0.1" in model_ids

    def test_distserve_sweep_with_variant(self):
        configs = list(generate_distserve_sweep(
            prefill_tp_sizes=[1],
            prefill_pp_sizes=[1],
            decode_tp_sizes=[1],
            decode_pp_sizes=[1],
            ttft_slos=[250],
            tpot_slos=[100],
            slo_scales=[1.0],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["mistral-7b"],
        ))
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.model_id == "mistralai/Mistral-7B-v0.1"

    def test_sarathi_sweep_with_variant(self):
        configs = list(generate_sarathi_sweep(
            chunk_sizes=[256],
            batching_schemes=["decode_maximal"],
            pd_ratios=[10],
            tp_sizes=[1],
            pp_sizes=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["qwen-2-7b"],
        ))
        assert len(configs) == 1
        assert configs[0].model_id == "Qwen/Qwen2-7B"

    def test_seesaw_sweep_with_variant(self):
        configs = list(generate_seesaw_sweep(
            cpu_kv_buffer_gb_values=[40],
            kv_cache_layouts=["HND"],
            transition_policies=["prefill_prioritizing"],
            tp_sizes=[1],
            pp_sizes=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-3-8b"],
        ))
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.model_id == "meta-llama/Meta-Llama-3-8B"

    def test_full_sweep_with_model_variants(self):
        configs = generate_full_sweep(
            max_gpus=4,
            suites=["vllm_parallelism"],
            model_variants=["llama-2-7b", "mistral-7b"],
        )
        model_ids = {cfg.model_id for cfg in configs}
        assert len(model_ids) == 2

    def test_large_model_pruned_on_small_cluster(self):
        # llama-2-70b is 140GB, needs at least TP=2 on 80GB GPUs
        configs = list(generate_parallelism_sweep(
            tp_sizes=[1],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-2-70b"],
        ))
        # TP=1 with 140GB model → OOM on single 80GB GPU → pruned
        assert len(configs) == 0

    def test_large_model_feasible_with_enough_tp(self):
        configs = list(generate_parallelism_sweep(
            tp_sizes=[4],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-2-70b"],
        ))
        # 140GB / 4 GPUs = 35GB per GPU, fits in 80*0.9=72GB
        assert len(configs) > 0

    def test_sweep_without_model_variants_uses_model_params_gb(self):
        # Without model_variants, should use model_params_gb as before
        configs = list(generate_parallelism_sweep(
            tp_sizes=[1],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_params_gb=7.0,
        ))
        assert len(configs) > 0

    def test_model_variant_sets_max_model_len(self):
        configs = list(generate_parallelism_sweep(
            tp_sizes=[1],
            pp_sizes=[1],
            dp_replicas=[1],
            datasets=["sharegpt"],
            max_gpus=8,
            model_variants=["llama-2-7b"],
        ))
        assert len(configs) > 0
        # llama-2-7b default_max_model_len is 4096
        assert configs[0].max_model_len == 4096


# ---------------------------------------------------------------------------
# Validation of model_id
# ---------------------------------------------------------------------------

class TestModelVariantValidation:
    def test_valid_model_id(self):
        cfg = ConfigPoint(model_id="meta-llama/Llama-2-7b-hf")
        validator = ConfigValidator()
        result = validator.validate(cfg)
        model_issues = [i for i in result.issues if "model_id" in i.rule]
        assert len(model_issues) == 0

    def test_empty_model_id_error(self):
        cfg = ConfigPoint(model_id="")
        validator = ConfigValidator()
        result = validator.validate(cfg)
        errors = [i for i in result.errors if i.rule == "model_id_required"]
        assert len(errors) == 1

    def test_short_name_model_id_warning(self):
        cfg = ConfigPoint(model_id="my-custom-model")
        validator = ConfigValidator()
        result = validator.validate(cfg)
        warnings = [i for i in result.warnings if i.rule == "model_id_format"]
        assert len(warnings) == 1
        assert "does not contain '/'" in warnings[0].message

    def test_strict_mode_promotes_model_id_warning(self):
        cfg = ConfigPoint(model_id="my-custom-model")
        validator = ConfigValidator(strict=True)
        result = validator.validate(cfg)
        errors = [i for i in result.errors if i.rule == "model_id_format"]
        assert len(errors) == 1
