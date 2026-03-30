"""Configuration space explorer with sensitivity analysis.

Inspired by:
* BOHB: Robust and Efficient Hyperparameter Optimization at Scale (Falkner et al. 2018)
* Optuna: A Next-generation Hyperparameter Optimization Framework
* fANOVA: Functional ANOVA for Hyperparameter Importance
* "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"

Provides parameter sensitivity analysis, interaction effect
detection, feature importance ranking, and suggests promising
configurations based on observed performance patterns.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benchmark.config.schema import BenchmarkMetrics
from benchmark.config.sweep import ConfigPoint


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class ParameterSensitivity:
    """Sensitivity of a performance metric to a config parameter."""

    param_name: str
    metric_name: str
    importance_score: float = 0.0  # 0-1, higher = more important
    correlation: float = 0.0  # Pearson correlation
    best_value: Any = None
    worst_value: Any = None
    value_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class InteractionEffect:
    """Interaction between two parameters."""

    param_a: str
    param_b: str
    interaction_strength: float = 0.0  # 0-1
    best_combination: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ConfigSuggestion:
    """A suggested configuration to try."""

    config: Dict[str, Any]
    predicted_throughput: float = 0.0
    predicted_latency: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ExplorationProfile:
    """Full config space exploration results."""

    total_configs_evaluated: int = 0
    total_configs_in_space: int = 0
    coverage_pct: float = 0.0
    top_sensitivities: List[ParameterSensitivity] = field(default_factory=list)
    interactions: List[InteractionEffect] = field(default_factory=list)
    suggestions: List[ConfigSuggestion] = field(default_factory=list)
    param_rankings: Dict[str, float] = field(default_factory=dict)
    best_config: Optional[Dict[str, Any]] = None
    worst_config: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Config Space Exploration",
            f"  Configs evaluated  : {self.total_configs_evaluated}",
            f"  Space coverage     : {self.coverage_pct:.1f}%",
        ]
        if self.param_rankings:
            lines.append("  Parameter importance (for throughput):")
            for param, score in sorted(self.param_rankings.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    {param}: {score:.3f}")
        if self.interactions:
            lines.append("  Significant interactions:")
            for ia in self.interactions[:5]:
                lines.append(f"    {ia.param_a} × {ia.param_b}: strength={ia.interaction_strength:.3f}")
        if self.suggestions:
            lines.append("  Suggested configs to try:")
            for s in self.suggestions[:3]:
                lines.append(f"    [{s.confidence:.0%}] {s.reasoning}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config extraction helpers
# ---------------------------------------------------------------------------


_NUMERIC_PARAMS = [
    "tp", "pp", "dp", "max_batched_tokens", "max_num_seqs", "gpu_mem_util",
    "block_size", "chunk_size", "pd_ratio", "request_rate_rps", "num_requests",
]

_CATEGORICAL_PARAMS = [
    "dtype", "quantization", "attention_backend", "kv_dtype",
    "dataset", "arrival_process", "disaggregation_mode",
    "batching_scheme", "flash_attn_version",
]


def _extract_param(cfg: ConfigPoint, name: str) -> Any:
    """Extract a parameter value from a ConfigPoint."""
    return getattr(cfg, name, None)


def _extract_metric(metrics: BenchmarkMetrics, name: str) -> float:
    """Extract a metric value from BenchmarkMetrics."""
    val = getattr(metrics, name, None)
    if val is None:
        return 0.0
    return float(val)


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


class ConfigSpaceExplorer:
    """Analyses the configuration space and identifies important parameters.

    Usage::

        explorer = ConfigSpaceExplorer()
        for cfg, metrics in results:
            explorer.add_observation(cfg, metrics)
        profile = explorer.analyse(target_metric="throughput_tps")
        print(profile.summary())
    """

    def __init__(self, total_space_size: int = 0) -> None:
        self._observations: List[Tuple[ConfigPoint, BenchmarkMetrics]] = []
        self._total_space = total_space_size

    def add_observation(self, cfg: ConfigPoint, metrics: BenchmarkMetrics) -> None:
        self._observations.append((cfg, metrics))

    def add_observations(
        self, results: List[Tuple[ConfigPoint, BenchmarkMetrics]]
    ) -> None:
        self._observations.extend(results)

    def _compute_sensitivity(
        self, param_name: str, target_metric: str
    ) -> Optional[ParameterSensitivity]:
        """Compute sensitivity of target metric to a parameter."""
        # Group results by param value
        groups: Dict[str, List[float]] = {}
        for cfg, metrics in self._observations:
            val = _extract_param(cfg, param_name)
            if val is None:
                continue
            key = str(val)
            metric_val = _extract_metric(metrics, target_metric)
            groups.setdefault(key, []).append(metric_val)

        if len(groups) < 2:
            return None

        # Importance: variance between groups / total variance
        all_vals = [v for vals in groups.values() for v in vals]
        if not all_vals or len(all_vals) < 2:
            return None

        total_var = statistics.variance(all_vals)
        if total_var == 0:
            return None

        group_means = {k: statistics.mean(v) for k, v in groups.items()}
        grand_mean = statistics.mean(all_vals)
        between_var = sum(
            len(v) * (statistics.mean(v) - grand_mean) ** 2
            for v in groups.values()
        ) / len(all_vals)

        importance = between_var / total_var

        # Best/worst value
        best_val = max(group_means.items(), key=lambda x: x[1])[0]
        worst_val = min(group_means.items(), key=lambda x: x[1])[0]

        # Correlation (numeric params only)
        corr = 0.0
        if param_name in _NUMERIC_PARAMS:
            param_vals = []
            metric_vals = []
            for cfg, metrics in self._observations:
                pv = _extract_param(cfg, param_name)
                mv = _extract_metric(metrics, target_metric)
                if pv is not None:
                    try:
                        param_vals.append(float(pv))
                        metric_vals.append(mv)
                    except (TypeError, ValueError):
                        pass
            if len(param_vals) > 2:
                corr = _pearson_corr(param_vals, metric_vals)

        return ParameterSensitivity(
            param_name=param_name,
            metric_name=target_metric,
            importance_score=min(1.0, importance),
            correlation=corr,
            best_value=best_val,
            worst_value=worst_val,
            value_performance=group_means,
        )

    def _detect_interactions(
        self, target_metric: str, top_params: List[str]
    ) -> List[InteractionEffect]:
        """Detect interaction effects between parameter pairs."""
        interactions: List[InteractionEffect] = []

        for i in range(len(top_params)):
            for j in range(i + 1, len(top_params)):
                pa, pb = top_params[i], top_params[j]

                # Group by (pa, pb) combination
                combo_groups: Dict[str, List[float]] = {}
                for cfg, metrics in self._observations:
                    va = _extract_param(cfg, pa)
                    vb = _extract_param(cfg, pb)
                    if va is None or vb is None:
                        continue
                    key = f"{va}:{vb}"
                    metric_val = _extract_metric(metrics, target_metric)
                    combo_groups.setdefault(key, []).append(metric_val)

                if len(combo_groups) < 3:
                    continue

                # Simple interaction detection: variance of combo means beyond additive
                all_vals = [v for vals in combo_groups.values() for v in vals]
                if not all_vals or len(all_vals) < 3:
                    continue

                total_var = statistics.variance(all_vals) if len(all_vals) > 1 else 0
                if total_var == 0:
                    continue

                combo_means = {k: statistics.mean(v) for k, v in combo_groups.items()}
                grand_mean = statistics.mean(all_vals)
                combo_var = sum(
                    len(v) * (statistics.mean(v) - grand_mean) ** 2
                    for v in combo_groups.values()
                ) / len(all_vals)

                strength = min(1.0, combo_var / total_var)

                if strength > 0.1:
                    best_combo = max(combo_means.items(), key=lambda x: x[1])
                    parts = best_combo[0].split(":")
                    interactions.append(
                        InteractionEffect(
                            param_a=pa,
                            param_b=pb,
                            interaction_strength=strength,
                            best_combination={pa: parts[0], pb: parts[1] if len(parts) > 1 else ""},
                            description=f"{pa}×{pb} interaction explains {strength:.0%} of variance",
                        )
                    )

        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
        return interactions

    def _generate_suggestions(
        self,
        sensitivities: List[ParameterSensitivity],
        target_metric: str,
    ) -> List[ConfigSuggestion]:
        """Generate configuration suggestions based on observed patterns."""
        if not sensitivities:
            return []

        suggestions: List[ConfigSuggestion] = []

        # Suggestion 1: Use best value for each sensitive parameter
        best_config: Dict[str, Any] = {}
        for s in sensitivities[:6]:
            if s.best_value is not None:
                best_config[s.param_name] = s.best_value

        if best_config:
            suggestions.append(
                ConfigSuggestion(
                    config=best_config,
                    confidence=0.8,
                    reasoning="Combines best-performing values for most important parameters.",
                )
            )

        # Suggestion 2: Explore underrepresented regions
        for s in sensitivities[:3]:
            if len(s.value_performance) > 2:
                # Find least-tested value
                all_counts: Dict[str, int] = {}
                for cfg, _ in self._observations:
                    v = _extract_param(cfg, s.param_name)
                    if v is not None:
                        all_counts[str(v)] = all_counts.get(str(v), 0) + 1

                if all_counts:
                    least = min(all_counts.items(), key=lambda x: x[1])
                    suggestions.append(
                        ConfigSuggestion(
                            config={s.param_name: least[0]},
                            confidence=0.5,
                            reasoning=f"Explore undersampled {s.param_name}={least[0]} (only {least[1]} samples).",
                        )
                    )

        return suggestions[:5]

    def analyse(
        self, target_metric: str = "throughput_tps"
    ) -> ExplorationProfile:
        """Run full config space exploration analysis."""
        if not self._observations:
            return ExplorationProfile()

        # Compute sensitivities
        all_params = _NUMERIC_PARAMS + _CATEGORICAL_PARAMS
        sensitivities: List[ParameterSensitivity] = []
        for param in all_params:
            s = self._compute_sensitivity(param, target_metric)
            if s:
                sensitivities.append(s)

        sensitivities.sort(key=lambda s: s.importance_score, reverse=True)

        # Rankings
        rankings = {s.param_name: s.importance_score for s in sensitivities}

        # Interaction effects
        top_params = [s.param_name for s in sensitivities[:6]]
        interactions = self._detect_interactions(target_metric, top_params)

        # Suggestions
        suggestions = self._generate_suggestions(sensitivities, target_metric)

        # Best/worst config
        best_obs = max(self._observations, key=lambda x: _extract_metric(x[1], target_metric))
        worst_obs = min(self._observations, key=lambda x: _extract_metric(x[1], target_metric))
        best_cfg = {p: str(_extract_param(best_obs[0], p)) for p in all_params if _extract_param(best_obs[0], p) is not None}
        worst_cfg = {p: str(_extract_param(worst_obs[0], p)) for p in all_params if _extract_param(worst_obs[0], p) is not None}

        coverage = 100.0 * len(self._observations) / self._total_space if self._total_space > 0 else 0.0

        recs = self._generate_recommendations(sensitivities, interactions, coverage)

        return ExplorationProfile(
            total_configs_evaluated=len(self._observations),
            total_configs_in_space=self._total_space,
            coverage_pct=coverage,
            top_sensitivities=sensitivities[:10],
            interactions=interactions[:5],
            suggestions=suggestions,
            param_rankings=rankings,
            best_config=best_cfg,
            worst_config=worst_cfg,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        sensitivities: List[ParameterSensitivity],
        interactions: List[InteractionEffect],
        coverage: float,
    ) -> List[str]:
        recs: List[str] = []

        if coverage < 5.0:
            recs.append(
                f"Only {coverage:.1f}% of the config space explored. "
                "Consider increasing the sweep range for more comprehensive analysis."
            )

        if sensitivities:
            top = sensitivities[0]
            recs.append(
                f"Most impactful parameter: {top.param_name} "
                f"(importance={top.importance_score:.3f}). Best value: {top.best_value}."
            )

        if interactions:
            top_ia = interactions[0]
            recs.append(
                f"Strongest interaction: {top_ia.param_a} × {top_ia.param_b} "
                f"(strength={top_ia.interaction_strength:.3f}). "
                "These parameters should be tuned together."
            )

        # Check for low-importance params
        low = [s for s in sensitivities if s.importance_score < 0.01]
        if low:
            names = [s.param_name for s in low[:3]]
            recs.append(
                f"Low-impact parameters: {', '.join(names)}. "
                "These can be fixed to reduce the search space."
            )

        return recs

    def reset(self) -> None:
        self._observations.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson_corr(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)
