"""
Statistical significance tester — analysis module.

Provides hypothesis tests and confidence-interval estimates for comparing
two groups of per-request latency observations:

  - Mann-Whitney U test  (non-parametric, no normality assumption)
  - Bootstrap confidence intervals for mean and percentiles
  - Cohen's d effect size estimate

All tests are intentionally lightweight (no external scipy dependency for
the Mann-Whitney U test; a pure-Python implementation is provided and a
scipy fast-path is used when available).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Pure-Python Mann-Whitney U statistic
# ---------------------------------------------------------------------------

def _mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute the Mann-Whitney U statistic and approximate p-value.

    Uses a normal approximation which is accurate when both samples contain
    more than ~20 observations.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Tuple (U, p_value_two_sided).

    Raises:
        ValueError: If either sample is empty.
    """
    if not x or not y:
        raise ValueError("Both samples must be non-empty")

    n1, n2 = len(x), len(y)
    # Count concordant pairs
    u1 = sum(1 for xi in x for yj in y if xi > yj) + 0.5 * sum(
        1 for xi in x for yj in y if xi == yj
    )
    u2 = n1 * n2 - u1

    # Normal approximation
    mu_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_u == 0:
        p_value = 1.0
    else:
        z = (min(u1, u2) - mu_u) / sigma_u
        # Two-tailed p-value using standard normal CDF approximation (Abramowitz & Stegun)
        p_value = 2.0 * _norm_cdf(z)

    return min(u1, u2), p_value


def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF using erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: List[float],
    statistic_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        data: Observed sample.
        statistic_fn: Callable(List[float]) -> float, e.g. statistics.mean.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level in (0, 1), e.g. 0.95 for 95% CI.
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the confidence interval.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("data must be non-empty")

    rng = random.Random(seed)
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        resample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        stats.append(statistic_fn(resample))

    stats.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(math.floor(alpha / 2.0 * n_bootstrap))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1
    lo_idx = max(0, lo_idx)
    hi_idx = min(n_bootstrap - 1, hi_idx)
    return stats[lo_idx], stats[hi_idx]


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------

def cohens_d(x: List[float], y: List[float]) -> float:
    """Compute Cohen's d effect size between two samples.

    Args:
        x: First sample (e.g., TTFT latencies for config A).
        y: Second sample (e.g., TTFT latencies for config B).

    Returns:
        Cohen's d (signed; positive means x > y).
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    var_x = sum((v - mean_x) ** 2 for v in x) / (len(x) - 1)
    var_y = sum((v - mean_y) ** 2 for v in y) / (len(y) - 1)
    pooled_sd = math.sqrt((var_x + var_y) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return (mean_x - mean_y) / pooled_sd


# ---------------------------------------------------------------------------
# Significance test result
# ---------------------------------------------------------------------------

@dataclass
class SignificanceResult:
    """Result of a statistical significance test between two latency samples."""

    metric_name: str
    sample_a_size: int
    sample_b_size: int
    mean_a: float
    mean_b: float
    u_statistic: float
    p_value: float
    cohens_d: float
    ci_mean_a: Tuple[float, float]
    ci_mean_b: Tuple[float, float]
    alpha: float = 0.05

    @property
    def is_significant(self) -> bool:
        """True when p_value < alpha (reject null hypothesis of equal medians)."""
        return self.p_value < self.alpha

    @property
    def effect_size_label(self) -> str:
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.is_significant else "not significant"
        direction = "A > B" if self.mean_a > self.mean_b else "A < B"
        return (
            f"{self.metric_name}: {sig} (p={self.p_value:.4f}, α={self.alpha})  "
            f"mean_A={self.mean_a:.3f}  mean_B={self.mean_b:.3f}  "
            f"({direction})  d={self.cohens_d:.3f} [{self.effect_size_label}]  "
            f"CI_A=({self.ci_mean_a[0]:.3f}, {self.ci_mean_a[1]:.3f})  "
            f"CI_B=({self.ci_mean_b[0]:.3f}, {self.ci_mean_b[1]:.3f})"
        )


# ---------------------------------------------------------------------------
# StatisticalTester
# ---------------------------------------------------------------------------

class StatisticalTester:
    """Runs Mann-Whitney U tests and bootstrap CIs on latency distributions.

    Args:
        alpha: Significance level for hypothesis tests (default: 0.05).
        n_bootstrap: Number of bootstrap resamples for CI estimation.
        ci_level: Confidence level for bootstrap CIs (default: 0.95).
        use_scipy: Whether to use scipy for faster Mann-Whitney U computation
            when available (default: True).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        use_scipy: bool = True,
    ) -> None:
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self._use_scipy = use_scipy and _HAS_SCIPY

    def _mean(self, data: List[float]) -> float:
        return sum(data) / len(data) if data else 0.0

    def _mann_whitney(
        self, x: List[float], y: List[float]
    ) -> Tuple[float, float]:
        if self._use_scipy:
            res = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        return _mann_whitney_u(x, y)

    def test(
        self,
        sample_a: List[float],
        sample_b: List[float],
        metric_name: str = "latency_ms",
    ) -> SignificanceResult:
        """Run a two-sample significance test.

        Args:
            sample_a: Latency observations for configuration A (ms).
            sample_b: Latency observations for configuration B (ms).
            metric_name: Name of the metric for reporting.

        Returns:
            SignificanceResult with p-value, effect size, and bootstrap CIs.

        Raises:
            ValueError: If either sample is empty.
        """
        if not sample_a or not sample_b:
            raise ValueError("Both samples must be non-empty")

        u_stat, p_value = self._mann_whitney(sample_a, sample_b)

        mean_a = self._mean(sample_a)
        mean_b = self._mean(sample_b)
        d = cohens_d(sample_a, sample_b)

        ci_a = bootstrap_ci(
            sample_a, self._mean, self.n_bootstrap, self.ci_level
        )
        ci_b = bootstrap_ci(
            sample_b, self._mean, self.n_bootstrap, self.ci_level
        )

        return SignificanceResult(
            metric_name=metric_name,
            sample_a_size=len(sample_a),
            sample_b_size=len(sample_b),
            mean_a=mean_a,
            mean_b=mean_b,
            u_statistic=u_stat,
            p_value=p_value,
            cohens_d=d,
            ci_mean_a=ci_a,
            ci_mean_b=ci_b,
            alpha=self.alpha,
        )

    def test_ttft(
        self,
        ttft_a: List[float],
        ttft_b: List[float],
    ) -> SignificanceResult:
        """Test significance of TTFT difference between two configs.

        Args:
            ttft_a: TTFT latencies (ms) for config A.
            ttft_b: TTFT latencies (ms) for config B.

        Returns:
            SignificanceResult for TTFT.
        """
        return self.test(ttft_a, ttft_b, metric_name="ttft_ms")

    def test_tpot(
        self,
        tpot_a: List[float],
        tpot_b: List[float],
    ) -> SignificanceResult:
        """Test significance of TPOT difference between two configs.

        Args:
            tpot_a: TPOT latencies (ms) for config A.
            tpot_b: TPOT latencies (ms) for config B.

        Returns:
            SignificanceResult for TPOT.
        """
        return self.test(tpot_a, tpot_b, metric_name="tpot_ms")

    def test_e2e(
        self,
        e2e_a: List[float],
        e2e_b: List[float],
    ) -> SignificanceResult:
        """Test significance of end-to-end latency difference.

        Args:
            e2e_a: E2E latencies (ms) for config A.
            e2e_b: E2E latencies (ms) for config B.

        Returns:
            SignificanceResult for E2E latency.
        """
        return self.test(e2e_a, e2e_b, metric_name="e2e_latency_ms")
