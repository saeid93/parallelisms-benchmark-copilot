"""Tests for the statistical significance tester (Stage analysis)."""

import math
import random

import pytest

from benchmark.analysis.statistical import (
    StatisticalTester,
    SignificanceResult,
    bootstrap_ci,
    cohens_d,
    _mann_whitney_u,
    _norm_cdf,
)


class TestNormCdf:
    def test_zero_point_five(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_positive_z(self):
        assert _norm_cdf(1.96) > 0.97

    def test_negative_z(self):
        assert _norm_cdf(-1.96) < 0.03

    def test_symmetry(self):
        assert abs(_norm_cdf(1.0) + _norm_cdf(-1.0) - 1.0) < 1e-6


class TestMannWhitneyU:
    def test_identical_samples(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        u, p = _mann_whitney_u(x, y)
        # p should be high (not significant)
        assert p > 0.05

    def test_clearly_different_samples(self):
        x = list(range(1, 51))       # 1–50
        y = list(range(101, 151))    # 101–150
        u, p = _mann_whitney_u(x, y)
        # Very different; p should be tiny
        assert p < 0.01

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _mann_whitney_u([], [1.0, 2.0])

    def test_result_types(self):
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        u, p = _mann_whitney_u(x, y)
        assert isinstance(u, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestBootstrapCI:
    def test_returns_tuple(self):
        data = list(range(1, 101))
        lo, hi = bootstrap_ci(data, lambda d: sum(d) / len(d), n_bootstrap=200)
        assert lo < hi

    def test_ci_contains_true_mean(self):
        rng = random.Random(0)
        data = [rng.gauss(50.0, 5.0) for _ in range(200)]
        lo, hi = bootstrap_ci(data, lambda d: sum(d) / len(d), n_bootstrap=1000, seed=42)
        assert lo <= 50.0 <= hi or abs(lo - 50.0) < 5.0  # generous check

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_ci([], lambda d: 0.0)

    def test_reproducible_with_same_seed(self):
        data = [float(i) for i in range(50)]
        lo1, hi1 = bootstrap_ci(data, lambda d: sum(d) / len(d), seed=7)
        lo2, hi2 = bootstrap_ci(data, lambda d: sum(d) / len(d), seed=7)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seeds_may_differ(self):
        data = [float(i) for i in range(50)]
        lo1, _ = bootstrap_ci(data, lambda d: sum(d) / len(d), seed=1, n_bootstrap=100)
        lo2, _ = bootstrap_ci(data, lambda d: sum(d) / len(d), seed=2, n_bootstrap=100)
        # Not guaranteed to differ but almost always will with different seeds
        # Just check they are both finite
        assert math.isfinite(lo1)
        assert math.isfinite(lo2)


class TestCohensD:
    def test_identical_samples_zero_d(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cohens_d(x, y) == 0.0

    def test_large_effect(self):
        # Use data with variance (not identical values) to get a valid pooled SD
        import random
        rng = random.Random(42)
        x = [100.0 + rng.gauss(0, 2.0) for _ in range(20)]
        y = [1.0 + rng.gauss(0, 2.0) for _ in range(20)]
        d = cohens_d(x, y)
        assert abs(d) > 1.0  # large effect

    def test_sign(self):
        import random
        rng = random.Random(7)
        x = [10.0 + rng.gauss(0, 1.0) for _ in range(20)]
        y = [1.0 + rng.gauss(0, 1.0) for _ in range(20)]
        d = cohens_d(x, y)
        assert d > 0  # x > y (mean_x ≈ 10 > mean_y ≈ 1)

        d2 = cohens_d(y, x)
        assert d2 < 0  # y < x

    def test_too_few_samples(self):
        assert cohens_d([1.0], [1.0]) == 0.0

    def test_zero_variance(self):
        x = [5.0] * 10
        y = [5.0] * 10
        assert cohens_d(x, y) == 0.0


class TestStatisticalTester:
    def test_significant_difference(self):
        rng = random.Random(42)
        a = [rng.gauss(100.0, 5.0) for _ in range(100)]
        b = [rng.gauss(200.0, 5.0) for _ in range(100)]
        tester = StatisticalTester(alpha=0.05, n_bootstrap=200, use_scipy=False)
        result = tester.test(a, b, metric_name="ttft_ms")
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.metric_name == "ttft_ms"

    def test_not_significant_identical(self):
        rng = random.Random(0)
        data = [rng.gauss(50.0, 3.0) for _ in range(50)]
        tester = StatisticalTester(alpha=0.05, n_bootstrap=200, use_scipy=False)
        result = tester.test(data, data, metric_name="e2e")
        assert not result.is_significant

    def test_empty_sample_raises(self):
        tester = StatisticalTester(use_scipy=False)
        with pytest.raises(ValueError):
            tester.test([], [1.0, 2.0, 3.0])

    def test_effect_size_label_negligible(self):
        result = SignificanceResult(
            metric_name="x",
            sample_a_size=10,
            sample_b_size=10,
            mean_a=1.0,
            mean_b=1.0,
            u_statistic=50.0,
            p_value=0.8,
            cohens_d=0.1,
            ci_mean_a=(0.9, 1.1),
            ci_mean_b=(0.9, 1.1),
        )
        assert result.effect_size_label == "negligible"

    def test_effect_size_label_large(self):
        result = SignificanceResult(
            metric_name="x",
            sample_a_size=100,
            sample_b_size=100,
            mean_a=100.0,
            mean_b=1.0,
            u_statistic=50.0,
            p_value=0.001,
            cohens_d=2.5,
            ci_mean_a=(90.0, 110.0),
            ci_mean_b=(0.5, 1.5),
        )
        assert result.effect_size_label == "large"

    def test_test_ttft_convenience(self):
        rng = random.Random(1)
        a = [rng.gauss(100.0, 10.0) for _ in range(50)]
        b = [rng.gauss(150.0, 10.0) for _ in range(50)]
        tester = StatisticalTester(n_bootstrap=100, use_scipy=False)
        result = tester.test_ttft(a, b)
        assert result.metric_name == "ttft_ms"

    def test_test_tpot_convenience(self):
        rng = random.Random(2)
        a = [rng.gauss(50.0, 5.0) for _ in range(50)]
        b = [rng.gauss(60.0, 5.0) for _ in range(50)]
        tester = StatisticalTester(n_bootstrap=100, use_scipy=False)
        result = tester.test_tpot(a, b)
        assert result.metric_name == "tpot_ms"

    def test_test_e2e_convenience(self):
        rng = random.Random(3)
        a = [rng.gauss(200.0, 20.0) for _ in range(50)]
        b = [rng.gauss(220.0, 20.0) for _ in range(50)]
        tester = StatisticalTester(n_bootstrap=100, use_scipy=False)
        result = tester.test_e2e(a, b)
        assert result.metric_name == "e2e_latency_ms"

    def test_summary_string(self):
        rng = random.Random(42)
        a = [rng.gauss(100.0, 5.0) for _ in range(50)]
        b = [rng.gauss(200.0, 5.0) for _ in range(50)]
        tester = StatisticalTester(n_bootstrap=100, use_scipy=False)
        result = tester.test(a, b, "ttft_ms")
        s = result.summary()
        assert "ttft_ms" in s
        assert "p=" in s
