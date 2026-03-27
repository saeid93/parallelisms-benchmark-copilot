"""Tests for the workload generator (Stage 2)."""

import pytest

from benchmark.workload.generator import (
    WorkloadGenerator,
    WorkloadRequest,
    _offline_arrivals,
    _poisson_arrivals,
    _sample_uniform,
    _sample_zipf,
)
import random


class TestArrivalGenerators:
    def test_offline_all_zero(self):
        arrivals = _offline_arrivals(10)
        assert len(arrivals) == 10
        assert all(t == 0.0 for t in arrivals)

    def test_poisson_strictly_increasing(self):
        rng = random.Random(42)
        arrivals = _poisson_arrivals(100, rate_rps=2.0, rng=rng)
        assert len(arrivals) == 100
        for i in range(1, len(arrivals)):
            assert arrivals[i] > arrivals[i - 1]

    def test_poisson_invalid_rate(self):
        rng = random.Random(0)
        with pytest.raises(ValueError, match="request_rate_rps must be > 0"):
            _poisson_arrivals(10, rate_rps=0.0, rng=rng)

    def test_poisson_rate_mean(self):
        rng = random.Random(123)
        n = 10_000
        rate = 5.0
        arrivals = _poisson_arrivals(n, rate_rps=rate, rng=rng)
        # Mean inter-arrival ≈ 1/rate; total time ≈ n/rate
        expected_total = n / rate
        assert abs(arrivals[-1] - expected_total) / expected_total < 0.05


class TestTokenSamplers:
    def test_uniform_in_range(self):
        rng = random.Random(0)
        for _ in range(100):
            val = _sample_uniform(rng, 10, 50)
            assert 10 <= val <= 50

    def test_zipf_in_range(self):
        rng = random.Random(0)
        for _ in range(100):
            val = _sample_zipf(rng, 1000)
            assert 1 <= val <= 1000


class TestWorkloadGenerator:
    def test_generate_offline(self):
        gen = WorkloadGenerator(
            dataset="sharegpt",
            arrival_process="offline",
            num_requests=50,
            seed=0,
        )
        requests = gen.generate()
        assert len(requests) == 50
        assert all(r.arrival_time_s == 0.0 for r in requests)

    def test_generate_poisson(self):
        gen = WorkloadGenerator(
            dataset="sharegpt",
            arrival_process="poisson",
            request_rate_rps=2.0,
            num_requests=100,
            seed=42,
        )
        requests = gen.generate()
        assert len(requests) == 100
        # Arrival times should be strictly increasing
        for i in range(1, len(requests)):
            assert requests[i].arrival_time_s > requests[i - 1].arrival_time_s

    def test_invalid_arrival_process(self):
        gen = WorkloadGenerator(arrival_process="batch", num_requests=10)
        with pytest.raises(ValueError, match="Unknown arrival_process"):
            gen.generate()

    def test_synthetic_uniform_tokens(self):
        gen = WorkloadGenerator(
            dataset="synthetic_uniform",
            arrival_process="offline",
            num_requests=200,
            seed=7,
        )
        requests = gen.generate()
        assert all(r.input_tokens >= 128 for r in requests)
        assert all(r.input_tokens <= 1024 for r in requests)

    def test_synthetic_zipf_tokens(self):
        gen = WorkloadGenerator(
            dataset="synthetic_zipf",
            arrival_process="offline",
            num_requests=200,
            seed=7,
        )
        requests = gen.generate()
        assert all(r.input_tokens >= 1 for r in requests)

    def test_reproducible_with_same_seed(self):
        kwargs = dict(
            dataset="sharegpt",
            arrival_process="poisson",
            request_rate_rps=1.0,
            num_requests=20,
            seed=99,
        )
        gen1 = WorkloadGenerator(**kwargs)
        gen2 = WorkloadGenerator(**kwargs)
        r1 = gen1.generate()
        r2 = gen2.generate()
        assert all(a.input_tokens == b.input_tokens for a, b in zip(r1, r2))
        assert all(a.arrival_time_s == b.arrival_time_s for a, b in zip(r1, r2))

    def test_different_seeds_differ(self):
        gen1 = WorkloadGenerator(seed=1, num_requests=50)
        gen2 = WorkloadGenerator(seed=2, num_requests=50)
        r1 = gen1.generate()
        r2 = gen2.generate()
        tokens1 = [r.input_tokens for r in r1]
        tokens2 = [r.input_tokens for r in r2]
        assert tokens1 != tokens2

    def test_compute_actual_stats(self):
        gen = WorkloadGenerator(
            dataset="synthetic_uniform",
            arrival_process="offline",
            num_requests=100,
            seed=0,
        )
        requests = gen.generate()
        avg_in, avg_out, pd_ratio = gen.compute_actual_stats(requests)
        assert avg_in > 0
        assert avg_out > 0
        assert pd_ratio is not None
        assert pd_ratio > 0

    def test_compute_actual_stats_empty(self):
        gen = WorkloadGenerator()
        avg_in, avg_out, pd_ratio = gen.compute_actual_stats([])
        assert avg_in == 0.0
        assert avg_out == 0.0
        assert pd_ratio is None

    def test_iter_requests(self):
        gen = WorkloadGenerator(num_requests=10, arrival_process="offline")
        items = list(gen.iter_requests())
        assert len(items) == 10
        assert all(isinstance(r, WorkloadRequest) for r in items)

    def test_all_datasets(self):
        datasets = [
            "sharegpt", "humaneval", "longbench",
            "arxiv_summarization", "synthetic_uniform", "synthetic_zipf",
        ]
        for dataset in datasets:
            gen = WorkloadGenerator(
                dataset=dataset,
                arrival_process="offline",
                num_requests=10,
                seed=0,
            )
            requests = gen.generate()
            assert len(requests) == 10, f"Failed for dataset={dataset}"
