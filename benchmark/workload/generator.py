"""
Stage 2 — Workload generator.

Replays datasets with configurable arrival processes (Poisson or offline
batch) and synthetic distributions (uniform / Zipf) as specified in
Section 1F of the problem statement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Request dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkloadRequest:
    """A single synthetic request emitted by the workload generator."""

    request_id: int
    input_tokens: int
    output_tokens: int
    arrival_time_s: float = 0.0
    dataset: str = "synthetic_uniform"


# ---------------------------------------------------------------------------
# Token-length samplers
# ---------------------------------------------------------------------------

def _sample_uniform(
    rng: random.Random,
    min_len: int,
    max_len: int,
) -> int:
    return rng.randint(min_len, max_len)


def _sample_zipf(rng: random.Random, n: int, alpha: float = 1.5) -> int:
    """Inverse CDF sampler for Zipf distribution over [1, n]."""
    # Use rejection sampling against a simple approximation.
    x = rng.uniform(0, 1)
    # Approximate harmonic number H_n ≈ ln(n) + 0.5772
    h_n = math.log(n) + 0.5772
    # Map uniform to Zipf rank, clamped to [1, n]
    rank = max(1, min(n, int(math.exp(h_n * x))))
    return rank


# Dataset approximate token-length ranges (derived from public benchmarks).
_DATASET_INPUT_RANGE = {
    "sharegpt": (64, 2048),
    "humaneval": (128, 512),
    "longbench": (1024, 4096),
    "arxiv_summarization": (2048, 8192),
    "synthetic_uniform": (128, 1024),
    "synthetic_zipf": (1, 4096),
}

_DATASET_OUTPUT_RANGE = {
    "sharegpt": (64, 512),
    "humaneval": (64, 256),
    "longbench": (64, 512),
    "arxiv_summarization": (128, 512),
    "synthetic_uniform": (64, 256),
    "synthetic_zipf": (1, 1024),
}


# ---------------------------------------------------------------------------
# Arrival-time generators
# ---------------------------------------------------------------------------

def _offline_arrivals(num_requests: int) -> List[float]:
    """All requests arrive at t=0 (offline batch mode)."""
    return [0.0] * num_requests


def _poisson_arrivals(
    num_requests: int,
    rate_rps: float,
    rng: random.Random,
) -> List[float]:
    """Generate arrival times from a Poisson process at *rate_rps* req/s."""
    if rate_rps <= 0:
        raise ValueError(f"request_rate_rps must be > 0, got {rate_rps}")
    arrival_times: List[float] = []
    t = 0.0
    for _ in range(num_requests):
        # Inter-arrival time ~ Exponential(rate_rps)
        t += rng.expovariate(rate_rps)
        arrival_times.append(t)
    return arrival_times


# ---------------------------------------------------------------------------
# WorkloadGenerator
# ---------------------------------------------------------------------------

@dataclass
class WorkloadGenerator:
    """Generate synthetic workload requests for a benchmark config.

    Attributes:
        dataset: Dataset name from Section 1F.
        arrival_process: Either "offline" or "poisson".
        request_rate_rps: Arrival rate for Poisson process (ignored for
            offline).
        num_requests: Total number of requests to generate.
        avg_input_tokens: Target average input length (informational).
        avg_output_tokens: Target average output length (informational).
        seed: Random seed for reproducibility.
    """

    dataset: str = "sharegpt"
    arrival_process: str = "poisson"
    request_rate_rps: float = 2.0
    num_requests: int = 1000
    avg_input_tokens: int = 755
    avg_output_tokens: int = 200
    seed: int = 42

    def generate(self) -> List[WorkloadRequest]:
        """Generate and return the list of requests.

        Returns:
            Ordered list of WorkloadRequest objects with arrival times.
        """
        rng = random.Random(self.seed)

        # Token-length sampler
        in_min, in_max = _DATASET_INPUT_RANGE.get(
            self.dataset, (128, 1024)
        )
        out_min, out_max = _DATASET_OUTPUT_RANGE.get(
            self.dataset, (64, 256)
        )

        def sample_input() -> int:
            if self.dataset == "synthetic_zipf":
                return _sample_zipf(rng, in_max)
            return _sample_uniform(rng, in_min, in_max)

        def sample_output() -> int:
            if self.dataset == "synthetic_zipf":
                return _sample_zipf(rng, out_max)
            return _sample_uniform(rng, out_min, out_max)

        # Arrival times
        if self.arrival_process == "offline":
            arrivals = _offline_arrivals(self.num_requests)
        elif self.arrival_process == "poisson":
            arrivals = _poisson_arrivals(
                self.num_requests, self.request_rate_rps, rng
            )
        else:
            raise ValueError(
                f"Unknown arrival_process: {self.arrival_process!r}. "
                "Expected 'offline' or 'poisson'."
            )

        requests = []
        for i, arrival in enumerate(arrivals):
            requests.append(
                WorkloadRequest(
                    request_id=i,
                    input_tokens=sample_input(),
                    output_tokens=sample_output(),
                    arrival_time_s=arrival,
                    dataset=self.dataset,
                )
            )
        return requests

    def compute_actual_stats(
        self, requests: List[WorkloadRequest]
    ) -> Tuple[float, float, Optional[float]]:
        """Compute actual average token counts and pd_ratio.

        Args:
            requests: Generated request list.

        Returns:
            Tuple of (avg_input_tokens, avg_output_tokens, pd_ratio_actual).
            pd_ratio_actual is the ratio of total prefill tokens to total
            decode tokens across the batch.
        """
        if not requests:
            return 0.0, 0.0, None
        total_input = sum(r.input_tokens for r in requests)
        total_output = sum(r.output_tokens for r in requests)
        avg_input = total_input / len(requests)
        avg_output = total_output / len(requests)
        pd_ratio = total_input / total_output if total_output > 0 else None
        return avg_input, avg_output, pd_ratio

    def iter_requests(self) -> Iterator[WorkloadRequest]:
        """Lazily yield requests one by one."""
        yield from self.generate()
