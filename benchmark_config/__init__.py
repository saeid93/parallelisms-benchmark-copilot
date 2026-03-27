"""vLLM Parallelism & Serving Config Research Benchmark Framework.

Provides configuration sweep generation, validation, and structured output
for benchmarking vLLM serving configurations across parallelism strategies,
batching schemes, KV cache settings, and workload profiles.
"""

from benchmark_config.schema import (
    ParallelismConfig,
    DisaggregatedConfig,
    DynamicReshardingConfig,
    BatchingConfig,
    ChunkedPrefillConfig,
    KVCacheConfig,
    ComputePrecisionConfig,
    AttentionBackendConfig,
    WorkloadConfig,
    SLOTargetConfig,
    BenchmarkRunConfig,
)
from benchmark_config.result_schema import BenchmarkResult
from benchmark_config.sweep import SweepGenerator
from benchmark_config.validation import ConfigValidator

__all__ = [
    "ParallelismConfig",
    "DisaggregatedConfig",
    "DynamicReshardingConfig",
    "BatchingConfig",
    "ChunkedPrefillConfig",
    "KVCacheConfig",
    "ComputePrecisionConfig",
    "AttentionBackendConfig",
    "WorkloadConfig",
    "SLOTargetConfig",
    "BenchmarkRunConfig",
    "BenchmarkResult",
    "SweepGenerator",
    "ConfigValidator",
]
