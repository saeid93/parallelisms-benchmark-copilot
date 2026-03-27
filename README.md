# parallelisms-benchmark-copilot

vLLM Parallelism & Serving Config Research Benchmark Framework.

## Overview

This framework provides configuration sweep generation, validation, and
structured output for benchmarking vLLM serving configurations across:

- **Parallelism strategies** — tensor, pipeline, data, expert parallelism
  (unified) and disaggregated prefill/decode placement (DistServe, Seesaw)
- **Batching & scheduling** — token budgets, sequence limits, chunked
  prefill (SARATHI-style)
- **KV cache** — dtype, block size, memory utilisation, offloading
- **Compute & precision** — dtype, quantisation, CUDA graph capture
- **Attention backends** — FlashAttention, FlashInfer, Triton
- **Workloads** — dataset, arrival process, request rate, I/O lengths
- **SLO targets** — TTFT, TPOT, attainment percentile (DistServe suite)

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Usage

```python
from benchmark_config import SweepGenerator, ConfigValidator, BenchmarkResult

# Generate a parallelism sweep (TP × PP grid, 8 GPUs)
gen = SweepGenerator(max_gpus=8)
for cfg in gen.sweep_parallelism(tp_sizes=[1, 2, 4], pp_sizes=[1, 2, 4]):
    print(cfg.parallelism.tensor_parallel_size, cfg.parallelism.pipeline_parallel_size)

# Validate a single config
validator = ConfigValidator(max_gpus=8)
errors = validator.validate(cfg)

# Structured result output
result = BenchmarkResult(config=cfg, status="completed")
print(result.to_json())
```

## Package structure

```
benchmark_config/
  __init__.py           # Public API
  schema.py             # Config dataclasses (Sections 1A-1F)
  result_schema.py      # Structured output (Section 2)
  validation.py         # Field + cross-field validation
  sweep.py              # Sweep grid generation
tests/
  test_schema.py
  test_validation.py
  test_sweep.py
  test_result_schema.py
```