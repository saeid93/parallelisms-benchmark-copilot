# parallelisms-benchmark-copilot

**vLLM Parallelism & Serving Config Research Agent** — a 7-stage iterative DAG benchmark system for exploring the full vLLM parallelism and serving configuration space.

## Architecture

The system implements the following pipeline stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `benchmark/config/sweep.py` | Config space generator — emits the full sweep matrix and prunes infeasible combinations |
| 2 | `benchmark/workload/generator.py` | Workload generator — Poisson/offline arrivals, synthetic/dataset token distributions |
| 3 | `benchmark/runner/benchmark_runner.py` | Parallel benchmark runners — Kubernetes Job orchestration with early stopping |
| 4 | `benchmark/metrics/collector.py` | Metrics collector — Prometheus scraping, per-request TTFT/TPOT/e2e, SLO attainment |
| 5 | `benchmark/analysis/slo_evaluator.py` | SLO attainment evaluator — binary-search goodput curves, slo_scale sweep |
| 6 | `benchmark/analysis/pareto.py` | Pareto analyser — multi-objective Pareto frontier per benchmark suite |
| 7 | `benchmark/analysis/recommender.py` | Recommendation synthesiser — plain-text tradeoff report, paper cross-validation |

## Config Sweep Dimensions

- **Parallelism**: TP × PP cross-product grid, DP replicas, disaggregated prefill/decode (DistServe), dynamic re-sharding (Seesaw), expert parallel (MoE)
- **Batching**: Chunked prefill (SARATHI), batching schemes, pd_ratio sweep
- **KV Cache**: dtype, block size, prefix caching, CPU offload, KV offloading backends
- **Compute**: dtype, quantization, attention backend (FlashAttn v2/v3/v4, FlashInfer, Triton)
- **Workload**: 6 datasets, Poisson/offline arrivals, request rate sweep for goodput curves
- **SLO**: TTFT/TPOT targets, attainment percentage, SLO scale sweep

## Benchmark Suites

| Suite | Focus |
|-------|-------|
| `vllm_parallelism` | General TP/PP/DP grid — throughput vs latency vs memory |
| `distserve` | Disaggregated prefill/decode — goodput per GPU under SLO |
| `sarathi` | Chunked prefill — decode speedup vs pipeline bubble ratio |
| `seesaw` | Dynamic re-sharding — offline throughput, prefill/decode balance |

## Output Schema

Each benchmark run emits a JSON record conforming to the `BenchmarkRun` Pydantic model (`benchmark/config/schema.py`), including full `config` and `metrics` sub-objects as defined in Section 2 of the problem statement.

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline (dry-run mode, no real vLLM cluster required)
python -c "
from benchmark.dag.pipeline import BenchmarkPipeline, PipelineConfig
from benchmark.config.schema import BenchmarkMetrics

# Provide a synthetic metrics function for local testing
def mock_metrics(cfg):
    m = BenchmarkMetrics()
    m.throughput_tps = cfg.tp * 1000.0
    m.goodput_rps = cfg.tp * 2.0
    m.joint_slo_attainment_pct = 95.0
    return m

pipeline = BenchmarkPipeline(
    config=PipelineConfig(
        max_gpus=4,
        model_params_gb=7.0,
        suites=['vllm_parallelism'],
        dry_run=True,
    ),
    metrics_fn_override=mock_metrics,
)
report = pipeline.run()
print(report)
"
```

## Tests

```bash
python -m pytest tests/ -v
```

## Kubernetes Deployment

Job and ConfigMap templates are in `k8s/`. Each benchmark run is submitted as a separate `batch/v1 Job` with GPU resource requests matching `tp × pp` (or `prefill_tp × prefill_pp + decode_tp × decode_pp` for disaggregated configs). Results are persisted to a PersistentVolumeClaim and scraped via Prometheus.
