# parallelisms-benchmark-copilot

**vLLM Parallelism & Serving Config Research Agent** — a 14-stage iterative DAG benchmark system for exploring the full vLLM parallelism and serving configuration space.

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
| 8 | `benchmark/config/validation.py` | Config validator — pre-submission feasibility checks |
| 9 | `benchmark/analysis/bottleneck.py` | Bottleneck analyser — classify performance bottlenecks |
| 10 | `benchmark/analysis/cost_estimator.py` | Cost estimator — GPU cloud cost estimation per config |
| 11 | `benchmark/analysis/regression.py` | Regression detector — compare against saved baselines |
| 12 | `benchmark/reporting/exporter.py` | Report exporter — JSON, CSV, Markdown, HTML export |
| 13 | `benchmark/profiler/gpu_profiler.py` | GPU profiler — NVML hardware sampling (util, mem, power, NVLink) |
| 14 | `benchmark/profiler/trace_recorder.py` | Trace recorder — per-request Chrome TEF / OTLP / flame graph export |

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

## Per-Request Trace Recording

The trace recorder (Stage 14) captures complete per-request execution traces:

- **Request lifecycle phases**: queue → prefill → decode → completion
- **Decode token iteration timing**: per-output-token step durations for TPOT analysis
- **Layer-level timing**: per-layer execution times during prefill and decode
- **GPU kernel events**: kernel launch timestamps and durations per CUDA stream
- **KV cache events**: hit, miss, eviction, swap-in, swap-out with block counts

Export formats:
- **Chrome DevTools / Perfetto** (JSON Trace Event Format)
- **OpenTelemetry spans** (OTLP-compatible JSON, suitable for Jaeger/Zipkin)
- **Flame graph** (folded-stacks format)

## Prometheus Integration

The Prometheus bridge (`benchmark/metrics/prometheus_bridge.py`) provides:

- **ServiceMonitor-aware scraping** in Kubernetes mode — auto-discovers vLLM pods via in-cluster Service DNS
- **Pushgateway support** — pushes aggregated per-run metrics so they survive pod termination
- **vLLM metric parsing** — extracts throughput, latency histograms, cache usage, scheduler state from vLLM's `/metrics` endpoint
- **Metrics enrichment** — augments `BenchmarkMetrics` with live Prometheus data

## Kubernetes Deployment (Helm)

All Kubernetes resources are packaged as a Helm chart in `helm/benchmark/`:

```bash
# Install the benchmark infrastructure
helm upgrade --install benchmark helm/benchmark/ \
  --namespace benchmark \
  --create-namespace \
  --set image.repository=vllm/vllm-openai \
  --set image.tag=latest
```

Helm chart includes:
- **Namespace**, **ServiceAccount**, **RBAC** (Role + RoleBinding)
- **PersistentVolumeClaim** for result storage
- **Service** fronting vLLM pods for Prometheus scraping
- **ServiceMonitor** for Prometheus Operator auto-discovery
- **Job** and **ConfigMap** templates for benchmark runs

## Automation Script

The `scripts/run_benchmark.sh` script automates the full workflow:

```bash
./scripts/run_benchmark.sh \
  --namespace benchmark \
  --image vllm/vllm-openai:latest \
  --max-gpus 8 \
  --suites vllm_parallelism,distserve \
  --results-dir ./results \
  --pushgateway-url http://pushgw:9091
```

The script:
1. Validates prerequisites (helm, kubectl, python3)
2. Installs / upgrades the Helm chart
3. Runs the Python benchmark pipeline in Kubernetes mode
4. Collects results
5. Optionally tears down the Helm release (`--teardown`)

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
