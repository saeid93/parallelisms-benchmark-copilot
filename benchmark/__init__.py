"""vLLM Parallelism & Serving Config Research Agent.

A fully-fledged profiler and benchmarking framework for testing many
different configurations and situations with full traces.

Modules
-------
profiler/
    gpu_profiler         — NVML-based GPU hardware sampling
    trace_recorder       — Chrome Trace / OTLP request tracing
    memory_profiler      — GPU memory allocation tracking & fragmentation
    communication_profiler — NCCL collective profiling & bandwidth analysis
    energy_profiler      — Power, energy & carbon footprint tracking
    tokenizer_profiler   — Tokenisation overhead & vocab utilisation
    attention_profiler   — Attention kernel & sparsity analysis
    request_lifecycle    — End-to-end request Gantt charts & tail-latency causes

analysis/
    slo_evaluator        — Binary-search goodput & SLO attainment
    pareto               — Pareto frontier computation
    recommender          — Configuration recommendation synthesis
    bottleneck           — Bottleneck classification (9 categories)
    cost_estimator       — GPU cost-per-token estimation
    regression           — Performance regression detection
    statistical          — Mann-Whitney U, bootstrap CI, Cohen's d
    scheduler_analyzer   — Batch scheduling & fairness (Jain's index)
    roofline             — Berkeley Roofline model analysis
    scalability          — Amdahl / Gustafson scaling analysis
    pipeline_bubble      — Pipeline-parallel bubble quantification
    quantization_analyzer — Quantisation quality/speed trade-off
    kv_cache_analyzer    — KV cache utilisation & block-size tuning
    speculative_analyzer — Speculative decoding depth optimisation
    anomaly_detector     — Z-score / IQR anomaly detection
    workload_characterizer — Request distribution & phase detection

config/
    schema               — Pydantic config & metrics models
    sweep                — Grid-search sweep generation
    validation           — Configuration feasibility rules
    space_explorer       — Parameter sensitivity & interaction analysis

reporting/
    exporter             — JSON / CSV / Markdown / HTML export
    comparative_reporter — Side-by-side comparison & executive summary
    dashboard            — Structured JSON for Grafana / web dashboards
"""
