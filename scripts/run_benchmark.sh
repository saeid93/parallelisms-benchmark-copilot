#!/usr/bin/env bash
# --------------------------------------------------------------------------
# run_benchmark.sh — End-to-end automation script for the vLLM parallelism
# benchmark pipeline on Kubernetes.
#
# This script:
#   1. Validates prerequisites (helm, kubectl, python)
#   2. Installs / upgrades the Helm chart to provision K8s resources
#   3. Runs the Python benchmark pipeline in Kubernetes mode
#   4. Collects results from the PVC
#   5. Optionally tears down the Helm release
#
# Usage:
#   ./scripts/run_benchmark.sh [OPTIONS]
#
# Options:
#   --namespace       Kubernetes namespace       (default: benchmark)
#   --release         Helm release name          (default: benchmark)
#   --kubeconfig      Path to kubeconfig         (default: $KUBECONFIG)
#   --image           Container image            (default: vllm/vllm-openai:latest)
#   --max-gpus        Maximum GPUs to use        (default: 8)
#   --suites          Comma-separated suites     (default: vllm_parallelism)
#   --model-variants  Comma-separated variants   (default: none)
#   --results-dir     Local results directory    (default: ./results)
#   --values          Extra Helm values file     (default: none)
#   --dry-run         Pipeline dry-run mode      (default: false)
#   --teardown        Remove Helm release after  (default: false)
#   --pushgateway-url Prometheus pushgateway URL (default: none)
#   --help            Show this help message
# --------------------------------------------------------------------------
set -euo pipefail

# ---- Defaults ----
NAMESPACE="benchmark"
RELEASE="benchmark"
KUBECONFIG_PATH="${KUBECONFIG:-}"
IMAGE="vllm/vllm-openai:latest"
MAX_GPUS=8
SUITES="vllm_parallelism"
MODEL_VARIANTS=""
RESULTS_DIR="./results"
VALUES_FILE=""
DRY_RUN="false"
TEARDOWN="false"
PUSHGATEWAY_URL=""
CHART_DIR="$(cd "$(dirname "$0")/../helm/benchmark" && pwd)"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)       NAMESPACE="$2";       shift 2 ;;
        --release)         RELEASE="$2";         shift 2 ;;
        --kubeconfig)      KUBECONFIG_PATH="$2"; shift 2 ;;
        --image)           IMAGE="$2";           shift 2 ;;
        --max-gpus)        MAX_GPUS="$2";        shift 2 ;;
        --suites)          SUITES="$2";          shift 2 ;;
        --model-variants)  MODEL_VARIANTS="$2";  shift 2 ;;
        --results-dir)     RESULTS_DIR="$2";     shift 2 ;;
        --values)          VALUES_FILE="$2";     shift 2 ;;
        --dry-run)         DRY_RUN="true";       shift   ;;
        --teardown)        TEARDOWN="true";      shift   ;;
        --pushgateway-url) PUSHGATEWAY_URL="$2"; shift 2 ;;
        --help)
            sed -n '2,/^# ----/{ /^# ----/d; s/^# //; p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---- Prerequisite checks ----
echo "=== Checking prerequisites ==="

for cmd in helm kubectl python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is required but not found on PATH." >&2
        exit 1
    fi
done

echo "  helm:    $(helm version --short 2>/dev/null || echo 'unknown')"
echo "  kubectl: $(kubectl version --client --short 2>/dev/null || echo 'unknown')"
echo "  python3: $(python3 --version 2>/dev/null || echo 'unknown')"

# ---- Helm install / upgrade ----
echo ""
echo "=== Installing / upgrading Helm chart ==="

HELM_CMD=(helm upgrade --install "$RELEASE" "$CHART_DIR"
    --namespace "$NAMESPACE"
    --create-namespace
    --set "namespace=$NAMESPACE"
    --set "image.repository=${IMAGE%%:*}"
    --set "image.tag=${IMAGE##*:}"
)

if [[ -n "$KUBECONFIG_PATH" ]]; then
    export KUBECONFIG="$KUBECONFIG_PATH"
fi

if [[ -n "$VALUES_FILE" ]]; then
    HELM_CMD+=(-f "$VALUES_FILE")
fi

if [[ -n "$PUSHGATEWAY_URL" ]]; then
    HELM_CMD+=(
        --set "prometheus.pushgateway.enabled=true"
        --set "prometheus.pushgateway.url=$PUSHGATEWAY_URL"
    )
fi

echo "  ${HELM_CMD[*]}"
"${HELM_CMD[@]}"

echo ""
echo "=== Helm release '$RELEASE' deployed to namespace '$NAMESPACE' ==="

# ---- Wait for resources ----
echo ""
echo "=== Waiting for PVC to be bound ==="
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
    pvc/benchmark-results \
    -n "$NAMESPACE" \
    --timeout=120s 2>/dev/null || echo "  (PVC wait skipped or timed out — continuing)"

# ---- Run the benchmark pipeline ----
echo ""
echo "=== Running benchmark pipeline ==="

PIPELINE_ARGS=(
    "--execution-mode" "kubernetes"
    "--namespace" "$NAMESPACE"
    "--image" "$IMAGE"
    "--max-gpus" "$MAX_GPUS"
    "--suites" "$SUITES"
    "--results-dir" "$RESULTS_DIR"
)

if [[ -n "$KUBECONFIG_PATH" ]]; then
    PIPELINE_ARGS+=("--kubeconfig" "$KUBECONFIG_PATH")
fi

if [[ -n "$MODEL_VARIANTS" ]]; then
    PIPELINE_ARGS+=("--model-variants" "$MODEL_VARIANTS")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    PIPELINE_ARGS+=("--dry-run")
fi

if [[ -n "$PUSHGATEWAY_URL" ]]; then
    PIPELINE_ARGS+=("--pushgateway-url" "$PUSHGATEWAY_URL")
fi

python3 -c "
import sys
import argparse
from benchmark.dag.pipeline import BenchmarkPipeline, PipelineConfig
from benchmark.config.schema import BenchmarkMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--execution-mode', default='local')
parser.add_argument('--namespace', default='benchmark')
parser.add_argument('--image', default='vllm/vllm-openai:latest')
parser.add_argument('--max-gpus', type=int, default=8)
parser.add_argument('--suites', default='vllm_parallelism')
parser.add_argument('--results-dir', default='./results')
parser.add_argument('--kubeconfig', default=None)
parser.add_argument('--model-variants', default=None)
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--pushgateway-url', default=None)
args = parser.parse_args()

suites = [s.strip() for s in args.suites.split(',')]
model_variants = (
    [v.strip() for v in args.model_variants.split(',')]
    if args.model_variants
    else None
)

config = PipelineConfig(
    max_gpus=args.max_gpus,
    suites=suites,
    model_variants=model_variants,
    execution_mode=args.execution_mode,
    namespace=args.namespace,
    image=args.image,
    kubeconfig=args.kubeconfig,
    results_dir=args.results_dir,
    dry_run=args.dry_run,
    enable_gpu_profiler=True,
    enable_trace_recorder=True,
    pushgateway_url=args.pushgateway_url,
)

pipeline = BenchmarkPipeline(config=config)
report = pipeline.run()
print(report)
" -- "${PIPELINE_ARGS[@]}"

# ---- Collect results ----
echo ""
echo "=== Results available in $RESULTS_DIR ==="
if [[ -d "$RESULTS_DIR" ]]; then
    ls -la "$RESULTS_DIR"/ 2>/dev/null || true
fi

# ---- Teardown (optional) ----
if [[ "$TEARDOWN" == "true" ]]; then
    echo ""
    echo "=== Tearing down Helm release '$RELEASE' ==="
    helm uninstall "$RELEASE" -n "$NAMESPACE"
    echo "  Done."
fi

echo ""
echo "=== Benchmark run complete ==="
