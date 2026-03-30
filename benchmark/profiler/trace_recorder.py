"""
Trace recorder — profiler stage.

Records per-request execution traces at sub-millisecond granularity.
Traces include:
  - Request lifecycle: arrival → queue → prefill → decode → completion
  - Per-layer timing (when available from instrumented vLLM)
  - Decode token iteration timing (per output token step)
  - KV cache events: hit, miss, eviction, swap
  - GPU kernel launch timestamps

The recorder emits traces in a format compatible with:
  - Chrome DevTools / Perfetto (JSON trace format)
  - OpenTelemetry spans (via OTLP export)
  - Flame graph rendering (folded-stacks format)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Trace event kinds
# ---------------------------------------------------------------------------

class TraceEventKind(str, Enum):
    """Phase event types following Chrome Trace Event Format (TEF)."""

    DURATION_BEGIN = "B"   # Begin of a duration event
    DURATION_END = "E"     # End of a duration event
    COMPLETE = "X"         # Complete event with duration
    INSTANT = "i"          # Instant event (no duration)
    COUNTER = "C"          # Counter/metric value
    ASYNC_BEGIN = "b"      # Async event begin
    ASYNC_END = "e"        # Async event end
    METADATA = "M"         # Metadata


# ---------------------------------------------------------------------------
# Trace span dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceSpan:
    """A single timing span in the trace."""

    name: str
    category: str
    kind: TraceEventKind
    timestamp_us: float          # Microseconds since epoch
    duration_us: float = 0.0     # Duration (for COMPLETE events)
    request_id: Optional[str] = None
    thread_id: int = 0
    process_id: int = 0
    args: Dict = field(default_factory=dict)

    def to_chrome_event(self) -> Dict:
        """Serialise to Chrome Trace Event Format dict."""
        event: Dict = {
            "name": self.name,
            "cat": self.category,
            "ph": self.kind.value,
            "ts": self.timestamp_us,
            "pid": self.process_id,
            "tid": self.thread_id,
        }
        if self.kind == TraceEventKind.COMPLETE:
            event["dur"] = self.duration_us
        if self.args:
            event["args"] = self.args
        return event


# ---------------------------------------------------------------------------
# KV cache event
# ---------------------------------------------------------------------------

class KVCacheEventKind(str, Enum):
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    SWAP_IN = "swap_in"
    SWAP_OUT = "swap_out"


@dataclass
class KVCacheEvent:
    """A KV cache lifecycle event."""

    kind: KVCacheEventKind
    timestamp_us: float
    request_id: str
    num_blocks: int = 0
    layer: Optional[int] = None


# ---------------------------------------------------------------------------
# Decode token iteration timing
# ---------------------------------------------------------------------------

@dataclass
class DecodeTokenStep:
    """Timing for a single decode output token iteration."""

    token_index: int
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


# ---------------------------------------------------------------------------
# Layer timing
# ---------------------------------------------------------------------------

@dataclass
class LayerTiming:
    """Timing for a single model layer execution."""

    layer_index: int
    phase: str           # "prefill" or "decode"
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


# ---------------------------------------------------------------------------
# GPU kernel event
# ---------------------------------------------------------------------------

@dataclass
class GPUKernelEvent:
    """A GPU kernel launch/completion event."""

    kernel_name: str
    timestamp_us: float
    duration_us: float = 0.0
    stream_id: int = 0


# ---------------------------------------------------------------------------
# Per-request trace
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    """Complete trace for a single request."""

    request_id: str
    arrival_us: float = 0.0
    queue_start_us: float = 0.0
    queue_end_us: float = 0.0
    prefill_start_us: float = 0.0
    prefill_end_us: float = 0.0
    decode_start_us: float = 0.0
    decode_end_us: float = 0.0
    completion_us: float = 0.0

    kv_cache_events: List[KVCacheEvent] = field(default_factory=list)
    extra_spans: List[TraceSpan] = field(default_factory=list)
    decode_token_steps: List[DecodeTokenStep] = field(default_factory=list)
    layer_timings: List[LayerTiming] = field(default_factory=list)
    gpu_kernel_events: List[GPUKernelEvent] = field(default_factory=list)

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def queue_time_us(self) -> float:
        return max(0.0, self.queue_end_us - self.queue_start_us)

    @property
    def prefill_time_us(self) -> float:
        return max(0.0, self.prefill_end_us - self.prefill_start_us)

    @property
    def decode_time_us(self) -> float:
        return max(0.0, self.decode_end_us - self.decode_start_us)

    @property
    def ttft_us(self) -> float:
        """Time to first token (arrival → first token)."""
        return max(0.0, self.prefill_end_us - self.arrival_us)

    @property
    def e2e_latency_us(self) -> float:
        return max(0.0, self.completion_us - self.arrival_us)

    @property
    def mean_tpot_us(self) -> float:
        """Mean time per output token from decode token steps."""
        if not self.decode_token_steps:
            return 0.0
        total = sum(s.duration_us for s in self.decode_token_steps)
        return total / len(self.decode_token_steps)

    def to_chrome_spans(self) -> List[Dict]:
        """Convert this trace to a list of Chrome TEF events."""
        spans: List[Dict] = []
        tid = hash(self.request_id) % 1000

        def _add(name: str, start_us: float, end_us: float, cat: str = "request") -> None:
            if end_us > start_us:
                spans.append(
                    TraceSpan(
                        name=name,
                        category=cat,
                        kind=TraceEventKind.COMPLETE,
                        timestamp_us=start_us,
                        duration_us=end_us - start_us,
                        request_id=self.request_id,
                        thread_id=tid,
                        args={
                            "request_id": self.request_id,
                            "input_tokens": self.input_tokens,
                        },
                    ).to_chrome_event()
                )

        _add("queue", self.queue_start_us, self.queue_end_us, "scheduler")
        _add("prefill", self.prefill_start_us, self.prefill_end_us, "compute")
        _add("decode", self.decode_start_us, self.decode_end_us, "compute")

        # Decode token iteration spans
        for step in self.decode_token_steps:
            _add(
                f"decode_token_{step.token_index}",
                step.start_us,
                step.end_us,
                "decode_iter",
            )

        # Layer timing spans
        for lt in self.layer_timings:
            _add(
                f"layer_{lt.layer_index}_{lt.phase}",
                lt.start_us,
                lt.end_us,
                "layer",
            )

        # GPU kernel spans
        for ke in self.gpu_kernel_events:
            if ke.duration_us > 0:
                spans.append(
                    TraceSpan(
                        name=ke.kernel_name,
                        category="gpu_kernel",
                        kind=TraceEventKind.COMPLETE,
                        timestamp_us=ke.timestamp_us,
                        duration_us=ke.duration_us,
                        request_id=self.request_id,
                        thread_id=tid + ke.stream_id,
                        args={"stream_id": ke.stream_id},
                    ).to_chrome_event()
                )

        # KV cache instant events
        for kv_ev in self.kv_cache_events:
            spans.append(
                TraceSpan(
                    name=f"kv_{kv_ev.kind.value}",
                    category="kv_cache",
                    kind=TraceEventKind.INSTANT,
                    timestamp_us=kv_ev.timestamp_us,
                    request_id=self.request_id,
                    thread_id=tid,
                    args={
                        "num_blocks": kv_ev.num_blocks,
                        "layer": kv_ev.layer,
                    },
                ).to_chrome_event()
            )

        for span in self.extra_spans:
            spans.append(span.to_chrome_event())

        return spans


# ---------------------------------------------------------------------------
# Flame graph helpers
# ---------------------------------------------------------------------------

def _to_folded_stack_line(trace: RequestTrace) -> str:
    """Emit one folded-stack line for flame graph rendering.

    Format: ``root;queue;prefill;decode <duration_us>``
    """
    phases = []
    if trace.queue_time_us > 0:
        phases.append("queue")
    if trace.prefill_time_us > 0:
        phases.append("prefill")
    if trace.decode_time_us > 0:
        phases.append("decode")
    stack = ";".join(["root"] + phases) if phases else "root"
    total_us = trace.e2e_latency_us
    return f"{stack} {int(total_us)}"


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------

class TraceRecorder:
    """Records and manages execution traces for benchmark runs.

    Args:
        run_id: Unique identifier for the current benchmark run.
        enable_kv_events: Whether to record KV cache events.
    """

    def __init__(
        self,
        run_id: str = "",
        enable_kv_events: bool = True,
    ) -> None:
        self.run_id = run_id
        self.enable_kv_events = enable_kv_events
        self._traces: List[RequestTrace] = []
        self._global_spans: List[TraceSpan] = []
        self._start_real_us = time.time() * 1_000_000.0

    # ------------------------------------------------------------------
    # Request trace recording
    # ------------------------------------------------------------------

    def record_request_trace(self, trace: RequestTrace) -> None:
        """Record a completed request trace.

        Args:
            trace: Fully populated RequestTrace for the completed request.
        """
        self._traces.append(trace)

    def new_trace(
        self,
        request_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> RequestTrace:
        """Create and register a new trace, anchored to current wall time.

        Args:
            request_id: Unique request identifier.
            input_tokens: Input sequence length in tokens.
            output_tokens: Expected output length in tokens.

        Returns:
            A fresh RequestTrace with arrival_us set to now.
        """
        now_us = time.time() * 1_000_000.0
        trace = RequestTrace(
            request_id=request_id,
            arrival_us=now_us,
            queue_start_us=now_us,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._traces.append(trace)
        return trace

    # ------------------------------------------------------------------
    # Global spans (pipeline-level events)
    # ------------------------------------------------------------------

    def add_global_span(
        self,
        name: str,
        category: str,
        start_us: float,
        duration_us: float,
        args: Optional[Dict] = None,
    ) -> None:
        """Record a pipeline-level span (e.g., "batch_prefill").

        Args:
            name: Span name.
            category: Category string (e.g., "scheduler", "compute").
            start_us: Start time in microseconds.
            duration_us: Duration in microseconds.
            args: Optional extra attributes.
        """
        self._global_spans.append(
            TraceSpan(
                name=name,
                category=category,
                kind=TraceEventKind.COMPLETE,
                timestamp_us=start_us,
                duration_us=duration_us,
                thread_id=0,
                process_id=0,
                args=args or {},
            )
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_chrome_trace(self) -> Dict:
        """Export all traces in Chrome Trace Event Format (Perfetto-compatible).

        Returns:
            Dict with ``traceEvents`` key, suitable for JSON serialisation.
        """
        events: List[Dict] = []

        # Metadata
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": 0,
                "tid": 0,
                "args": {"name": f"benchmark-run-{self.run_id}"},
            }
        )

        # Global spans
        for span in self._global_spans:
            events.append(span.to_chrome_event())

        # Request spans
        for trace in self._traces:
            events.extend(trace.to_chrome_spans())

        return {"traceEvents": events}

    def export_chrome_trace_json(self, path: str) -> None:
        """Write Chrome trace JSON to *path*.

        Args:
            path: Output file path (conventionally ending in ``.json``).
        """
        trace = self.export_chrome_trace()
        with open(path, "w") as fh:
            json.dump(trace, fh)

    def export_folded_stacks(self) -> str:
        """Export traces in folded-stacks format for flame graph rendering.

        Returns:
            Multi-line string, one line per request.
        """
        lines = [_to_folded_stack_line(t) for t in self._traces]
        return "\n".join(lines)

    def export_folded_stacks_file(self, path: str) -> None:
        """Write folded-stacks output to *path*.

        Args:
            path: Output file path.
        """
        with open(path, "w") as fh:
            fh.write(self.export_folded_stacks())

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics over all recorded traces.

        Returns:
            Dict mapping stat name to value.
        """
        if not self._traces:
            return {}

        n = len(self._traces)
        ttft_vals = [t.ttft_us / 1000.0 for t in self._traces]  # → ms
        e2e_vals = [t.e2e_latency_us / 1000.0 for t in self._traces]
        prefill_vals = [t.prefill_time_us / 1000.0 for t in self._traces]
        decode_vals = [t.decode_time_us / 1000.0 for t in self._traces]
        tpot_vals = [t.mean_tpot_us / 1000.0 for t in self._traces if t.mean_tpot_us > 0]

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def _pct(lst: List[float], p: float) -> float:
            s = sorted(lst)
            idx = int(p / 100.0 * (len(s) - 1))
            return s[idx]

        stats: Dict[str, float] = {
            "num_requests": float(n),
            "mean_ttft_ms": _mean(ttft_vals),
            "p50_ttft_ms": _pct(ttft_vals, 50),
            "p90_ttft_ms": _pct(ttft_vals, 90),
            "p99_ttft_ms": _pct(ttft_vals, 99),
            "mean_e2e_ms": _mean(e2e_vals),
            "p90_e2e_ms": _pct(e2e_vals, 90),
            "mean_prefill_ms": _mean(prefill_vals),
            "mean_decode_ms": _mean(decode_vals),
        }

        if tpot_vals:
            stats["mean_tpot_ms"] = _mean(tpot_vals)
            stats["p90_tpot_ms"] = _pct(tpot_vals, 90)

        # Layer timing stats
        all_layer_timings = [
            lt for t in self._traces for lt in t.layer_timings
        ]
        if all_layer_timings:
            stats["mean_layer_time_ms"] = _mean(
                [lt.duration_us / 1000.0 for lt in all_layer_timings]
            )
            stats["num_layer_events"] = float(len(all_layer_timings))

        # GPU kernel stats
        all_kernels = [
            ke for t in self._traces for ke in t.gpu_kernel_events
        ]
        if all_kernels:
            stats["mean_kernel_time_ms"] = _mean(
                [ke.duration_us / 1000.0 for ke in all_kernels]
            )
            stats["num_kernel_events"] = float(len(all_kernels))

        # KV cache event counts
        total_kv = sum(len(t.kv_cache_events) for t in self._traces)
        if total_kv > 0:
            stats["total_kv_cache_events"] = float(total_kv)

        return stats

    def reset(self) -> None:
        """Clear all recorded traces and spans."""
        self._traces.clear()
        self._global_spans.clear()

    def __len__(self) -> int:
        return len(self._traces)

    # ------------------------------------------------------------------
    # Convenience methods for recording sub-request events
    # ------------------------------------------------------------------

    def record_decode_token_step(
        self,
        request_id: str,
        token_index: int,
        start_us: float,
        end_us: float,
    ) -> None:
        """Record a decode token iteration step on an existing trace.

        Args:
            request_id: Request ID to attach the step to.
            token_index: Zero-based index of the output token.
            start_us: Step start time in microseconds.
            end_us: Step end time in microseconds.
        """
        step = DecodeTokenStep(
            token_index=token_index,
            start_us=start_us,
            end_us=end_us,
        )
        for trace in reversed(self._traces):
            if trace.request_id == request_id:
                trace.decode_token_steps.append(step)
                return

    def record_layer_timing(
        self,
        request_id: str,
        layer_index: int,
        phase: str,
        start_us: float,
        end_us: float,
    ) -> None:
        """Record a model layer execution timing on an existing trace.

        Args:
            request_id: Request ID to attach the timing to.
            layer_index: Zero-based layer index.
            phase: ``"prefill"`` or ``"decode"``.
            start_us: Layer start time in microseconds.
            end_us: Layer end time in microseconds.
        """
        lt = LayerTiming(
            layer_index=layer_index,
            phase=phase,
            start_us=start_us,
            end_us=end_us,
        )
        for trace in reversed(self._traces):
            if trace.request_id == request_id:
                trace.layer_timings.append(lt)
                return

    def record_gpu_kernel(
        self,
        request_id: str,
        kernel_name: str,
        timestamp_us: float,
        duration_us: float = 0.0,
        stream_id: int = 0,
    ) -> None:
        """Record a GPU kernel event on an existing trace.

        Args:
            request_id: Request ID to attach the kernel to.
            kernel_name: Name of the GPU kernel.
            timestamp_us: Kernel launch timestamp in microseconds.
            duration_us: Kernel duration in microseconds.
            stream_id: CUDA stream ID.
        """
        ke = GPUKernelEvent(
            kernel_name=kernel_name,
            timestamp_us=timestamp_us,
            duration_us=duration_us,
            stream_id=stream_id,
        )
        for trace in reversed(self._traces):
            if trace.request_id == request_id:
                trace.gpu_kernel_events.append(ke)
                return

    def record_kv_cache_event(
        self,
        request_id: str,
        kind: KVCacheEventKind,
        timestamp_us: float,
        num_blocks: int = 0,
        layer: Optional[int] = None,
    ) -> None:
        """Record a KV cache event on an existing trace.

        Args:
            request_id: Request ID to attach the event to.
            kind: Type of KV cache event.
            timestamp_us: Event timestamp in microseconds.
            num_blocks: Number of KV cache blocks affected.
            layer: Optional layer index.
        """
        ev = KVCacheEvent(
            kind=kind,
            timestamp_us=timestamp_us,
            request_id=request_id,
            num_blocks=num_blocks,
            layer=layer,
        )
        for trace in reversed(self._traces):
            if trace.request_id == request_id:
                trace.kv_cache_events.append(ev)
                return

    # ------------------------------------------------------------------
    # OpenTelemetry-style span export
    # ------------------------------------------------------------------

    def export_otlp_spans(self) -> List[Dict]:
        """Export all traces as OpenTelemetry-compatible span dicts.

        Each span has ``traceId``, ``spanId``, ``operationName``,
        ``startTime``, ``duration``, ``tags``, and ``references``.
        This is suitable for OTLP JSON export or ingestion by Jaeger/Zipkin.

        Returns:
            List of span dicts.
        """
        spans: List[Dict] = []
        for trace in self._traces:
            trace_id = trace.request_id
            parent_span_id = f"{trace_id}-root"

            # Root span (full request lifecycle)
            spans.append({
                "traceId": trace_id,
                "spanId": parent_span_id,
                "operationName": "request",
                "startTime": trace.arrival_us,
                "duration": trace.e2e_latency_us,
                "tags": {
                    "input_tokens": trace.input_tokens,
                    "output_tokens": trace.output_tokens,
                },
                "references": [],
            })

            # Phase spans
            for phase_name, start, end in [
                ("queue", trace.queue_start_us, trace.queue_end_us),
                ("prefill", trace.prefill_start_us, trace.prefill_end_us),
                ("decode", trace.decode_start_us, trace.decode_end_us),
            ]:
                dur = max(0.0, end - start)
                if dur > 0:
                    spans.append({
                        "traceId": trace_id,
                        "spanId": f"{trace_id}-{phase_name}",
                        "operationName": phase_name,
                        "startTime": start,
                        "duration": dur,
                        "tags": {},
                        "references": [{"refType": "CHILD_OF", "spanId": parent_span_id}],
                    })

            # Decode token steps as child spans
            for step in trace.decode_token_steps:
                spans.append({
                    "traceId": trace_id,
                    "spanId": f"{trace_id}-decode-token-{step.token_index}",
                    "operationName": f"decode_token_{step.token_index}",
                    "startTime": step.start_us,
                    "duration": step.duration_us,
                    "tags": {"token_index": step.token_index},
                    "references": [{"refType": "CHILD_OF", "spanId": f"{trace_id}-decode"}],
                })

            # Layer timings
            for lt in trace.layer_timings:
                spans.append({
                    "traceId": trace_id,
                    "spanId": f"{trace_id}-layer-{lt.layer_index}-{lt.phase}",
                    "operationName": f"layer_{lt.layer_index}_{lt.phase}",
                    "startTime": lt.start_us,
                    "duration": lt.duration_us,
                    "tags": {"layer_index": lt.layer_index, "phase": lt.phase},
                    "references": [{"refType": "CHILD_OF", "spanId": f"{trace_id}-{lt.phase}"}],
                })

        return spans

    def export_otlp_json(self, path: str) -> None:
        """Write OpenTelemetry spans to a JSON file.

        Args:
            path: Output file path.
        """
        spans = self.export_otlp_spans()
        with open(path, "w") as fh:
            json.dump({"spans": spans}, fh, indent=2)
