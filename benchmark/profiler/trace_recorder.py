"""
Trace recorder — profiler stage.

Records per-request execution traces at sub-millisecond granularity.
Traces include:
  - Request lifecycle: arrival → queue → prefill → decode → completion
  - Per-layer timing (when available from instrumented vLLM)
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

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def _pct(lst: List[float], p: float) -> float:
            s = sorted(lst)
            idx = int(p / 100.0 * (len(s) - 1))
            return s[idx]

        return {
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

    def reset(self) -> None:
        """Clear all recorded traces and spans."""
        self._traces.clear()
        self._global_spans.clear()

    def __len__(self) -> int:
        return len(self._traces)
