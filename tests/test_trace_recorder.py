"""Tests for the trace recorder (profiler)."""

import json

import pytest

from benchmark.profiler.trace_recorder import (
    KVCacheEvent,
    KVCacheEventKind,
    RequestTrace,
    TraceEventKind,
    TraceRecorder,
    TraceSpan,
    _to_folded_stack_line,
)


class TestTraceSpan:
    def test_to_chrome_event_complete(self):
        span = TraceSpan(
            name="prefill",
            category="compute",
            kind=TraceEventKind.COMPLETE,
            timestamp_us=1000.0,
            duration_us=500.0,
            request_id="req-1",
            thread_id=42,
            args={"tokens": 256},
        )
        event = span.to_chrome_event()
        assert event["name"] == "prefill"
        assert event["cat"] == "compute"
        assert event["ph"] == "X"
        assert event["ts"] == 1000.0
        assert event["dur"] == 500.0
        assert event["args"]["tokens"] == 256

    def test_to_chrome_event_instant_no_dur(self):
        span = TraceSpan(
            name="ev",
            category="scheduler",
            kind=TraceEventKind.INSTANT,
            timestamp_us=2000.0,
        )
        event = span.to_chrome_event()
        assert "dur" not in event


class TestRequestTrace:
    def _make_trace(self) -> RequestTrace:
        return RequestTrace(
            request_id="r1",
            arrival_us=0.0,
            queue_start_us=0.0,
            queue_end_us=100.0,
            prefill_start_us=100.0,
            prefill_end_us=600.0,
            decode_start_us=600.0,
            decode_end_us=1600.0,
            completion_us=1600.0,
            input_tokens=128,
            output_tokens=64,
        )

    def test_timing_properties(self):
        t = self._make_trace()
        assert t.queue_time_us == 100.0
        assert t.prefill_time_us == 500.0
        assert t.decode_time_us == 1000.0

    def test_ttft_us(self):
        t = self._make_trace()
        # TTFT = prefill_end - arrival = 600 - 0 = 600
        assert t.ttft_us == 600.0

    def test_e2e_latency_us(self):
        t = self._make_trace()
        assert t.e2e_latency_us == 1600.0

    def test_to_chrome_spans_returns_list(self):
        t = self._make_trace()
        spans = t.to_chrome_spans()
        assert isinstance(spans, list)
        assert len(spans) > 0

    def test_to_chrome_spans_all_dicts(self):
        t = self._make_trace()
        for span in t.to_chrome_spans():
            assert isinstance(span, dict)
            assert "name" in span
            assert "ph" in span

    def test_zero_duration_spans_not_emitted(self):
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            queue_start_us=0.0,
            queue_end_us=0.0,  # zero duration
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=400.0,
            completion_us=400.0,
        )
        spans = t.to_chrome_spans()
        names = [s["name"] for s in spans]
        assert "queue" not in names
        assert "prefill" in names


class TestFoldedStackLine:
    def test_has_phases(self):
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            queue_start_us=0.0,
            queue_end_us=50.0,
            prefill_start_us=50.0,
            prefill_end_us=300.0,
            decode_start_us=300.0,
            decode_end_us=800.0,
            completion_us=800.0,
        )
        line = _to_folded_stack_line(t)
        assert "root" in line
        assert "prefill" in line
        assert "decode" in line
        # Value = e2e_latency_us = 800
        assert "800" in line

    def test_format(self):
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=500.0,
            decode_start_us=500.0,
            decode_end_us=1000.0,
            completion_us=1000.0,
        )
        line = _to_folded_stack_line(t)
        parts = line.rsplit(" ", 1)
        assert len(parts) == 2
        assert int(parts[1]) >= 0


class TestTraceRecorder:
    def test_new_trace_added(self):
        recorder = TraceRecorder(run_id="test-run")
        t = recorder.new_trace("req-1", input_tokens=128, output_tokens=64)
        assert len(recorder) == 1
        assert t.request_id == "req-1"
        assert t.input_tokens == 128

    def test_record_request_trace(self):
        recorder = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=100.0,
            prefill_end_us=500.0,
            decode_start_us=500.0,
            decode_end_us=900.0,
            completion_us=900.0,
        )
        recorder.record_request_trace(t)
        assert len(recorder) == 1

    def test_add_global_span(self):
        recorder = TraceRecorder()
        recorder.add_global_span("batch_prefill", "compute", 1000.0, 200.0)
        trace = recorder.export_chrome_trace()
        events = trace["traceEvents"]
        global_ev = [e for e in events if e.get("name") == "batch_prefill"]
        assert len(global_ev) == 1

    def test_export_chrome_trace_structure(self):
        recorder = TraceRecorder(run_id="abc")
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=100.0,
            prefill_end_us=300.0,
            decode_start_us=300.0,
            decode_end_us=600.0,
            completion_us=600.0,
        )
        recorder.record_request_trace(t)
        trace = recorder.export_chrome_trace()
        assert "traceEvents" in trace
        assert len(trace["traceEvents"]) > 0

    def test_export_chrome_trace_json_file(self, tmp_path):
        recorder = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=500.0,
            completion_us=500.0,
        )
        recorder.record_request_trace(t)
        path = str(tmp_path / "trace.json")
        recorder.export_chrome_trace_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "traceEvents" in data

    def test_export_folded_stacks(self):
        recorder = TraceRecorder()
        for i in range(3):
            t = RequestTrace(
                request_id=f"r{i}",
                arrival_us=float(i * 1000),
                prefill_start_us=float(i * 1000 + 100),
                prefill_end_us=float(i * 1000 + 400),
                decode_start_us=float(i * 1000 + 400),
                decode_end_us=float(i * 1000 + 900),
                completion_us=float(i * 1000 + 900),
            )
            recorder.record_request_trace(t)
        output = recorder.export_folded_stacks()
        lines = [l for l in output.strip().split("\n") if l]
        assert len(lines) == 3
        for line in lines:
            assert "root" in line

    def test_export_folded_stacks_file(self, tmp_path):
        recorder = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=500.0,
            completion_us=500.0,
        )
        recorder.record_request_trace(t)
        path = str(tmp_path / "stacks.txt")
        recorder.export_folded_stacks_file(path)
        with open(path) as f:
            content = f.read()
        assert "root" in content

    def test_summary_stats(self):
        recorder = TraceRecorder()
        for i in range(5):
            t = RequestTrace(
                request_id=f"r{i}",
                arrival_us=0.0,
                prefill_start_us=0.0,
                prefill_end_us=200_000.0,   # 200 ms
                decode_start_us=200_000.0,
                decode_end_us=400_000.0,    # 200 ms
                completion_us=400_000.0,
            )
            recorder.record_request_trace(t)
        stats = recorder.summary_stats()
        assert stats["num_requests"] == 5.0
        assert abs(stats["mean_prefill_ms"] - 200.0) < 1.0
        assert abs(stats["mean_decode_ms"] - 200.0) < 1.0

    def test_summary_stats_empty(self):
        recorder = TraceRecorder()
        assert recorder.summary_stats() == {}

    def test_reset(self):
        recorder = TraceRecorder()
        recorder.new_trace("r0")
        recorder.reset()
        assert len(recorder) == 0
        assert recorder.summary_stats() == {}
