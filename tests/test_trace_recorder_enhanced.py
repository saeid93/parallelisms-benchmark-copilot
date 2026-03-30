"""Tests for the enhanced trace recorder features (decode steps, layers, kernels, OTLP)."""

import json

import pytest

from benchmark.profiler.trace_recorder import (
    DecodeTokenStep,
    GPUKernelEvent,
    KVCacheEvent,
    KVCacheEventKind,
    LayerTiming,
    RequestTrace,
    TraceRecorder,
)


# ---------------------------------------------------------------------------
# DecodeTokenStep
# ---------------------------------------------------------------------------

class TestDecodeTokenStep:
    def test_duration(self):
        step = DecodeTokenStep(token_index=0, start_us=100.0, end_us=200.0)
        assert step.duration_us == 100.0

    def test_zero_duration(self):
        step = DecodeTokenStep(token_index=0, start_us=100.0, end_us=100.0)
        assert step.duration_us == 0.0


# ---------------------------------------------------------------------------
# LayerTiming
# ---------------------------------------------------------------------------

class TestLayerTiming:
    def test_duration(self):
        lt = LayerTiming(layer_index=0, phase="prefill", start_us=0.0, end_us=500.0)
        assert lt.duration_us == 500.0

    def test_phase_values(self):
        lt = LayerTiming(layer_index=3, phase="decode", start_us=100.0, end_us=200.0)
        assert lt.phase == "decode"
        assert lt.layer_index == 3


# ---------------------------------------------------------------------------
# GPUKernelEvent
# ---------------------------------------------------------------------------

class TestGPUKernelEvent:
    def test_defaults(self):
        ke = GPUKernelEvent(kernel_name="flash_attn_fwd", timestamp_us=1000.0)
        assert ke.duration_us == 0.0
        assert ke.stream_id == 0

    def test_with_duration(self):
        ke = GPUKernelEvent(kernel_name="gemm", timestamp_us=1000.0, duration_us=50.0, stream_id=2)
        assert ke.duration_us == 50.0
        assert ke.stream_id == 2


# ---------------------------------------------------------------------------
# RequestTrace — enhanced features
# ---------------------------------------------------------------------------

class TestRequestTraceEnhanced:
    def _make_trace_with_steps(self) -> RequestTrace:
        t = RequestTrace(
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
            output_tokens=3,
            decode_token_steps=[
                DecodeTokenStep(token_index=0, start_us=600.0, end_us=900.0),
                DecodeTokenStep(token_index=1, start_us=900.0, end_us=1200.0),
                DecodeTokenStep(token_index=2, start_us=1200.0, end_us=1600.0),
            ],
            layer_timings=[
                LayerTiming(layer_index=0, phase="prefill", start_us=100.0, end_us=350.0),
                LayerTiming(layer_index=1, phase="prefill", start_us=350.0, end_us=600.0),
            ],
            gpu_kernel_events=[
                GPUKernelEvent(kernel_name="flash_attn", timestamp_us=100.0, duration_us=200.0),
            ],
            kv_cache_events=[
                KVCacheEvent(kind=KVCacheEventKind.MISS, timestamp_us=100.0, request_id="r1", num_blocks=4),
            ],
        )
        return t

    def test_mean_tpot_us(self):
        t = self._make_trace_with_steps()
        # steps: 300, 300, 400 → mean = 333.33
        assert abs(t.mean_tpot_us - 333.33) < 1.0

    def test_mean_tpot_us_no_steps(self):
        t = RequestTrace(request_id="r0")
        assert t.mean_tpot_us == 0.0

    def test_chrome_spans_include_decode_tokens(self):
        t = self._make_trace_with_steps()
        spans = t.to_chrome_spans()
        names = [s["name"] for s in spans]
        assert "decode_token_0" in names
        assert "decode_token_1" in names
        assert "decode_token_2" in names

    def test_chrome_spans_include_layers(self):
        t = self._make_trace_with_steps()
        spans = t.to_chrome_spans()
        names = [s["name"] for s in spans]
        assert "layer_0_prefill" in names
        assert "layer_1_prefill" in names

    def test_chrome_spans_include_gpu_kernels(self):
        t = self._make_trace_with_steps()
        spans = t.to_chrome_spans()
        names = [s["name"] for s in spans]
        assert "flash_attn" in names

    def test_chrome_spans_include_kv_cache_events(self):
        t = self._make_trace_with_steps()
        spans = t.to_chrome_spans()
        names = [s["name"] for s in spans]
        assert "kv_miss" in names


# ---------------------------------------------------------------------------
# TraceRecorder — convenience methods
# ---------------------------------------------------------------------------

class TestTraceRecorderConvenience:
    def test_record_decode_token_step(self):
        rec = TraceRecorder(run_id="test")
        trace = rec.new_trace("r1")
        rec.record_decode_token_step("r1", 0, 100.0, 200.0)
        assert len(trace.decode_token_steps) == 1
        assert trace.decode_token_steps[0].token_index == 0

    def test_record_layer_timing(self):
        rec = TraceRecorder(run_id="test")
        trace = rec.new_trace("r1")
        rec.record_layer_timing("r1", 0, "prefill", 100.0, 300.0)
        assert len(trace.layer_timings) == 1
        assert trace.layer_timings[0].phase == "prefill"

    def test_record_gpu_kernel(self):
        rec = TraceRecorder(run_id="test")
        trace = rec.new_trace("r1")
        rec.record_gpu_kernel("r1", "gemm_fp16", 500.0, 50.0, stream_id=1)
        assert len(trace.gpu_kernel_events) == 1
        assert trace.gpu_kernel_events[0].kernel_name == "gemm_fp16"

    def test_record_kv_cache_event(self):
        rec = TraceRecorder(run_id="test")
        trace = rec.new_trace("r1")
        rec.record_kv_cache_event("r1", KVCacheEventKind.HIT, 200.0, num_blocks=8)
        assert len(trace.kv_cache_events) == 1
        assert trace.kv_cache_events[0].kind == KVCacheEventKind.HIT

    def test_record_on_nonexistent_request_is_noop(self):
        rec = TraceRecorder(run_id="test")
        rec.new_trace("r1")
        # Recording on a non-existent request should not raise
        rec.record_decode_token_step("r_nonexistent", 0, 100.0, 200.0)
        rec.record_layer_timing("r_nonexistent", 0, "prefill", 100.0, 200.0)
        rec.record_gpu_kernel("r_nonexistent", "k", 100.0)
        rec.record_kv_cache_event("r_nonexistent", KVCacheEventKind.MISS, 100.0)
        # Nothing should have been added to r1
        assert len(rec._traces[0].decode_token_steps) == 0


# ---------------------------------------------------------------------------
# TraceRecorder — OTLP export
# ---------------------------------------------------------------------------

class TestTraceRecorderOTLP:
    def test_export_otlp_spans_empty(self):
        rec = TraceRecorder()
        spans = rec.export_otlp_spans()
        assert spans == []

    def test_export_otlp_spans_basic(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            queue_start_us=0.0,
            queue_end_us=100.0,
            prefill_start_us=100.0,
            prefill_end_us=500.0,
            decode_start_us=500.0,
            decode_end_us=1000.0,
            completion_us=1000.0,
            input_tokens=64,
            output_tokens=32,
        )
        rec.record_request_trace(t)
        spans = rec.export_otlp_spans()

        # Should have root + queue + prefill + decode = 4 spans
        assert len(spans) == 4
        root = [s for s in spans if s["operationName"] == "request"]
        assert len(root) == 1
        assert root[0]["duration"] == 1000.0

    def test_export_otlp_spans_with_decode_steps(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=400.0,
            completion_us=400.0,
            decode_token_steps=[
                DecodeTokenStep(token_index=0, start_us=200.0, end_us=300.0),
                DecodeTokenStep(token_index=1, start_us=300.0, end_us=400.0),
            ],
        )
        rec.record_request_trace(t)
        spans = rec.export_otlp_spans()
        decode_tokens = [s for s in spans if s["operationName"].startswith("decode_token_")]
        assert len(decode_tokens) == 2

    def test_export_otlp_spans_with_layers(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=400.0,
            completion_us=400.0,
            layer_timings=[
                LayerTiming(layer_index=0, phase="prefill", start_us=0.0, end_us=100.0),
            ],
        )
        rec.record_request_trace(t)
        spans = rec.export_otlp_spans()
        layer_spans = [s for s in spans if "layer_0" in s["operationName"]]
        assert len(layer_spans) == 1

    def test_export_otlp_json(self, tmp_path):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=500.0,
            completion_us=500.0,
        )
        rec.record_request_trace(t)
        path = str(tmp_path / "otlp.json")
        rec.export_otlp_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "spans" in data
        assert len(data["spans"]) > 0

    def test_otlp_span_references(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            queue_start_us=0.0,
            queue_end_us=50.0,
            prefill_start_us=50.0,
            prefill_end_us=200.0,
            decode_start_us=200.0,
            decode_end_us=400.0,
            completion_us=400.0,
        )
        rec.record_request_trace(t)
        spans = rec.export_otlp_spans()
        root = [s for s in spans if s["operationName"] == "request"][0]
        assert root["references"] == []
        children = [s for s in spans if s["operationName"] != "request"]
        for child in children:
            assert len(child["references"]) == 1
            assert child["references"][0]["refType"] == "CHILD_OF"


# ---------------------------------------------------------------------------
# TraceRecorder — enhanced summary_stats
# ---------------------------------------------------------------------------

class TestTraceRecorderEnhancedStats:
    def test_summary_includes_tpot(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200_000.0,
            decode_start_us=200_000.0,
            decode_end_us=400_000.0,
            completion_us=400_000.0,
            decode_token_steps=[
                DecodeTokenStep(0, 200_000.0, 300_000.0),
                DecodeTokenStep(1, 300_000.0, 400_000.0),
            ],
        )
        rec.record_request_trace(t)
        stats = rec.summary_stats()
        assert "mean_tpot_ms" in stats
        assert stats["mean_tpot_ms"] == 100.0  # 100_000 us = 100 ms

    def test_summary_includes_layer_stats(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200_000.0,
            decode_start_us=200_000.0,
            decode_end_us=400_000.0,
            completion_us=400_000.0,
            layer_timings=[
                LayerTiming(0, "prefill", 0.0, 100_000.0),
                LayerTiming(1, "prefill", 100_000.0, 200_000.0),
            ],
        )
        rec.record_request_trace(t)
        stats = rec.summary_stats()
        assert stats["num_layer_events"] == 2.0
        assert abs(stats["mean_layer_time_ms"] - 100.0) < 0.1

    def test_summary_includes_kernel_stats(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200_000.0,
            decode_start_us=200_000.0,
            decode_end_us=400_000.0,
            completion_us=400_000.0,
            gpu_kernel_events=[
                GPUKernelEvent("gemm", 0.0, 50_000.0),
            ],
        )
        rec.record_request_trace(t)
        stats = rec.summary_stats()
        assert stats["num_kernel_events"] == 1.0
        assert stats["mean_kernel_time_ms"] == 50.0

    def test_summary_includes_kv_count(self):
        rec = TraceRecorder()
        t = RequestTrace(
            request_id="r0",
            arrival_us=0.0,
            prefill_start_us=0.0,
            prefill_end_us=200_000.0,
            completion_us=400_000.0,
            kv_cache_events=[
                KVCacheEvent(KVCacheEventKind.HIT, 100.0, "r0"),
                KVCacheEvent(KVCacheEventKind.MISS, 200.0, "r0"),
            ],
        )
        rec.record_request_trace(t)
        stats = rec.summary_stats()
        assert stats["total_kv_cache_events"] == 2.0
