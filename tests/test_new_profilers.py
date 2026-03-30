"""Tests for all new profiler modules.

Covers:
  - MemoryProfiler
  - CommunicationProfiler
  - EnergyProfiler
  - TokenizerProfiler
  - AttentionProfiler
  - RequestLifecycleTracer
"""
import time
import pytest

# ---------------------------------------------------------------------------
# Memory Profiler
# ---------------------------------------------------------------------------
from benchmark.profiler.memory_profiler import (
    MemoryEvent,
    MemoryEventKind,
    MemoryPool,
    MemoryProfiler,
    MemorySnapshot,
    MemoryStats,
    TensorAllocation,
)


class TestMemoryProfiler:
    def test_basic_alloc_free(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=t, size_bytes=1024 * 1024, address=100,
            tensor_name="kv_block_0",
        ))
        snap = mp.take_snapshot(0)
        assert snap.total_allocated_bytes == 1024 * 1024
        assert snap.utilization_pct() > 0.0

        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.FREE, pool=MemoryPool.KV_CACHE,
            timestamp_us=t + 100, size_bytes=1024 * 1024, address=100,
        ))
        snap2 = mp.take_snapshot(0)
        assert snap2.total_allocated_bytes == 0

    def test_peak_tracking(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.ACTIVATIONS,
            timestamp_us=t, size_bytes=2 * 1024 * 1024,
        ))
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.FREE, pool=MemoryPool.ACTIVATIONS,
            timestamp_us=t + 50, size_bytes=2 * 1024 * 1024,
        ))
        stats = mp.compute_stats(0)
        assert stats.peak_allocated_bytes == 2 * 1024 * 1024

    def test_pool_breakdown(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.MODEL_WEIGHTS,
            timestamp_us=t, size_bytes=1000,
        ))
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=t + 10, size_bytes=500,
        ))
        breakdown = mp.pool_breakdown(0)
        assert breakdown[MemoryPool.MODEL_WEIGHTS.value] == 1000
        assert breakdown[MemoryPool.KV_CACHE.value] == 500

    def test_live_tensor_report(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.ACTIVATIONS,
            timestamp_us=t, size_bytes=4096, address=42,
            tensor_name="proj_weight",
        ))
        report = mp.live_tensor_report(0)
        assert len(report) == 1
        assert report[0]["tensor_name"] == "proj_weight"

    def test_swap_events(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=t, size_bytes=1000,
        ))
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.SWAP_OUT, pool=MemoryPool.KV_CACHE,
            timestamp_us=t + 10, size_bytes=500,
        ))
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.SWAP_IN, pool=MemoryPool.KV_CACHE,
            timestamp_us=t + 20, size_bytes=500,
        ))
        stats = mp.compute_stats(0)
        assert stats.total_swap_out_bytes == 500
        assert stats.total_swap_in_bytes == 500

    def test_oom_counting(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.OOM, pool=MemoryPool.KV_CACHE,
            timestamp_us=t, size_bytes=0,
        ))
        stats = mp.compute_stats(0)
        assert stats.total_oom_count == 1

    def test_export_timeline(self):
        mp = MemoryProfiler(device_indices=[0])
        t = time.time() * 1e6
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=t, size_bytes=1024,
        ))
        events = mp.export_timeline(0)
        assert len(events) == 1
        assert events[0]["cat"] == "memory"

    def test_export_snapshot_timeseries(self):
        mp = MemoryProfiler(device_indices=[0])
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=time.time() * 1e6, size_bytes=1024,
        ))
        mp.take_snapshot(0)
        ts = mp.export_snapshot_timeseries(0)
        assert len(ts) == 1
        assert "allocated_mib" in ts[0]

    def test_reset(self):
        mp = MemoryProfiler(device_indices=[0])
        mp.record_event(MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=time.time() * 1e6, size_bytes=1024,
        ))
        mp.reset()
        stats = mp.compute_stats(0)
        assert stats.num_events == 0

    def test_stats_summary(self):
        stats = MemoryStats(device_index=0, num_events=10, peak_allocated_bytes=1024 * 1024)
        summary = stats.summary()
        assert "Memory Stats" in summary
        assert "Peak allocated" in summary

    def test_snapshot_utilization(self):
        snap = MemorySnapshot(
            timestamp_us=0.0, total_allocated_bytes=500, total_free_bytes=500,
        )
        assert snap.utilization_pct() == 50.0

    def test_event_to_dict(self):
        e = MemoryEvent(
            kind=MemoryEventKind.ALLOC, pool=MemoryPool.KV_CACHE,
            timestamp_us=1000.0, size_bytes=512,
        )
        d = e.to_dict()
        assert d["kind"] == "alloc"
        assert d["pool"] == "kv_cache"


# ---------------------------------------------------------------------------
# Communication Profiler
# ---------------------------------------------------------------------------
from benchmark.profiler.communication_profiler import (
    CollectiveOp,
    CommBackend,
    CommEvent,
    CommProfile,
    CommTopology,
    CommunicationProfiler,
)


class TestCommunicationProfiler:
    def _make_event(self, op=CollectiveOp.ALL_REDUCE, dur=100.0, size=1024 * 1024):
        return CommEvent(
            op=op, start_us=1000.0, end_us=1000.0 + dur,
            message_size_bytes=size, world_size=8,
        )

    def test_basic_analysis(self):
        cp = CommunicationProfiler(world_size=8)
        cp.record(self._make_event())
        profile = cp.analyse(total_compute_time_us=1000.0)
        assert profile.total_comm_time_us > 0
        assert CollectiveOp.ALL_REDUCE.value in profile.per_op_stats

    def test_bandwidth_computation(self):
        e = self._make_event(dur=1000.0, size=1e9)  # 1GB in 1ms
        assert e.bandwidth_gbps > 0

    def test_algorithm_bandwidth(self):
        e = CommEvent(
            op=CollectiveOp.ALL_REDUCE, start_us=0.0, end_us=1000.0,
            message_size_bytes=int(1e9), world_size=8,
        )
        assert e.algorithm_bandwidth_gbps >= e.bandwidth_gbps

    def test_bottleneck_detection(self):
        cp = CommunicationProfiler(world_size=8)
        for _ in range(10):
            cp.record(CommEvent(
                op=CollectiveOp.ALL_REDUCE, start_us=0.0, end_us=500.0,
                message_size_bytes=1024, world_size=8,
            ))
        profile = cp.analyse(total_compute_time_us=100.0)
        assert len(profile.bottlenecks) > 0

    def test_per_phase_breakdown(self):
        cp = CommunicationProfiler()
        cp.record(CommEvent(
            op=CollectiveOp.ALL_REDUCE, start_us=0, end_us=100,
            message_size_bytes=1024, phase="prefill",
        ))
        cp.record(CommEvent(
            op=CollectiveOp.ALL_REDUCE, start_us=100, end_us=300,
            message_size_bytes=1024, phase="decode",
        ))
        breakdown = cp.per_phase_breakdown()
        assert "prefill" in breakdown
        assert "decode" in breakdown

    def test_export_timeline(self):
        cp = CommunicationProfiler()
        cp.record(self._make_event())
        timeline = cp.export_timeline()
        assert len(timeline) == 1
        assert timeline[0]["cat"] == "communication"

    def test_overlap_efficiency(self):
        cp = CommunicationProfiler()
        cp.record(CommEvent(
            op=CollectiveOp.ALL_REDUCE, start_us=0, end_us=100,
            message_size_bytes=1024, overlapped_with_compute=True,
        ))
        cp.record(CommEvent(
            op=CollectiveOp.ALL_REDUCE, start_us=100, end_us=200,
            message_size_bytes=1024, overlapped_with_compute=False,
        ))
        profile = cp.analyse()
        assert profile.overlap_efficiency == 0.5

    def test_reset(self):
        cp = CommunicationProfiler()
        cp.record(self._make_event())
        assert len(cp) == 1
        cp.reset()
        assert len(cp) == 0

    def test_event_to_dict(self):
        e = self._make_event()
        d = e.to_dict()
        assert "bandwidth_gbps" in d
        assert d["op"] == "all_reduce"

    def test_comm_stats_summary(self):
        from benchmark.profiler.communication_profiler import CommStats
        s = CommStats(op=CollectiveOp.ALL_REDUCE, count=5, total_time_us=500)
        assert "all_reduce" in s.summary()


# ---------------------------------------------------------------------------
# Energy Profiler
# ---------------------------------------------------------------------------
from benchmark.profiler.energy_profiler import (
    CarbonReport,
    EnergyProfiler,
    EnergyStats,
    PowerSample,
)


class TestEnergyProfiler:
    def test_basic_energy_computation(self):
        ep = EnergyProfiler(gpu_indices=[0], grid_region="us-west-2", pue=1.1)
        samples = [
            PowerSample(timestamp_s=0.0, gpu_index=0, power_w=300.0),
            PowerSample(timestamp_s=1.0, gpu_index=0, power_w=350.0),
            PowerSample(timestamp_s=2.0, gpu_index=0, power_w=300.0),
        ]
        for s in samples:
            ep.record_sample(s)
        stats = ep.compute_energy(0, total_tokens=1000, total_requests=10)
        assert stats.total_energy_j > 0
        assert stats.co2_grams > 0
        assert stats.energy_per_token_j > 0

    def test_carbon_report(self):
        ep = EnergyProfiler(gpu_indices=[0, 1])
        for gpu in [0, 1]:
            for t in range(3):
                ep.record_sample(PowerSample(
                    timestamp_s=float(t), gpu_index=gpu, power_w=250.0,
                ))
        report = ep.compute_report(total_tokens=5000, total_requests=50)
        assert report.total_energy_kwh > 0
        assert report.total_co2_grams >= 0
        assert len(report.per_gpu_stats) == 2

    def test_carbon_report_summary(self):
        report = CarbonReport(total_energy_kwh=0.001, total_co2_grams=0.5)
        assert "Carbon Footprint" in report.summary()

    def test_energy_stats_summary(self):
        stats = EnergyStats(gpu_index=0, total_energy_j=100.0, co2_grams=0.1)
        assert "Energy Stats" in stats.summary()

    def test_export_timeseries(self):
        ep = EnergyProfiler(gpu_indices=[0])
        ep.record_sample(PowerSample(timestamp_s=0.0, gpu_index=0, power_w=200.0))
        ts = ep.export_power_timeseries(0)
        assert len(ts) == 1
        assert ts[0]["power_w"] == 200.0

    def test_reset(self):
        ep = EnergyProfiler(gpu_indices=[0])
        ep.record_sample(PowerSample(timestamp_s=0.0, gpu_index=0, power_w=200.0))
        ep.reset()
        stats = ep.compute_energy(0)
        assert stats.num_samples == 0


# ---------------------------------------------------------------------------
# Tokenizer Profiler
# ---------------------------------------------------------------------------
from benchmark.profiler.tokenizer_profiler import (
    TokenizationEvent,
    TokenizerProfiler,
    TokenizerStats,
)


class TestTokenizerProfiler:
    def test_encode_decode_stats(self):
        tp = TokenizerProfiler(vocab_size=32000)
        tp.record(TokenizationEvent(
            request_id="r1", direction="encode",
            start_us=0.0, end_us=100.0, num_chars=500, num_tokens=100,
        ))
        tp.record(TokenizationEvent(
            request_id="r1", direction="decode",
            start_us=100.0, end_us=150.0, num_chars=200, num_tokens=40,
        ))
        stats = tp.compute_stats()
        assert stats.encode_count == 1
        assert stats.decode_count == 1
        assert stats.mean_encode_us == 100.0
        assert stats.mean_chars_per_token > 0

    def test_vocab_utilization(self):
        tp = TokenizerProfiler(vocab_size=100)
        tp.record_token_ids([1, 2, 3, 1, 2, 1])
        stats = tp.compute_stats()
        vu = stats.vocab_utilization
        assert vu is not None
        assert vu.unique_tokens_seen == 3
        assert vu.utilization_pct == 3.0
        assert len(vu.top_tokens) == 3

    def test_prompt_template_overhead(self):
        tp = TokenizerProfiler()
        tp.record_prompt_template(
            original_tokens=100, templated_tokens=130,
            template_name="chat_ml", latency_us=50.0,
        )
        stats = tp.compute_stats()
        assert stats.prompt_overhead is not None
        assert stats.prompt_overhead.overhead_tokens == 30

    def test_stats_summary(self):
        stats = TokenizerStats(encode_count=5, decode_count=3)
        assert "Tokenizer Profile" in stats.summary()

    def test_reset(self):
        tp = TokenizerProfiler()
        tp.record(TokenizationEvent(
            request_id="r1", direction="encode",
            start_us=0, end_us=10, num_chars=10, num_tokens=2,
        ))
        tp.reset()
        assert len(tp) == 0


# ---------------------------------------------------------------------------
# Attention Profiler
# ---------------------------------------------------------------------------
from benchmark.profiler.attention_profiler import (
    AttentionBackend,
    AttentionKernelEvent,
    AttentionPhase,
    AttentionProfile,
    AttentionProfiler,
    BoundClassification,
    SparsityAnalysis,
)


class TestAttentionProfiler:
    def test_basic_analysis(self):
        ap = AttentionProfiler(gpu_type="A100", num_layers=4)
        for layer in range(4):
            ap.record(AttentionKernelEvent(
                layer_index=layer, head_index=0,
                phase=AttentionPhase.FULL_ATTENTION,
                backend=AttentionBackend.FLASH_ATTN_V2,
                start_us=0.0, duration_us=100.0,
                flops=1e12, bytes_accessed=1e9, is_prefill=True,
            ))
        profile = ap.analyse()
        assert profile.total_attention_time_us == 400.0
        assert len(profile.per_layer_stats) == 4

    def test_sparsity_analysis(self):
        ap = AttentionProfiler()
        ap.record_sparsity(SparsityAnalysis(
            layer_index=0, head_index=0,
            sparsity_ratio=0.85, total_elements=1000, near_zero_elements=850,
        ))
        ap.record(AttentionKernelEvent(
            layer_index=0, head_index=0,
            phase=AttentionPhase.FULL_ATTENTION,
            backend=AttentionBackend.FLASH_ATTN_V2,
            start_us=0.0, duration_us=50.0,
            flops=1e12, bytes_accessed=1e9,
        ))
        profile = ap.analyse()
        assert profile.mean_sparsity == 0.85

    def test_prefill_vs_decode(self):
        ap = AttentionProfiler()
        ap.record(AttentionKernelEvent(
            layer_index=0, head_index=0,
            phase=AttentionPhase.FULL_ATTENTION,
            backend=AttentionBackend.FLASH_ATTN_V2,
            start_us=0, duration_us=200, is_prefill=True,
            flops=1e12, bytes_accessed=1e9,
        ))
        ap.record(AttentionKernelEvent(
            layer_index=0, head_index=0,
            phase=AttentionPhase.FULL_ATTENTION,
            backend=AttentionBackend.FLASH_ATTN_V2,
            start_us=200, duration_us=50, is_prefill=False,
            flops=1e12, bytes_accessed=1e9,
        ))
        profile = ap.analyse()
        assert profile.total_prefill_attn_us == 200
        assert profile.total_decode_attn_us == 50

    def test_roofline_export(self):
        ap = AttentionProfiler()
        ap.record(AttentionKernelEvent(
            layer_index=0, head_index=0,
            phase=AttentionPhase.FULL_ATTENTION,
            backend=AttentionBackend.FLASH_ATTN_V2,
            start_us=0, duration_us=100,
            flops=1e12, bytes_accessed=1e9,
        ))
        data = ap.export_roofline_data()
        assert len(data) == 1
        assert "arithmetic_intensity" in data[0]

    def test_kernel_event_properties(self):
        e = AttentionKernelEvent(
            layer_index=0, head_index=0,
            phase=AttentionPhase.FULL_ATTENTION,
            backend=AttentionBackend.FLASH_ATTN_V2,
            start_us=0, duration_us=1000,  # 1ms
            flops=1e12, bytes_accessed=1e9,
        )
        assert e.arithmetic_intensity == 1000.0
        assert e.tflops > 0
        assert e.bandwidth_gbps > 0

    def test_profile_summary(self):
        profile = AttentionProfile(total_attention_time_us=1000, backend="flash_attn_v2")
        assert "Attention Profile" in profile.summary()


# ---------------------------------------------------------------------------
# Request Lifecycle Tracer
# ---------------------------------------------------------------------------
from benchmark.profiler.request_lifecycle import (
    InterTokenTiming,
    LifecycleStats,
    PhaseTimestamp,
    RequestLifecycle,
    RequestLifecycleTracer,
    RequestPhase,
    TailLatencyCause,
)


class TestRequestLifecycleTracer:
    def _make_lifecycle(self, request_id="r1", queue_time=100, prefill_time=500, decode_time=2000):
        lc = RequestLifecycle(
            request_id=request_id,
            arrival_us=0.0,
            completion_us=float(queue_time + prefill_time + decode_time),
            input_tokens=100, output_tokens=50,
        )
        t = 0.0
        lc.phases.append(PhaseTimestamp(RequestPhase.QUEUE_WAIT, t, t + queue_time))
        t += queue_time
        lc.phases.append(PhaseTimestamp(RequestPhase.PREFILL, t, t + prefill_time))
        t += prefill_time
        lc.phases.append(PhaseTimestamp(RequestPhase.DECODE, t, t + decode_time))
        return lc

    def test_basic_stats(self):
        tracer = RequestLifecycleTracer()
        tracer.record(self._make_lifecycle("r1"))
        tracer.record(self._make_lifecycle("r2"))
        stats = tracer.compute_stats()
        assert stats.total_requests == 2
        assert stats.mean_e2e_us > 0

    def test_lifecycle_properties(self):
        lc = self._make_lifecycle()
        assert lc.e2e_latency_us == 2600.0
        assert lc.queue_time_us == 100.0
        assert lc.prefill_time_us == 500.0
        assert lc.decode_time_us == 2000.0

    def test_phase_breakdown(self):
        lc = self._make_lifecycle()
        breakdown = lc.phase_breakdown()
        assert RequestPhase.QUEUE_WAIT.value in breakdown
        assert RequestPhase.PREFILL.value in breakdown

    def test_gantt_export(self):
        tracer = RequestLifecycleTracer()
        tracer.record(self._make_lifecycle())
        gantt = tracer.export_gantt()
        assert len(gantt) == 3  # 3 phases

    def test_itl_jitter(self):
        lc = RequestLifecycle(request_id="r1", arrival_us=0, completion_us=1000)
        lc.inter_token_timings = [
            InterTokenTiming(0, 100, 50),
            InterTokenTiming(1, 150, 45),
            InterTokenTiming(2, 195, 55),
            InterTokenTiming(3, 250, 48),
        ]
        assert lc.mean_itl_us > 0
        assert lc.itl_jitter >= 0

    def test_tail_latency_requests(self):
        tracer = RequestLifecycleTracer()
        for i in range(100):
            lc = self._make_lifecycle(f"r{i}", decode_time=2000 + (i * 10))
            tracer.record(lc)
        tail = tracer.tail_latency_requests(percentile=99)
        assert len(tail) >= 1

    def test_itl_timeseries_export(self):
        tracer = RequestLifecycleTracer()
        lc = RequestLifecycle(request_id="r1", arrival_us=0, completion_us=500)
        lc.inter_token_timings = [InterTokenTiming(0, 100, 50)]
        tracer.record(lc)
        ts = tracer.export_itl_timeseries()
        assert len(ts) == 1

    def test_stats_summary(self):
        stats = LifecycleStats(total_requests=10, mean_e2e_us=5000)
        assert "Request Lifecycle" in stats.summary()

    def test_reset(self):
        tracer = RequestLifecycleTracer()
        tracer.record(self._make_lifecycle())
        tracer.reset()
        assert len(tracer) == 0
