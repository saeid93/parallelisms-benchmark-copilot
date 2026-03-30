"""Tests for all new analysis modules.

Covers:
  - SchedulerAnalyzer
  - RooflineAnalyzer
  - ScalabilityAnalyzer
  - PipelineBubbleAnalyzer
  - QuantizationAnalyzer
  - KVCacheAnalyzer
  - SpeculativeAnalyzer
  - AnomalyDetector
  - WorkloadCharacterizer
"""
import pytest

# ---------------------------------------------------------------------------
# Scheduler Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.scheduler_analyzer import (
    PreemptionReason,
    RequestSchedulingInfo,
    SchedulerAnalyzer,
    SchedulerStats,
    SchedulingDecision,
)


class TestSchedulerAnalyzer:
    def test_basic_analysis(self):
        sa = SchedulerAnalyzer(max_batch_tokens=4096)
        for i in range(10):
            sa.record_decision(SchedulingDecision(
                step_index=i, timestamp_us=float(i * 1000),
                batch_size=8, num_prefill_requests=2, num_decode_requests=6,
                total_tokens_in_batch=2048, queue_depth=5,
                scheduling_latency_us=50.0,
            ))
        stats = sa.analyse()
        assert stats.total_steps == 10
        assert stats.mean_batch_size == 8.0
        assert stats.batch_utilization_pct == 50.0  # 2048/4096

    def test_fairness_metrics(self):
        sa = SchedulerAnalyzer()
        sa.record_decision(SchedulingDecision(step_index=0, timestamp_us=0.0, batch_size=4))
        for i in range(5):
            sa.record_request(RequestSchedulingInfo(
                request_id=f"r{i}", total_wait_us=float(100 + i * 10),
            ))
        stats = sa.analyse()
        assert stats.fairness.jains_fairness_index > 0.0

    def test_preemption_tracking(self):
        sa = SchedulerAnalyzer()
        sa.record_decision(SchedulingDecision(
            step_index=0, timestamp_us=0.0,
            batch_size=4, preempted_request_ids=["r1", "r2"],
            preemption_reasons=[PreemptionReason.KV_CACHE_FULL, PreemptionReason.KV_CACHE_FULL],
        ))
        stats = sa.analyse()
        assert stats.total_preemptions == 2
        assert PreemptionReason.KV_CACHE_FULL.value in stats.preemption_reasons

    def test_batch_timeline_export(self):
        sa = SchedulerAnalyzer()
        sa.record_decision(SchedulingDecision(
            step_index=0, timestamp_us=1000.0, batch_size=4,
        ))
        timeline = sa.export_batch_timeline()
        assert len(timeline) == 1
        assert timeline[0]["batch_size"] == 4

    def test_stats_summary(self):
        stats = SchedulerStats(total_steps=10, mean_batch_size=8.0)
        assert "Scheduler Analysis" in stats.summary()


# ---------------------------------------------------------------------------
# Roofline Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.roofline import (
    BoundType,
    GPUSpec,
    OperationType,
    RooflineAnalyzer,
    RooflineDataPoint,
)


class TestRooflineAnalyzer:
    def test_classify_memory_bound(self):
        ra = RooflineAnalyzer(gpu="A100_80GB")
        point = RooflineDataPoint(
            operation=OperationType.ATTENTION,
            flops=1e9, bytes_accessed=1e9, duration_s=1e-3,
            label="decode_attn",
        )
        result = ra.classify_point(point)
        assert result.bound_type == BoundType.MEMORY_BOUND

    def test_classify_compute_bound(self):
        ra = RooflineAnalyzer(gpu="A100_80GB")
        point = RooflineDataPoint(
            operation=OperationType.GEMM,
            flops=1e15, bytes_accessed=1e9, duration_s=1e-3,
            label="large_gemm",
        )
        result = ra.classify_point(point)
        assert result.bound_type == BoundType.COMPUTE_BOUND

    def test_full_analysis(self):
        ra = RooflineAnalyzer(gpu="H100_SXM")
        ra.add_point(RooflineDataPoint(
            operation=OperationType.GEMM, flops=1e12, bytes_accessed=1e9,
            duration_s=1e-3, is_prefill=True,
        ))
        ra.add_point(RooflineDataPoint(
            operation=OperationType.ATTENTION, flops=1e9, bytes_accessed=1e9,
            duration_s=1e-3, is_prefill=False,
        ))
        profile = ra.analyse()
        assert profile.total_operations == 2
        assert profile.gpu_name == "H100 SXM"

    def test_export_plot_data(self):
        ra = RooflineAnalyzer()
        ra.add_point(RooflineDataPoint(
            operation=OperationType.GEMM, flops=1e12, bytes_accessed=1e9,
            duration_s=1e-3,
        ))
        data = ra.export_plot_data()
        assert "ridge_point" in data
        assert len(data["points"]) == 1

    def test_profile_summary(self):
        from benchmark.analysis.roofline import RooflineProfile
        p = RooflineProfile(gpu_name="A100", total_operations=5)
        assert "Roofline Analysis" in p.summary()

    def test_gpu_spec_ridge_point(self):
        spec = GPUSpec(name="test", peak_fp16_tflops=100, peak_fp32_tflops=50,
                       peak_int8_tops=200, hbm_bandwidth_tbs=2.0)
        assert spec.ridge_point_fp16 == 50.0


# ---------------------------------------------------------------------------
# Scalability Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.scalability import (
    ScalabilityAnalyzer,
    ScalingDataPoint,
    ScalingType,
)


class TestScalabilityAnalyzer:
    def test_strong_scaling(self):
        sa = ScalabilityAnalyzer(scaling_type=ScalingType.STRONG)
        sa.add_point(ScalingDataPoint(num_gpus=1, throughput_tps=100))
        sa.add_point(ScalingDataPoint(num_gpus=2, throughput_tps=180))
        sa.add_point(ScalingDataPoint(num_gpus=4, throughput_tps=320))
        profile = sa.analyse()
        assert len(profile.efficiencies) == 2
        assert profile.efficiencies[0].speedup == 1.8  # 180/100

    def test_amdahl_fit(self):
        sa = ScalabilityAnalyzer()
        sa.add_point(ScalingDataPoint(num_gpus=1, throughput_tps=100))
        sa.add_point(ScalingDataPoint(num_gpus=2, throughput_tps=190))
        sa.add_point(ScalingDataPoint(num_gpus=4, throughput_tps=350))
        sa.add_point(ScalingDataPoint(num_gpus=8, throughput_tps=600))
        profile = sa.analyse()
        assert profile.amdahl_fit is not None
        assert 0.0 < profile.amdahl_fit.serial_fraction < 1.0

    def test_predict_throughput(self):
        sa = ScalabilityAnalyzer()
        sa.add_point(ScalingDataPoint(num_gpus=1, throughput_tps=100))
        sa.add_point(ScalingDataPoint(num_gpus=2, throughput_tps=190))
        prediction = sa.predict_throughput(4)
        assert prediction is not None
        assert prediction > 100

    def test_profile_summary(self):
        from benchmark.analysis.scalability import ScalabilityProfile
        p = ScalabilityProfile(scaling_type=ScalingType.STRONG)
        assert "Scalability" in p.summary()


# ---------------------------------------------------------------------------
# Pipeline Bubble Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.pipeline_bubble import (
    MicrobatchEvent,
    MicrobatchPhase,
    PipelineBubbleAnalyzer,
    PipelineSchedule,
)


class TestPipelineBubbleAnalyzer:
    def test_basic_analysis(self):
        pba = PipelineBubbleAnalyzer(num_stages=4, schedule=PipelineSchedule.ONE_F_ONE_B)
        # 4 stages, 4 microbatches with forward + idle
        for stage in range(4):
            for mb in range(4):
                pba.record(MicrobatchEvent(
                    stage_index=stage, microbatch_index=mb,
                    phase=MicrobatchPhase.FORWARD,
                    start_us=float((stage * 4 + mb) * 100),
                    end_us=float((stage * 4 + mb) * 100 + 80),
                ))
                pba.record(MicrobatchEvent(
                    stage_index=stage, microbatch_index=mb,
                    phase=MicrobatchPhase.IDLE,
                    start_us=float((stage * 4 + mb) * 100 + 80),
                    end_us=float((stage * 4 + mb) * 100 + 100),
                ))
        profile = pba.analyse()
        assert profile.num_stages == 4
        assert profile.num_microbatches == 4
        assert profile.bubble_ratio > 0

    def test_schedule_comparison(self):
        pba = PipelineBubbleAnalyzer(num_stages=4, schedule=PipelineSchedule.GPIPE)
        pba.record(MicrobatchEvent(0, 0, MicrobatchPhase.FORWARD, 0, 100))
        profile = pba.analyse()
        assert len(profile.schedule_comparison) > 0
        assert "zero_bubble" in profile.schedule_comparison

    def test_gantt_export(self):
        pba = PipelineBubbleAnalyzer(num_stages=2)
        pba.record(MicrobatchEvent(0, 0, MicrobatchPhase.FORWARD, 0, 100))
        gantt = pba.export_gantt()
        assert len(gantt) == 1
        assert gantt[0]["phase"] == "forward"

    def test_profile_summary(self):
        from benchmark.analysis.pipeline_bubble import PipelineBubbleProfile
        p = PipelineBubbleProfile(num_stages=4, bubble_ratio=0.25)
        assert "Pipeline Bubble" in p.summary()


# ---------------------------------------------------------------------------
# Quantization Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.quantization_analyzer import (
    LayerSensitivity,
    QuantizationAnalyzer,
    QuantizationResult,
    QuantMethod,
)


class TestQuantizationAnalyzer:
    def test_basic_analysis(self):
        qa = QuantizationAnalyzer()
        qa.add_result(QuantizationResult(
            method=QuantMethod.NONE, bits=16,
            throughput_tps=100, baseline_throughput_tps=100,
            perplexity=5.0, baseline_perplexity=5.0,
        ))
        qa.add_result(QuantizationResult(
            method=QuantMethod.AWQ, bits=4,
            throughput_tps=250, baseline_throughput_tps=100,
            perplexity=5.3, baseline_perplexity=5.0,
            model_size_gb=4.0, baseline_model_size_gb=14.0,
            memory_savings_pct=71.0,
        ))
        profile = qa.analyse()
        assert profile.best_speed_method == QuantMethod.AWQ
        assert len(profile.pareto_frontier) > 0

    def test_layer_sensitivity(self):
        qa = QuantizationAnalyzer()
        qa.add_result(QuantizationResult(method=QuantMethod.NONE, bits=16))
        for i in range(10):
            qa.add_layer_sensitivity(LayerSensitivity(
                layer_index=i, sensitivity_score=i * 0.1,
                recommended_bits=4 if i < 7 else 16,
                perplexity_delta=i * 0.01,
            ))
        profile = qa.analyse()
        assert profile.mixed_precision_plan is not None
        assert len(profile.mixed_precision_plan.sensitive_layers) > 0

    def test_result_properties(self):
        r = QuantizationResult(
            method=QuantMethod.GPTQ, bits=4,
            throughput_tps=200, baseline_throughput_tps=100,
            perplexity=5.5, baseline_perplexity=5.0,
            model_size_gb=4.0, baseline_model_size_gb=14.0,
        )
        assert r.speedup == 2.0
        assert r.perplexity_increase_pct == 10.0
        assert r.compression_ratio == 3.5

    def test_profile_summary(self):
        from benchmark.analysis.quantization_analyzer import QuantizationProfile
        p = QuantizationProfile()
        assert "Quantization" in p.summary()


# ---------------------------------------------------------------------------
# KV Cache Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.kv_cache_analyzer import (
    CacheEvent,
    CacheEventType,
    CacheSnapshot,
    KVCacheAnalyzer,
)


class TestKVCacheAnalyzer:
    def test_basic_analysis(self):
        ka = KVCacheAnalyzer(total_blocks=1024, block_size=16)
        ka.record(CacheEvent(CacheEventType.ALLOCATE, timestamp_us=0, num_blocks=10, num_tokens=150))
        ka.record(CacheEvent(CacheEventType.HIT, timestamp_us=100))
        ka.record(CacheEvent(CacheEventType.MISS, timestamp_us=200))
        profile = ka.analyse()
        assert profile.total_allocations == 1
        assert profile.hit_rate == 0.5

    def test_eviction_rate(self):
        ka = KVCacheAnalyzer()
        ka.record(CacheEvent(CacheEventType.ALLOCATE, timestamp_us=0, num_blocks=5))
        ka.record(CacheEvent(CacheEventType.EVICTION, timestamp_us=100, num_blocks=2))
        profile = ka.analyse()
        assert profile.eviction_rate == 1.0  # 1 eviction / 1 allocation

    def test_prefix_sharing(self):
        ka = KVCacheAnalyzer()
        ka.record(CacheEvent(CacheEventType.PREFIX_HIT, timestamp_us=0, prefix_length=50, num_tokens=100))
        ka.record(CacheEvent(CacheEventType.PREFIX_MISS, timestamp_us=100, prefix_length=0, num_tokens=100))
        ka.record(CacheEvent(CacheEventType.ALLOCATE, timestamp_us=200, num_tokens=200))
        profile = ka.analyse()
        assert profile.prefix_sharing is not None
        assert profile.prefix_sharing.hit_rate == 0.5

    def test_block_size_analysis(self):
        ka = KVCacheAnalyzer(block_size=16)
        for i in range(20):
            ka.record(CacheEvent(CacheEventType.ALLOCATE, timestamp_us=float(i), num_tokens=50 + i * 10))
        profile = ka.analyse()
        assert len(profile.block_size_analysis) > 0

    def test_snapshot_utilization(self):
        ka = KVCacheAnalyzer()
        ka.record_snapshot(CacheSnapshot(timestamp_us=0, utilization_pct=80.0, fragmentation_ratio=0.1))
        ka.record_snapshot(CacheSnapshot(timestamp_us=100, utilization_pct=90.0, fragmentation_ratio=0.15))
        ka.record(CacheEvent(CacheEventType.ALLOCATE, timestamp_us=0))
        profile = ka.analyse()
        assert profile.mean_utilization_pct == 85.0

    def test_profile_summary(self):
        from benchmark.analysis.kv_cache_analyzer import KVCacheProfile
        p = KVCacheProfile(total_events=100, hit_rate=0.8)
        assert "KV Cache" in p.summary()


# ---------------------------------------------------------------------------
# Speculative Analyzer
# ---------------------------------------------------------------------------
from benchmark.analysis.speculative_analyzer import (
    SpeculationMethod,
    SpeculationStep,
    SpeculativeAnalyzer,
)


class TestSpeculativeAnalyzer:
    def test_basic_analysis(self):
        sa = SpeculativeAnalyzer(
            method=SpeculationMethod.STANDARD,
            draft_model_name="llama-68m",
            draft_params_b=0.068, target_params_b=7.0,
        )
        for i in range(20):
            sa.record_step(SpeculationStep(
                step_index=i, draft_tokens=5, accepted_tokens=3,
                draft_time_us=50, verify_time_us=200, baseline_time_us=300,
            ))
        profile = sa.analyse()
        assert profile.mean_acceptance_rate == 0.6
        assert profile.overall_speedup > 1.0
        assert profile.draft_model_stats is not None

    def test_optimal_depth(self):
        sa = SpeculativeAnalyzer()
        for depth in [3, 5, 7]:
            for _ in range(10):
                sa.record_step(SpeculationStep(
                    step_index=0, draft_tokens=depth,
                    accepted_tokens=min(depth, 4),
                    draft_time_us=depth * 20,
                    verify_time_us=200,
                    baseline_time_us=300,
                ))
        profile = sa.analyse()
        assert profile.optimal_depth is not None
        assert profile.optimal_depth.optimal_depth > 0

    def test_step_properties(self):
        s = SpeculationStep(
            step_index=0, draft_tokens=5, accepted_tokens=3,
            draft_time_us=50, verify_time_us=200, baseline_time_us=300,
        )
        assert s.acceptance_rate == 0.6
        assert s.speedup == 300 / 250
        assert s.tokens_per_step == 4.0

    def test_profile_summary(self):
        from benchmark.analysis.speculative_analyzer import SpeculativeProfile
        p = SpeculativeProfile(method=SpeculationMethod.STANDARD, overall_speedup=1.5)
        assert "Speculative" in p.summary()


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------
from benchmark.analysis.anomaly_detector import (
    Anomaly,
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
    MetricSample,
)


class TestAnomalyDetector:
    def test_z_score_detection(self):
        ad = AnomalyDetector(z_threshold=2.0, min_samples=5)
        # Normal samples
        for i in range(20):
            ad.add_sample(MetricSample("latency_ms", float(i), 100.0))
        # Anomalous sample
        ad.add_sample(MetricSample("latency_ms", 20.0, 500.0))
        report = ad.detect()
        assert report.total_anomalies > 0

    def test_no_anomalies(self):
        ad = AnomalyDetector(min_samples=5)
        for i in range(20):
            ad.add_sample(MetricSample("latency_ms", float(i), 100.0 + i * 0.01))
        report = ad.detect()
        # With constant values, no anomalies expected
        assert report.total_samples == 20

    def test_iqr_detection(self):
        ad = AnomalyDetector(iqr_multiplier=1.0, min_samples=5)
        values = [10, 11, 10, 12, 11, 10, 11, 10, 12, 11, 50]  # 50 is outlier
        for i, v in enumerate(values):
            ad.add_sample(MetricSample("throughput_tps", float(i), float(v)))
        report = ad.detect()
        assert report.total_anomalies > 0

    def test_trend_detection(self):
        ad = AnomalyDetector(min_samples=10)
        # Monotonic increase in memory
        for i in range(30):
            ad.add_sample(MetricSample("memory_gb", float(i), 10.0 + i * 2.0))
        report = ad.detect()
        memory_leak = any(
            a.anomaly_type == AnomalyType.MEMORY_LEAK for a in report.anomalies
        )
        assert memory_leak

    def test_anomaly_to_dict(self):
        a = Anomaly(
            anomaly_type=AnomalyType.LATENCY_SPIKE,
            severity=AnomalySeverity.HIGH,
            timestamp_s=1.0, metric_name="latency",
            observed_value=500, expected_value=100,
            deviation=4.0,
            detection_method=__import__("benchmark.analysis.anomaly_detector", fromlist=["DetectionMethod"]).DetectionMethod.Z_SCORE,
            description="spike",
        )
        d = a.to_dict()
        assert d["type"] == "latency_spike"

    def test_report_summary(self):
        from benchmark.analysis.anomaly_detector import AnomalyReport
        r = AnomalyReport(total_samples=100, total_anomalies=3)
        assert "Anomaly" in r.summary()

    def test_reset(self):
        ad = AnomalyDetector()
        ad.add_sample(MetricSample("x", 0.0, 1.0))
        ad.reset()
        report = ad.detect()
        assert report.total_samples == 0


# ---------------------------------------------------------------------------
# Workload Characterizer
# ---------------------------------------------------------------------------
from benchmark.analysis.workload_characterizer import (
    IntensityClass,
    RequestCategory,
    RequestCharacteristics,
    WorkloadCharacterizer,
)


class TestWorkloadCharacterizer:
    def test_basic_characterization(self):
        wc = WorkloadCharacterizer()
        for i in range(50):
            wc.add_request(RequestCharacteristics(
                request_id=f"r{i}",
                arrival_time_s=i * 0.1,
                input_tokens=256 + i,
                output_tokens=128 + i,
                e2e_latency_ms=500 + i * 10,
            ))
        profile = wc.analyse()
        assert profile.total_requests == 50
        assert profile.mean_arrival_rate_rps > 0
        assert profile.input_token_stats is not None

    def test_category_distribution(self):
        wc = WorkloadCharacterizer()
        # Mix of short/long
        for i in range(20):
            wc.add_request(RequestCharacteristics(
                request_id=f"r{i}",
                arrival_time_s=i * 0.5,
                input_tokens=50 if i < 10 else 500,
                output_tokens=20 if i < 5 else 200,
            ))
        profile = wc.analyse()
        assert len(profile.category_distribution) > 0

    def test_burstiness(self):
        wc = WorkloadCharacterizer()
        # Bursty pattern
        for i in range(30):
            t = float(i) if i < 20 else 20.0 + (i - 20) * 0.01  # burst at end
            wc.add_request(RequestCharacteristics(
                request_id=f"r{i}", arrival_time_s=t,
                input_tokens=100, output_tokens=50,
            ))
        profile = wc.analyse()
        assert profile.inter_arrival_cv >= 0

    def test_synthetic_profile(self):
        wc = WorkloadCharacterizer()
        for i in range(30):
            wc.add_request(RequestCharacteristics(
                request_id=f"r{i}", arrival_time_s=i * 0.2,
                input_tokens=200, output_tokens=100,
            ))
        profile = wc.analyse()
        assert profile.synthetic_profile is not None
        assert profile.synthetic_profile.mean_rate_rps > 0

    def test_request_properties(self):
        r = RequestCharacteristics(
            request_id="r1", input_tokens=200, output_tokens=100,
        )
        assert r.total_tokens == 300
        assert r.io_ratio == 2.0

    def test_profile_summary(self):
        from benchmark.analysis.workload_characterizer import WorkloadProfile
        p = WorkloadProfile(total_requests=100)
        assert "Workload" in p.summary()
