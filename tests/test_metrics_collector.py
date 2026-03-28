"""Tests for the metrics collector (Stage 4)."""

import pytest

from benchmark.metrics.collector import (
    MetricsCollector,
    RequestTiming,
    _latency_percentiles,
    _percentile,
    parse_prometheus_float,
)


class TestPercentile:
    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0

    def test_p50(self):
        data = list(range(1, 101))
        assert abs(_percentile(data, 50) - 50.0) < 1.0

    def test_p99(self):
        data = list(range(1, 101))
        val = _percentile(data, 99)
        assert val >= 98.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _percentile([], 50)

    def test_p0(self):
        assert _percentile([3.0, 1.0, 2.0], 0) == 1.0

    def test_p100(self):
        assert _percentile([3.0, 1.0, 2.0], 100) == 3.0


class TestLatencyPercentiles:
    def test_empty(self):
        assert _latency_percentiles([]) == (0.0, 0.0, 0.0)

    def test_uniform(self):
        data = [100.0] * 100
        p50, p90, p99 = _latency_percentiles(data)
        assert p50 == 100.0
        assert p90 == 100.0
        assert p99 == 100.0


class TestParsePrometheusFloat:
    def test_match(self):
        text = 'vllm:tokens_total{model="llama"} 12345.0\n'
        pattern = r'vllm:tokens_total\{[^}]*\}\s+([\d.]+)'
        val = parse_prometheus_float(text, pattern)
        assert val == 12345.0

    def test_no_match(self):
        text = "# some comment\n"
        val = parse_prometheus_float(text, r'vllm:tokens_total\{[^}]*\}\s+([\d.]+)')
        assert val is None


class TestMetricsCollector:
    def _make_timing(self, req_id, ttft, tpot, e2e, prefill_exec=10.0,
                     prefill_queue=5.0, decode_exec=20.0, decode_queue=2.0,
                     kv_tx=0.0):
        return RequestTiming(
            request_id=req_id,
            ttft_ms=ttft,
            tpot_ms=tpot,
            e2e_latency_ms=e2e,
            prefill_exec_ms=prefill_exec,
            prefill_queue_ms=prefill_queue,
            decode_exec_ms=decode_exec,
            decode_queue_ms=decode_queue,
            kv_transmission_ms=kv_tx,
        )

    def test_empty_collect(self):
        collector = MetricsCollector()
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=0,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.throughput_tps == 0.0
        assert metrics.ttft_p50_ms == 0.0
        assert metrics.joint_slo_attainment_pct == 0.0

    def test_throughput_calculation(self):
        collector = MetricsCollector()
        t = self._make_timing("r0", ttft=100.0, tpot=50.0, e2e=200.0)
        collector.record_request(t)
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=1000,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.throughput_tps == 100.0  # 1000/10
        assert metrics.end_to_end_throughput_rps == 0.1  # 1/10

    def test_slo_attainment_all_pass(self):
        collector = MetricsCollector()
        for i in range(10):
            t = self._make_timing(str(i), ttft=100.0, tpot=50.0, e2e=200.0)
            collector.record_request(t)
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=1000,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.ttft_slo_attainment_pct == 100.0
        assert metrics.tpot_slo_attainment_pct == 100.0
        assert metrics.joint_slo_attainment_pct == 100.0

    def test_slo_attainment_none_pass(self):
        collector = MetricsCollector()
        for i in range(10):
            t = self._make_timing(str(i), ttft=500.0, tpot=200.0, e2e=800.0)
            collector.record_request(t)
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=1000,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.ttft_slo_attainment_pct == 0.0
        assert metrics.joint_slo_attainment_pct == 0.0

    def test_phase_timing(self):
        collector = MetricsCollector()
        for i in range(4):
            t = self._make_timing(
                str(i), ttft=100.0, tpot=50.0, e2e=200.0,
                prefill_exec=20.0, prefill_queue=5.0,
                decode_exec=40.0, decode_queue=3.0, kv_tx=10.0,
            )
            collector.record_request(t)
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=400,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.prefill_exec_time_ms == 20.0
        assert metrics.decode_exec_time_ms == 40.0
        assert metrics.kv_transmission_time_ms == 10.0
        # Phase percentages
        total = 20.0 + 40.0 + 10.0
        assert abs(metrics.prefill_phase_time_pct - 20.0 / total * 100.0) < 1e-6
        assert abs(metrics.decode_phase_time_pct - 40.0 / total * 100.0) < 1e-6
        assert abs(metrics.transmission_time_pct - 10.0 / total * 100.0) < 1e-6

    def test_reset(self):
        collector = MetricsCollector()
        t = self._make_timing("r0", 100.0, 50.0, 200.0)
        collector.record_request(t)
        collector.reset()
        metrics = collector.collect(
            total_time_s=10.0,
            num_output_tokens=100,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.joint_slo_attainment_pct == 0.0

    def test_record_requests_batch(self):
        collector = MetricsCollector()
        timings = [
            self._make_timing(str(i), ttft=100.0, tpot=50.0, e2e=200.0)
            for i in range(5)
        ]
        collector.record_requests(timings)
        metrics = collector.collect(
            total_time_s=5.0,
            num_output_tokens=500,
            ttft_slo_ms=250.0,
            tpot_slo_ms=100.0,
        )
        assert metrics.end_to_end_throughput_rps == 1.0  # 5/5
