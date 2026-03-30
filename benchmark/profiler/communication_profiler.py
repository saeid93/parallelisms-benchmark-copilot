"""Communication profiler for distributed LLM serving.

Inspired by:
* DeepSpeed Flops Profiler & Communication Logger
* NCCL Tests (nccl-tests) bandwidth benchmarks
* Megatron-LM communication overlap analysis
* Alpa inter-operator / intra-operator parallelism profiling

Profiles collective communication operations (AllReduce, AllGather,
ReduceScatter, Send/Recv, AllToAll) with per-operation timing,
bandwidth measurement, overlap detection, and hotspot identification.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CollectiveOp(str, Enum):
    """NCCL collective operation types."""

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"
    REDUCE = "reduce"
    GATHER = "gather"
    SCATTER = "scatter"


class CommBackend(str, Enum):
    """Communication backend."""

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    CUSTOM = "custom"


class CommTopology(str, Enum):
    """Interconnect topology."""

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"
    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class CommEvent:
    """A single communication operation record."""

    op: CollectiveOp
    start_us: float
    end_us: float
    message_size_bytes: int
    src_rank: int = 0
    dst_rank: int = -1
    world_size: int = 1
    backend: CommBackend = CommBackend.NCCL
    topology: CommTopology = CommTopology.INTRA_NODE
    group_name: str = ""
    phase: str = ""  # "prefill", "decode", "pipeline_flush", etc.
    overlapped_with_compute: bool = False
    stream_id: int = 0

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def bandwidth_gbps(self) -> float:
        dur_s = self.duration_us / 1e6
        if dur_s <= 0:
            return 0.0
        return (self.message_size_bytes * 8) / (dur_s * 1e9)

    @property
    def algorithm_bandwidth_gbps(self) -> float:
        """Algorithm bandwidth corrected for collective type."""
        raw = self.bandwidth_gbps
        n = max(self.world_size, 1)
        if self.op in (CollectiveOp.ALL_REDUCE,):
            return raw * (2 * (n - 1) / n) if n > 1 else raw
        if self.op in (CollectiveOp.ALL_GATHER, CollectiveOp.REDUCE_SCATTER):
            return raw * ((n - 1) / n) if n > 1 else raw
        return raw

    def to_dict(self) -> Dict:
        return {
            "op": self.op.value,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "duration_us": self.duration_us,
            "message_size_bytes": self.message_size_bytes,
            "bandwidth_gbps": round(self.bandwidth_gbps, 3),
            "algo_bw_gbps": round(self.algorithm_bandwidth_gbps, 3),
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "world_size": self.world_size,
            "backend": self.backend.value,
            "topology": self.topology.value,
            "group_name": self.group_name,
            "phase": self.phase,
            "overlapped": self.overlapped_with_compute,
        }


@dataclass
class CommStats:
    """Aggregated statistics for a communication operation type."""

    op: CollectiveOp
    count: int = 0
    total_time_us: float = 0.0
    mean_time_us: float = 0.0
    p50_time_us: float = 0.0
    p99_time_us: float = 0.0
    max_time_us: float = 0.0
    total_bytes: int = 0
    mean_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    overlap_ratio: float = 0.0  # fraction of time overlapped with compute

    def summary(self) -> str:
        return (
            f"{self.op.value}: count={self.count}, "
            f"total={self.total_time_us / 1e3:.1f}ms, "
            f"mean={self.mean_time_us:.1f}us, "
            f"p99={self.p99_time_us:.1f}us, "
            f"bw={self.mean_bandwidth_gbps:.2f}Gbps, "
            f"overlap={self.overlap_ratio:.1%}"
        )


@dataclass
class CommBottleneck:
    """An identified communication bottleneck."""

    description: str
    severity: float  # 0.0 - 1.0
    op: CollectiveOp
    affected_ranks: List[int] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class CommProfile:
    """Full communication profile summary."""

    total_comm_time_us: float = 0.0
    total_compute_time_us: float = 0.0
    comm_compute_ratio: float = 0.0
    per_op_stats: Dict[str, CommStats] = field(default_factory=dict)
    bottlenecks: List[CommBottleneck] = field(default_factory=list)
    inter_node_fraction: float = 0.0
    intra_node_fraction: float = 0.0
    overlap_efficiency: float = 0.0
    total_bytes_transferred: int = 0

    def summary(self) -> str:
        lines = [
            "Communication Profile",
            f"  Total comm time     : {self.total_comm_time_us / 1e3:.1f} ms",
            f"  Total compute time  : {self.total_compute_time_us / 1e3:.1f} ms",
            f"  Comm/Compute ratio  : {self.comm_compute_ratio:.3f}",
            f"  Overlap efficiency  : {self.overlap_efficiency:.1%}",
            f"  Bytes transferred   : {self.total_bytes_transferred / (1024**2):.1f} MiB",
            f"  Inter-node fraction : {self.inter_node_fraction:.1%}",
            f"  Intra-node fraction : {self.intra_node_fraction:.1%}",
            "",
            "  Per-op breakdown:",
        ]
        for stats in self.per_op_stats.values():
            lines.append(f"    {stats.summary()}")
        if self.bottlenecks:
            lines.append("")
            lines.append("  Bottlenecks:")
            for b in self.bottlenecks:
                lines.append(f"    [{b.severity:.2f}] {b.description}")
                if b.suggestion:
                    lines.append(f"           -> {b.suggestion}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


class CommunicationProfiler:
    """Records and analyses distributed communication patterns.

    Usage::

        cp = CommunicationProfiler(world_size=8)
        cp.record(CommEvent(...))
        profile = cp.analyse(total_compute_time_us=1e6)
        print(profile.summary())
    """

    def __init__(self, world_size: int = 1) -> None:
        self._world_size = world_size
        self._events: List[CommEvent] = []

    def record(self, event: CommEvent) -> None:
        """Record a communication event."""
        self._events.append(event)

    def record_batch(self, events: List[CommEvent]) -> None:
        """Record multiple events."""
        self._events.extend(events)

    def analyse(
        self, total_compute_time_us: float = 0.0
    ) -> CommProfile:
        """Analyse recorded communication events."""
        if not self._events:
            return CommProfile()

        # group by operation type
        by_op: Dict[str, List[CommEvent]] = {}
        for e in self._events:
            by_op.setdefault(e.op.value, []).append(e)

        per_op: Dict[str, CommStats] = {}
        total_comm_us = 0.0
        total_bytes = 0
        total_overlapped_us = 0.0

        for op_name, events in by_op.items():
            durations = [e.duration_us for e in events]
            bw_list = [e.bandwidth_gbps for e in events]
            overlap_count = sum(1 for e in events if e.overlapped_with_compute)
            total_op_time = sum(durations)
            total_op_bytes = sum(e.message_size_bytes for e in events)
            overlapped_time = sum(
                e.duration_us for e in events if e.overlapped_with_compute
            )

            stats = CommStats(
                op=CollectiveOp(op_name),
                count=len(events),
                total_time_us=total_op_time,
                mean_time_us=statistics.mean(durations) if durations else 0.0,
                p50_time_us=_percentile(durations, 50),
                p99_time_us=_percentile(durations, 99),
                max_time_us=max(durations) if durations else 0.0,
                total_bytes=total_op_bytes,
                mean_bandwidth_gbps=(
                    statistics.mean(bw_list) if bw_list else 0.0
                ),
                peak_bandwidth_gbps=max(bw_list) if bw_list else 0.0,
                overlap_ratio=(
                    overlap_count / len(events) if events else 0.0
                ),
            )
            per_op[op_name] = stats
            total_comm_us += total_op_time
            total_bytes += total_op_bytes
            total_overlapped_us += overlapped_time

        # topology breakdown
        inter_count = sum(
            1
            for e in self._events
            if e.topology in (CommTopology.INTER_NODE, CommTopology.INFINIBAND, CommTopology.ETHERNET)
        )
        intra_count = sum(
            1
            for e in self._events
            if e.topology in (CommTopology.INTRA_NODE, CommTopology.NVLINK, CommTopology.PCIE)
        )
        total_count = len(self._events)

        # bottleneck detection
        bottlenecks = self._detect_bottlenecks(per_op, total_comm_us, total_compute_time_us)

        overlap_eff = (
            total_overlapped_us / total_comm_us if total_comm_us > 0 else 0.0
        )

        return CommProfile(
            total_comm_time_us=total_comm_us,
            total_compute_time_us=total_compute_time_us,
            comm_compute_ratio=(
                total_comm_us / total_compute_time_us
                if total_compute_time_us > 0
                else 0.0
            ),
            per_op_stats=per_op,
            bottlenecks=bottlenecks,
            inter_node_fraction=(
                inter_count / total_count if total_count > 0 else 0.0
            ),
            intra_node_fraction=(
                intra_count / total_count if total_count > 0 else 0.0
            ),
            overlap_efficiency=overlap_eff,
            total_bytes_transferred=total_bytes,
        )

    def _detect_bottlenecks(
        self,
        per_op: Dict[str, CommStats],
        total_comm_us: float,
        total_compute_us: float,
    ) -> List[CommBottleneck]:
        bottlenecks: List[CommBottleneck] = []

        # High comm/compute ratio
        if total_compute_us > 0:
            ratio = total_comm_us / total_compute_us
            if ratio > 0.3:
                bottlenecks.append(
                    CommBottleneck(
                        description=f"Communication dominates compute ({ratio:.1%} ratio)",
                        severity=min(1.0, ratio),
                        op=CollectiveOp.ALL_REDUCE,
                        suggestion="Consider increasing TP to reduce message sizes or enable communication-computation overlap.",
                    )
                )

        # AllReduce domination
        ar = per_op.get(CollectiveOp.ALL_REDUCE.value)
        if ar and total_comm_us > 0:
            ar_frac = ar.total_time_us / total_comm_us
            if ar_frac > 0.6:
                bottlenecks.append(
                    CommBottleneck(
                        description=f"AllReduce dominates communication ({ar_frac:.1%})",
                        severity=ar_frac,
                        op=CollectiveOp.ALL_REDUCE,
                        suggestion="Use gradient compression or reduce DP degree.",
                    )
                )

        # Low bandwidth utilisation
        for stats in per_op.values():
            if stats.peak_bandwidth_gbps > 0 and stats.mean_bandwidth_gbps > 0:
                util = stats.mean_bandwidth_gbps / stats.peak_bandwidth_gbps
                if util < 0.5 and stats.count >= 5:
                    bottlenecks.append(
                        CommBottleneck(
                            description=(
                                f"{stats.op.value}: low bandwidth utilisation "
                                f"({util:.1%} of peak)"
                            ),
                            severity=1.0 - util,
                            op=stats.op,
                            suggestion="Small messages are inefficient; batch or pipeline them.",
                        )
                    )

        # Tail latency
        for stats in per_op.values():
            if stats.mean_time_us > 0 and stats.p99_time_us > 3 * stats.mean_time_us:
                bottlenecks.append(
                    CommBottleneck(
                        description=(
                            f"{stats.op.value}: high tail latency "
                            f"(p99={stats.p99_time_us:.0f}us vs mean={stats.mean_time_us:.0f}us)"
                        ),
                        severity=min(
                            1.0, stats.p99_time_us / (stats.mean_time_us * 5)
                        ),
                        op=stats.op,
                        suggestion="Check for stragglers or network congestion.",
                    )
                )

        bottlenecks.sort(key=lambda b: b.severity, reverse=True)
        return bottlenecks

    def export_timeline(self) -> List[Dict]:
        """Export as Chrome Trace events for visualisation."""
        events: List[Dict] = []
        for e in self._events:
            events.append(
                {
                    "name": e.op.value,
                    "cat": "communication",
                    "ph": "X",
                    "ts": e.start_us,
                    "dur": e.duration_us,
                    "pid": e.src_rank,
                    "tid": e.stream_id,
                    "args": {
                        "size_bytes": e.message_size_bytes,
                        "bw_gbps": round(e.bandwidth_gbps, 3),
                        "topology": e.topology.value,
                        "phase": e.phase,
                    },
                }
            )
        return events

    def per_phase_breakdown(self) -> Dict[str, float]:
        """Return total comm time per phase (e.g. prefill vs decode)."""
        result: Dict[str, float] = {}
        for e in self._events:
            phase = e.phase or "unknown"
            result[phase] = result.get(phase, 0.0) + e.duration_us
        return result

    def reset(self) -> None:
        """Clear all recorded events."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
