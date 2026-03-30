"""Pipeline bubble analyzer for pipeline-parallel LLM serving.

Inspired by:
* GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al. 2019)
* PipeDream: Generalized Pipeline Parallelism for DNN Training (Narayanan et al. 2019)
* Megatron-LM 1F1B Pipeline Schedule (Narayanan et al. 2021)
* Zero Bubble Pipeline Parallelism (Qi et al. 2023)
* Interleaved 1F1B scheduling analysis

Provides detailed pipeline bubble quantification, micro-batch
scheduling visualization, idle time analysis per stage, and
comparison across scheduling policies.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PipelineSchedule(str, Enum):
    """Pipeline scheduling strategies."""

    GPIPE = "gpipe"
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED_1F1B = "interleaved_1f1b"
    ZERO_BUBBLE = "zero_bubble"
    V_SCHEDULE = "v_schedule"
    CHIMERA = "chimera"


class MicrobatchPhase(str, Enum):
    """Phase within a pipeline micro-batch."""

    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    COMMUNICATION = "communication"
    WARMUP = "warmup"
    COOLDOWN = "cooldown"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class MicrobatchEvent:
    """A micro-batch execution event on a pipeline stage."""

    stage_index: int
    microbatch_index: int
    phase: MicrobatchPhase
    start_us: float
    end_us: float
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


@dataclass
class StageStats:
    """Per-stage pipeline statistics."""

    stage_index: int
    total_time_us: float = 0.0
    compute_time_us: float = 0.0
    idle_time_us: float = 0.0
    comm_time_us: float = 0.0
    bubble_ratio: float = 0.0
    num_microbatches: int = 0
    mean_forward_us: float = 0.0
    mean_backward_us: float = 0.0
    is_bottleneck: bool = False

    def summary(self) -> str:
        return (
            f"Stage {self.stage_index}: "
            f"compute={self.compute_time_us / 1e3:.1f}ms, "
            f"idle={self.idle_time_us / 1e3:.1f}ms, "
            f"bubble={self.bubble_ratio:.1%}"
            + (" [BOTTLENECK]" if self.is_bottleneck else "")
        )


@dataclass
class PipelineBubbleProfile:
    """Full pipeline bubble analysis."""

    schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B
    num_stages: int = 0
    num_microbatches: int = 0
    total_pipeline_time_us: float = 0.0
    total_compute_time_us: float = 0.0
    total_bubble_time_us: float = 0.0
    bubble_ratio: float = 0.0
    theoretical_bubble_ratio: float = 0.0
    per_stage_stats: List[StageStats] = field(default_factory=list)
    warmup_time_us: float = 0.0
    cooldown_time_us: float = 0.0
    steady_state_time_us: float = 0.0
    stage_imbalance_ratio: float = 0.0
    bottleneck_stage: int = -1
    schedule_comparison: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Pipeline Bubble Analysis ({self.schedule.value})",
            f"  Stages × Microbatches : {self.num_stages} × {self.num_microbatches}",
            f"  Total pipeline time   : {self.total_pipeline_time_us / 1e3:.1f} ms",
            f"  Total compute time    : {self.total_compute_time_us / 1e3:.1f} ms",
            f"  Total bubble time     : {self.total_bubble_time_us / 1e3:.1f} ms",
            f"  Bubble ratio          : {self.bubble_ratio:.1%}",
            f"  Theoretical bubble    : {self.theoretical_bubble_ratio:.1%}",
            f"  Warmup time           : {self.warmup_time_us / 1e3:.1f} ms",
            f"  Cooldown time         : {self.cooldown_time_us / 1e3:.1f} ms",
            f"  Steady state time     : {self.steady_state_time_us / 1e3:.1f} ms",
            f"  Stage imbalance       : {self.stage_imbalance_ratio:.2f}",
            f"  Bottleneck stage      : {self.bottleneck_stage}",
        ]
        if self.per_stage_stats:
            lines.append("  Per-stage:")
            for s in self.per_stage_stats:
                lines.append(f"    {s.summary()}")
        if self.schedule_comparison:
            lines.append("  Schedule comparison (bubble ratio):")
            for sched, ratio in sorted(self.schedule_comparison.items(), key=lambda x: x[1]):
                lines.append(f"    {sched}: {ratio:.1%}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _theoretical_bubble_ratio(
    schedule: PipelineSchedule, num_stages: int, num_microbatches: int
) -> float:
    """Compute theoretical bubble ratio for a scheduling strategy."""
    p = num_stages
    m = max(num_microbatches, 1)

    if schedule == PipelineSchedule.GPIPE:
        # bubble = (p-1) / (p-1+m) for forward only; with backward: (p-1)/(m+p-1)
        return (p - 1) / (m + p - 1) if (m + p - 1) > 0 else 0.0
    elif schedule == PipelineSchedule.ONE_F_ONE_B:
        # bubble ≈ (p-1) / m for large m
        return (p - 1) / m if m > 0 else 0.0
    elif schedule == PipelineSchedule.INTERLEAVED_1F1B:
        # With v virtual stages: bubble ≈ (p-1) / (m*v)
        # Typically v = num_layers / p, approximate with v=2
        v = 2
        return (p - 1) / (m * v) if m > 0 else 0.0
    elif schedule == PipelineSchedule.ZERO_BUBBLE:
        return 0.0  # theoretically zero bubble
    else:
        return (p - 1) / m if m > 0 else 0.0


class PipelineBubbleAnalyzer:
    """Analyses pipeline parallel bubble overhead.

    Usage::

        pba = PipelineBubbleAnalyzer(num_stages=4, schedule=PipelineSchedule.ONE_F_ONE_B)
        pba.record(MicrobatchEvent(...))
        profile = pba.analyse()
        print(profile.summary())
    """

    def __init__(
        self,
        num_stages: int = 4,
        schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B,
    ) -> None:
        self._num_stages = num_stages
        self._schedule = schedule
        self._events: List[MicrobatchEvent] = []

    def record(self, event: MicrobatchEvent) -> None:
        self._events.append(event)

    def record_batch(self, events: List[MicrobatchEvent]) -> None:
        self._events.extend(events)

    def analyse(self) -> PipelineBubbleProfile:
        if not self._events:
            return PipelineBubbleProfile(
                schedule=self._schedule, num_stages=self._num_stages
            )

        # Group by stage
        by_stage: Dict[int, List[MicrobatchEvent]] = {}
        for e in self._events:
            by_stage.setdefault(e.stage_index, []).append(e)

        microbatch_indices = set(e.microbatch_index for e in self._events)
        num_microbatches = len(microbatch_indices)

        # Per-stage stats
        stage_stats: List[StageStats] = []
        stage_compute_times: List[float] = []

        for stage_idx in range(self._num_stages):
            events = by_stage.get(stage_idx, [])
            compute = sum(
                e.duration_us
                for e in events
                if e.phase in (MicrobatchPhase.FORWARD, MicrobatchPhase.BACKWARD)
            )
            idle = sum(
                e.duration_us
                for e in events
                if e.phase == MicrobatchPhase.IDLE
            )
            comm = sum(
                e.duration_us
                for e in events
                if e.phase == MicrobatchPhase.COMMUNICATION
            )
            total = compute + idle + comm
            fwd_times = [
                e.duration_us for e in events if e.phase == MicrobatchPhase.FORWARD
            ]
            bwd_times = [
                e.duration_us for e in events if e.phase == MicrobatchPhase.BACKWARD
            ]

            bubble = idle / total if total > 0 else 0.0
            stage_compute_times.append(compute)

            stage_stats.append(
                StageStats(
                    stage_index=stage_idx,
                    total_time_us=total,
                    compute_time_us=compute,
                    idle_time_us=idle,
                    comm_time_us=comm,
                    bubble_ratio=bubble,
                    num_microbatches=len(set(e.microbatch_index for e in events)),
                    mean_forward_us=(
                        statistics.mean(fwd_times) if fwd_times else 0.0
                    ),
                    mean_backward_us=(
                        statistics.mean(bwd_times) if bwd_times else 0.0
                    ),
                )
            )

        # Identify bottleneck (stage with most compute)
        if stage_compute_times:
            bottleneck_idx = stage_compute_times.index(max(stage_compute_times))
            stage_stats[bottleneck_idx].is_bottleneck = True
        else:
            bottleneck_idx = -1

        # Stage imbalance
        if stage_compute_times and len(stage_compute_times) >= 2:
            mean_comp = statistics.mean(stage_compute_times)
            max_comp = max(stage_compute_times)
            imbalance = max_comp / mean_comp if mean_comp > 0 else 1.0
        else:
            imbalance = 1.0

        # Overall times
        total_pipeline = sum(s.total_time_us for s in stage_stats)
        total_compute = sum(s.compute_time_us for s in stage_stats)
        total_bubble = sum(s.idle_time_us for s in stage_stats)

        # Warmup / cooldown (approximate from first/last microbatch)
        warmup = sum(
            e.duration_us
            for e in self._events
            if e.phase == MicrobatchPhase.WARMUP
        )
        cooldown = sum(
            e.duration_us
            for e in self._events
            if e.phase == MicrobatchPhase.COOLDOWN
        )

        # Theoretical comparison
        theoretical = _theoretical_bubble_ratio(
            self._schedule, self._num_stages, num_microbatches
        )

        # Compare all schedules
        schedule_comp: Dict[str, float] = {}
        for sched in PipelineSchedule:
            schedule_comp[sched.value] = _theoretical_bubble_ratio(
                sched, self._num_stages, num_microbatches
            )

        bubble_ratio = total_bubble / total_pipeline if total_pipeline > 0 else 0.0

        recs = self._generate_recommendations(
            bubble_ratio, theoretical, imbalance, num_microbatches
        )

        return PipelineBubbleProfile(
            schedule=self._schedule,
            num_stages=self._num_stages,
            num_microbatches=num_microbatches,
            total_pipeline_time_us=total_pipeline,
            total_compute_time_us=total_compute,
            total_bubble_time_us=total_bubble,
            bubble_ratio=bubble_ratio,
            theoretical_bubble_ratio=theoretical,
            per_stage_stats=stage_stats,
            warmup_time_us=warmup,
            cooldown_time_us=cooldown,
            steady_state_time_us=max(0, total_pipeline - warmup - cooldown),
            stage_imbalance_ratio=imbalance,
            bottleneck_stage=bottleneck_idx,
            schedule_comparison=schedule_comp,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        actual_bubble: float,
        theoretical_bubble: float,
        imbalance: float,
        num_microbatches: int,
    ) -> List[str]:
        recs: List[str] = []

        if actual_bubble > 0.3:
            recs.append(
                f"Bubble ratio is high ({actual_bubble:.0%}). "
                "Increase the number of microbatches to reduce pipeline stalls."
            )

        if actual_bubble > theoretical_bubble * 1.5 and theoretical_bubble > 0:
            recs.append(
                f"Actual bubble ({actual_bubble:.0%}) exceeds theoretical "
                f"({theoretical_bubble:.0%}) by {(actual_bubble - theoretical_bubble) / theoretical_bubble:.0%}. "
                "Communication overhead or stage imbalance may be the cause."
            )

        if imbalance > 1.3:
            recs.append(
                f"Stage imbalance ratio is {imbalance:.2f}. "
                "Consider redistributing layers across stages for better balance."
            )

        if self._schedule == PipelineSchedule.GPIPE and actual_bubble > 0.2:
            recs.append(
                "Consider switching from GPipe to 1F1B schedule to reduce bubble overhead."
            )

        if num_microbatches < self._num_stages * 2:
            recs.append(
                f"Only {num_microbatches} microbatches for {self._num_stages} stages. "
                "Rule of thumb: use at least 4× stages for good pipeline efficiency."
            )

        return recs

    def export_gantt(self) -> List[Dict]:
        """Export events for pipeline Gantt chart visualization."""
        return [
            {
                "stage": e.stage_index,
                "microbatch": e.microbatch_index,
                "phase": e.phase.value,
                "start_us": e.start_us,
                "end_us": e.end_us,
                "duration_us": e.duration_us,
            }
            for e in self._events
        ]

    def reset(self) -> None:
        self._events.clear()
