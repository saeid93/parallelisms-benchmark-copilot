"""Energy and carbon footprint profiler for LLM serving.

Inspired by:
* "Quantifying the Carbon Emissions of Machine Learning" (Lacoste et al. 2019)
* "Green AI" (Schwartz et al. 2020)
* ML CO2 Impact Calculator (mlco2.github.io)
* CodeCarbon / Carbontracker projects

Tracks GPU power draw over time, computes energy per token / request,
and estimates CO₂ emissions based on regional grid carbon intensity.
Supports PUE-aware datacenter modeling.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carbon intensity data (gCO₂eq / kWh) — 2023 averages
# ---------------------------------------------------------------------------

GRID_CARBON_INTENSITY: Dict[str, float] = {
    "us-east-1": 379.0,       # Virginia
    "us-west-2": 102.0,       # Oregon (hydro)
    "eu-west-1": 296.0,       # Ireland
    "eu-central-1": 338.0,    # Frankfurt
    "eu-north-1": 8.0,        # Stockholm (hydro+nuclear)
    "ap-northeast-1": 462.0,  # Tokyo
    "ap-southeast-1": 408.0,  # Singapore
    "us-central1": 440.0,     # Iowa (GCP)
    "global-average": 475.0,
}


class GridRegion(str, Enum):
    """Well-known cloud regions for carbon accounting."""

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    US_CENTRAL1 = "us-central1"
    GLOBAL_AVERAGE = "global-average"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class PowerSample:
    """A single GPU power reading."""

    timestamp_s: float
    gpu_index: int
    power_w: float
    temperature_c: float = 0.0
    gpu_utilization_pct: float = 0.0


@dataclass
class EnergyStats:
    """Energy statistics for a profiling window."""

    gpu_index: int = 0
    num_samples: int = 0
    duration_s: float = 0.0
    total_energy_j: float = 0.0
    total_energy_kwh: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    min_power_w: float = 0.0
    idle_power_w: float = 0.0
    energy_per_token_j: float = 0.0
    energy_per_request_j: float = 0.0
    co2_grams: float = 0.0
    co2_per_token_mg: float = 0.0
    co2_per_request_mg: float = 0.0
    grid_region: str = "global-average"
    pue: float = 1.0

    def summary(self) -> str:
        lines = [
            f"Energy Stats (GPU {self.gpu_index})",
            f"  Duration         : {self.duration_s:.2f} s",
            f"  Total energy     : {self.total_energy_j:.1f} J ({self.total_energy_kwh:.6f} kWh)",
            f"  Mean power       : {self.mean_power_w:.1f} W",
            f"  Peak power       : {self.peak_power_w:.1f} W",
            f"  Energy/token     : {self.energy_per_token_j:.4f} J",
            f"  Energy/request   : {self.energy_per_request_j:.4f} J",
            f"  CO₂ total        : {self.co2_grams:.4f} g",
            f"  CO₂/token        : {self.co2_per_token_mg:.4f} mg",
            f"  CO₂/request      : {self.co2_per_request_mg:.4f} mg",
            f"  Grid region      : {self.grid_region}",
            f"  PUE              : {self.pue}",
        ]
        return "\n".join(lines)


@dataclass
class CarbonReport:
    """Aggregated carbon footprint report across all GPUs."""

    total_energy_kwh: float = 0.0
    total_co2_grams: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    per_gpu_stats: Dict[int, EnergyStats] = field(default_factory=dict)
    equivalent_km_driven: float = 0.0       # avg car ~120g CO2/km
    equivalent_trees_days: float = 0.0      # tree absorbs ~22kg CO2/year ≈ 60g/day
    grid_region: str = "global-average"
    pue: float = 1.0

    def summary(self) -> str:
        lines = [
            "Carbon Footprint Report",
            f"  Total energy       : {self.total_energy_kwh:.6f} kWh",
            f"  Total CO₂          : {self.total_co2_grams:.4f} g",
            f"  Tokens processed   : {self.total_tokens}",
            f"  Requests processed : {self.total_requests}",
            f"  Equiv. km driven   : {self.equivalent_km_driven:.4f} km",
            f"  Equiv. tree·days   : {self.equivalent_trees_days:.4f}",
            f"  Grid: {self.grid_region}, PUE: {self.pue}",
        ]
        for gpu_idx, stats in sorted(self.per_gpu_stats.items()):
            lines.append(f"\n  GPU {gpu_idx}:")
            lines.append(f"    Energy: {stats.total_energy_j:.1f} J, CO₂: {stats.co2_grams:.4f} g")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Energy Profiler
# ---------------------------------------------------------------------------


class EnergyProfiler:
    """Tracks power consumption and estimates carbon footprint.

    Usage::

        ep = EnergyProfiler(grid_region="us-west-2", pue=1.1)
        ep.start_sampling()
        # ... run workload ...
        ep.stop_sampling()
        report = ep.compute_report(total_tokens=10000, total_requests=100)
        print(report.summary())
    """

    def __init__(
        self,
        gpu_indices: Optional[List[int]] = None,
        sample_interval_s: float = 1.0,
        grid_region: str = "global-average",
        pue: float = 1.0,
        idle_power_w: float = 50.0,
    ) -> None:
        self._gpu_indices = gpu_indices or [0]
        self._sample_interval = sample_interval_s
        self._grid_region = grid_region
        self._pue = max(1.0, pue)
        self._idle_power_w = idle_power_w
        self._samples: Dict[int, List[PowerSample]] = {
            d: [] for d in self._gpu_indices
        }
        self._sampling = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---- recording ----

    def record_sample(self, sample: PowerSample) -> None:
        """Manually record a power sample."""
        dev = sample.gpu_index
        if dev not in self._samples:
            self._samples[dev] = []
        with self._lock:
            self._samples[dev].append(sample)

    def start_sampling(self, interval_s: Optional[float] = None) -> None:
        """Start background power sampling."""
        if self._sampling:
            return
        self._sampling = True
        iv = interval_s or self._sample_interval
        self._thread = threading.Thread(
            target=self._sample_loop, args=(iv,), daemon=True
        )
        self._thread.start()

    def stop_sampling(self) -> None:
        """Stop background sampling."""
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _sample_loop(self, interval_s: float) -> None:
        while self._sampling:
            ts = time.time()
            for gpu in self._gpu_indices:
                # In real deployment, read from NVML; here we record stubs
                self.record_sample(
                    PowerSample(
                        timestamp_s=ts,
                        gpu_index=gpu,
                        power_w=0.0,
                    )
                )
            time.sleep(interval_s)

    # ---- analysis ----

    def compute_energy(
        self,
        gpu_index: int = 0,
        total_tokens: int = 0,
        total_requests: int = 0,
    ) -> EnergyStats:
        """Compute energy stats for a single GPU."""
        samples = self._samples.get(gpu_index, [])
        if not samples:
            return EnergyStats(gpu_index=gpu_index)

        powers = [s.power_w for s in samples]
        timestamps = [s.timestamp_s for s in samples]
        duration = max(timestamps) - min(timestamps)

        # trapezoidal integration for energy
        energy_j = 0.0
        for i in range(1, len(samples)):
            dt = timestamps[i] - timestamps[i - 1]
            avg_power = (powers[i] + powers[i - 1]) / 2.0
            energy_j += avg_power * dt

        # apply PUE
        energy_j *= self._pue
        energy_kwh = energy_j / 3_600_000.0

        # carbon
        ci = GRID_CARBON_INTENSITY.get(self._grid_region, 475.0)
        co2_g = energy_kwh * ci

        return EnergyStats(
            gpu_index=gpu_index,
            num_samples=len(samples),
            duration_s=duration,
            total_energy_j=energy_j,
            total_energy_kwh=energy_kwh,
            mean_power_w=statistics.mean(powers) if powers else 0.0,
            peak_power_w=max(powers) if powers else 0.0,
            min_power_w=min(powers) if powers else 0.0,
            idle_power_w=self._idle_power_w,
            energy_per_token_j=energy_j / total_tokens if total_tokens > 0 else 0.0,
            energy_per_request_j=energy_j / total_requests if total_requests > 0 else 0.0,
            co2_grams=co2_g,
            co2_per_token_mg=(co2_g * 1000 / total_tokens) if total_tokens > 0 else 0.0,
            co2_per_request_mg=(co2_g * 1000 / total_requests) if total_requests > 0 else 0.0,
            grid_region=self._grid_region,
            pue=self._pue,
        )

    def compute_report(
        self, total_tokens: int = 0, total_requests: int = 0
    ) -> CarbonReport:
        """Compute a full carbon footprint report across all GPUs."""
        per_gpu: Dict[int, EnergyStats] = {}
        total_kwh = 0.0
        total_co2 = 0.0

        for gpu in self._gpu_indices:
            stats = self.compute_energy(gpu, total_tokens, total_requests)
            per_gpu[gpu] = stats
            total_kwh += stats.total_energy_kwh
            total_co2 += stats.co2_grams

        # equivalencies
        equiv_km = total_co2 / 120.0 if total_co2 > 0 else 0.0  # 120g CO2/km
        equiv_trees = total_co2 / 60.0 if total_co2 > 0 else 0.0  # 60g CO2/tree·day

        return CarbonReport(
            total_energy_kwh=total_kwh,
            total_co2_grams=total_co2,
            total_tokens=total_tokens,
            total_requests=total_requests,
            per_gpu_stats=per_gpu,
            equivalent_km_driven=equiv_km,
            equivalent_trees_days=equiv_trees,
            grid_region=self._grid_region,
            pue=self._pue,
        )

    def export_power_timeseries(
        self, gpu_index: int = 0
    ) -> List[Dict]:
        """Export power samples as time-series dicts."""
        samples = self._samples.get(gpu_index, [])
        return [
            {
                "timestamp_s": s.timestamp_s,
                "power_w": s.power_w,
                "temperature_c": s.temperature_c,
                "utilization_pct": s.gpu_utilization_pct,
            }
            for s in samples
        ]

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            for d in self._gpu_indices:
                self._samples[d] = []
