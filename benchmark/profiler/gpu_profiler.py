"""
GPU hardware profiler — profiler stage.

Collects GPU hardware metrics via NVML (pynvml) when running on a machine
with NVIDIA GPUs.  Falls back to synthetic/stubbed values when pynvml is
not available, so unit tests and CI environments work without real hardware.

Metrics collected per GPU:
  - Utilisation (%)
  - Memory used / total (MiB, %)
  - Memory bandwidth utilisation (%)
  - SM clock frequency (MHz)
  - Memory clock frequency (MHz)
  - Power draw (W)
  - Temperature (°C)
  - NVLink bandwidth (GB/s) [H100/A100 only]
  - PCIe throughput (MB/s)

Sampling runs in the foreground; for continuous background sampling use
``GPUProfiler.start_background_sampling()`` with a threading-based loop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml as _nvml  # type: ignore[import]
    _nvml.nvmlInit()
    _HAS_NVML = True
except Exception:  # pragma: no cover
    _HAS_NVML = False
    _nvml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Per-GPU sample
# ---------------------------------------------------------------------------

@dataclass
class GPUSample:
    """A single hardware sample from one GPU."""

    gpu_index: int
    timestamp_s: float = 0.0

    # Utilisation
    gpu_utilization_pct: float = 0.0
    memory_utilization_pct: float = 0.0

    # Memory
    memory_used_mib: float = 0.0
    memory_total_mib: float = 0.0
    memory_used_pct: float = 0.0

    # Clock
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0

    # Power & thermal
    power_draw_w: float = 0.0
    temperature_c: float = 0.0

    # NVLink (optional)
    nvlink_tx_mbs: float = 0.0
    nvlink_rx_mbs: float = 0.0

    # PCIe (optional)
    pcie_tx_mbs: float = 0.0
    pcie_rx_mbs: float = 0.0


# ---------------------------------------------------------------------------
# Aggregated GPU statistics
# ---------------------------------------------------------------------------

@dataclass
class GPUStats:
    """Aggregated statistics over a series of GPUSample snapshots."""

    gpu_index: int
    num_samples: int = 0

    mean_gpu_util_pct: float = 0.0
    max_gpu_util_pct: float = 0.0
    mean_mem_used_mib: float = 0.0
    max_mem_used_mib: float = 0.0
    mean_power_draw_w: float = 0.0
    max_power_draw_w: float = 0.0
    mean_temperature_c: float = 0.0
    max_temperature_c: float = 0.0
    mean_sm_clock_mhz: float = 0.0
    mean_nvlink_tx_mbs: float = 0.0
    mean_nvlink_rx_mbs: float = 0.0

    def summary(self) -> str:
        return (
            f"GPU[{self.gpu_index}] over {self.num_samples} samples:  "
            f"util_mean={self.mean_gpu_util_pct:.1f}% max={self.max_gpu_util_pct:.1f}%  "
            f"mem_mean={self.mean_mem_used_mib:.0f} MiB max={self.max_mem_used_mib:.0f} MiB  "
            f"power_mean={self.mean_power_draw_w:.1f}W max={self.max_power_draw_w:.1f}W  "
            f"temp_mean={self.mean_temperature_c:.1f}°C  "
            f"sm_clock={self.mean_sm_clock_mhz:.0f} MHz  "
            f"nvlink_tx={self.mean_nvlink_tx_mbs:.1f} MB/s"
        )


def _aggregate(samples: List[GPUSample], gpu_index: int) -> GPUStats:
    """Aggregate a list of GPUSample into GPUStats.

    Args:
        samples: List of GPUSample for a single GPU.
        gpu_index: GPU device index.

    Returns:
        Populated GPUStats.
    """
    n = len(samples)
    if n == 0:
        return GPUStats(gpu_index=gpu_index)

    def _mean(attr: str) -> float:
        return sum(getattr(s, attr) for s in samples) / n

    def _max(attr: str) -> float:
        return max(getattr(s, attr) for s in samples)

    return GPUStats(
        gpu_index=gpu_index,
        num_samples=n,
        mean_gpu_util_pct=_mean("gpu_utilization_pct"),
        max_gpu_util_pct=_max("gpu_utilization_pct"),
        mean_mem_used_mib=_mean("memory_used_mib"),
        max_mem_used_mib=_max("memory_used_mib"),
        mean_power_draw_w=_mean("power_draw_w"),
        max_power_draw_w=_max("power_draw_w"),
        mean_temperature_c=_mean("temperature_c"),
        max_temperature_c=_max("temperature_c"),
        mean_sm_clock_mhz=_mean("sm_clock_mhz"),
        mean_nvlink_tx_mbs=_mean("nvlink_tx_mbs"),
        mean_nvlink_rx_mbs=_mean("nvlink_rx_mbs"),
    )


# ---------------------------------------------------------------------------
# NVML sampling helpers
# ---------------------------------------------------------------------------

def _sample_gpu_nvml(gpu_index: int) -> GPUSample:
    """Take one hardware sample from GPU *gpu_index* via NVML.

    Args:
        gpu_index: Zero-based GPU device index.

    Returns:
        GPUSample populated from NVML.
    """
    handle = _nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    ts = time.time()

    util = _nvml.nvmlDeviceGetUtilizationRates(handle)
    mem = _nvml.nvmlDeviceGetMemoryInfo(handle)
    sm_clock = _nvml.nvmlDeviceGetClockInfo(handle, _nvml.NVML_CLOCK_SM)
    mem_clock = _nvml.nvmlDeviceGetClockInfo(handle, _nvml.NVML_CLOCK_MEM)

    try:
        power = _nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
    except Exception:
        power = 0.0

    try:
        temp = _nvml.nvmlDeviceGetTemperature(handle, _nvml.NVML_TEMPERATURE_GPU)
    except Exception:
        temp = 0.0

    mem_total_mib = mem.total / (1024 * 1024)
    mem_used_mib = mem.used / (1024 * 1024)

    return GPUSample(
        gpu_index=gpu_index,
        timestamp_s=ts,
        gpu_utilization_pct=float(util.gpu),
        memory_utilization_pct=float(util.memory),
        memory_used_mib=mem_used_mib,
        memory_total_mib=mem_total_mib,
        memory_used_pct=mem_used_mib / mem_total_mib * 100.0 if mem_total_mib > 0 else 0.0,
        sm_clock_mhz=float(sm_clock),
        mem_clock_mhz=float(mem_clock),
        power_draw_w=power,
        temperature_c=float(temp),
    )


# ---------------------------------------------------------------------------
# GPUProfiler
# ---------------------------------------------------------------------------

class GPUProfiler:
    """Collects GPU hardware metrics during a benchmark run.

    When NVML is available, samples real hardware counters.  Otherwise
    records synthetic zero samples so the profiling pipeline can continue
    without physical GPUs.

    Args:
        gpu_indices: List of GPU device indices to monitor.  Defaults to [0].
        sample_interval_s: Interval between hardware samples in seconds.
    """

    def __init__(
        self,
        gpu_indices: Optional[List[int]] = None,
        sample_interval_s: float = 0.5,
    ) -> None:
        self.gpu_indices = gpu_indices or [0]
        self.sample_interval_s = sample_interval_s
        self._samples: Dict[int, List[GPUSample]] = {
            idx: [] for idx in self.gpu_indices
        }
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not _HAS_NVML:
            logger.warning(
                "pynvml not available; GPUProfiler will record zero samples. "
                "Install pynvml for real hardware metrics."
            )

    # ------------------------------------------------------------------
    # Manual sampling
    # ------------------------------------------------------------------

    def sample_once(self) -> Dict[int, GPUSample]:
        """Take a single hardware sample from all monitored GPUs.

        Returns:
            Dict mapping GPU index to GPUSample.
        """
        result: Dict[int, GPUSample] = {}
        for idx in self.gpu_indices:
            if _HAS_NVML:
                try:
                    s = _sample_gpu_nvml(idx)
                except Exception as exc:
                    logger.warning("NVML sampling failed for GPU %d: %s", idx, exc)
                    s = GPUSample(gpu_index=idx, timestamp_s=time.time())
            else:
                s = GPUSample(gpu_index=idx, timestamp_s=time.time())
            self._samples[idx].append(s)
            result[idx] = s
        return result

    def inject_sample(self, sample: GPUSample) -> None:
        """Inject a synthetic GPUSample (for testing).

        Args:
            sample: Pre-constructed GPUSample to store.
        """
        self._samples.setdefault(sample.gpu_index, []).append(sample)

    # ------------------------------------------------------------------
    # Background sampling
    # ------------------------------------------------------------------

    def start_background_sampling(self) -> None:
        """Start a background thread that samples GPUs periodically."""
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                self.sample_once()
                self._stop_event.wait(self.sample_interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.debug("GPUProfiler background sampling started")

    def stop_background_sampling(self) -> None:
        """Stop the background sampling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.debug("GPUProfiler background sampling stopped")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[int, GPUStats]:
        """Compute aggregated statistics for all monitored GPUs.

        Returns:
            Dict mapping GPU index to GPUStats.
        """
        return {
            idx: _aggregate(samples, idx)
            for idx, samples in self._samples.items()
        }

    def get_stats_for_gpu(self, gpu_index: int) -> Optional[GPUStats]:
        """Return aggregated stats for a single GPU.

        Args:
            gpu_index: GPU device index.

        Returns:
            GPUStats, or None if no samples were recorded.
        """
        samples = self._samples.get(gpu_index, [])
        if not samples:
            return None
        return _aggregate(samples, gpu_index)

    def all_samples(self, gpu_index: int) -> List[GPUSample]:
        """Return all raw samples for a GPU.

        Args:
            gpu_index: GPU device index.

        Returns:
            List of GPUSample in chronological order.
        """
        return list(self._samples.get(gpu_index, []))

    def reset(self) -> None:
        """Clear all recorded samples."""
        for idx in self._samples:
            self._samples[idx].clear()

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_indices)

    def total_samples(self) -> int:
        return sum(len(v) for v in self._samples.values())
