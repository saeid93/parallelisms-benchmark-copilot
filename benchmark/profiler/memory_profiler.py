"""GPU / CPU memory profiler for LLM serving workloads.

Inspired by:
* PyTorch CUDA Memory Snapshot
* Megatron-LM memory analysis utilities
* vLLM block-manager memory accounting

Tracks allocation events, fragmentation, peak usage, tensor-level
attribution, and KV-cache memory lifecycle over time.  The profiler
runs in the background (or can be driven manually) and produces a
timeline that can be exported as Chrome Trace JSON, time-series CSV,
or summary statistics for automated comparison.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemoryEventKind(str, Enum):
    """Kinds of memory events we record."""

    ALLOC = "alloc"
    FREE = "free"
    RESIZE = "resize"
    OOM = "oom"
    CACHE_ALLOC = "cache_alloc"
    CACHE_FREE = "cache_free"
    SWAP_OUT = "swap_out"
    SWAP_IN = "swap_in"
    DEFRAG = "defrag"


class MemoryPool(str, Enum):
    """Logical memory pools tracked independently."""

    MODEL_WEIGHTS = "model_weights"
    KV_CACHE = "kv_cache"
    ACTIVATIONS = "activations"
    WORKSPACE = "workspace"
    COMMUNICATION = "communication"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryEvent:
    """A single memory allocation / free event."""

    kind: MemoryEventKind
    pool: MemoryPool
    timestamp_us: float
    size_bytes: int
    address: int = 0
    device_index: int = 0
    tensor_name: str = ""
    request_id: str = ""
    layer_index: Optional[int] = None
    stack_trace: str = ""

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind.value,
            "pool": self.pool.value,
            "timestamp_us": self.timestamp_us,
            "size_bytes": self.size_bytes,
            "address": self.address,
            "device_index": self.device_index,
            "tensor_name": self.tensor_name,
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "stack_trace": self.stack_trace,
        }


@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of memory usage across pools."""

    timestamp_us: float
    device_index: int = 0
    total_allocated_bytes: int = 0
    total_reserved_bytes: int = 0
    total_free_bytes: int = 0
    pool_usage: Dict[str, int] = field(default_factory=dict)
    fragmentation_ratio: float = 0.0
    num_live_tensors: int = 0
    peak_allocated_bytes: int = 0

    def utilization_pct(self) -> float:
        total = self.total_allocated_bytes + self.total_free_bytes
        if total == 0:
            return 0.0
        return 100.0 * self.total_allocated_bytes / total


@dataclass
class MemoryStats:
    """Aggregated memory statistics over a profiling window."""

    device_index: int = 0
    num_events: int = 0
    peak_allocated_bytes: int = 0
    mean_allocated_bytes: float = 0.0
    peak_kv_cache_bytes: int = 0
    mean_kv_cache_bytes: float = 0.0
    total_alloc_count: int = 0
    total_free_count: int = 0
    total_oom_count: int = 0
    total_swap_out_bytes: int = 0
    total_swap_in_bytes: int = 0
    total_defrag_count: int = 0
    mean_fragmentation: float = 0.0
    peak_fragmentation: float = 0.0
    allocation_rate_per_s: float = 0.0
    duration_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Memory Stats (GPU {self.device_index})",
            f"  Events recorded  : {self.num_events}",
            f"  Peak allocated   : {self.peak_allocated_bytes / (1024**2):.1f} MiB",
            f"  Mean allocated   : {self.mean_allocated_bytes / (1024**2):.1f} MiB",
            f"  Peak KV cache    : {self.peak_kv_cache_bytes / (1024**2):.1f} MiB",
            f"  Mean KV cache    : {self.mean_kv_cache_bytes / (1024**2):.1f} MiB",
            f"  OOM events       : {self.total_oom_count}",
            f"  Swap out total   : {self.total_swap_out_bytes / (1024**2):.1f} MiB",
            f"  Swap in total    : {self.total_swap_in_bytes / (1024**2):.1f} MiB",
            f"  Defrag count     : {self.total_defrag_count}",
            f"  Mean frag ratio  : {self.mean_fragmentation:.3f}",
            f"  Peak frag ratio  : {self.peak_fragmentation:.3f}",
            f"  Alloc rate       : {self.allocation_rate_per_s:.1f} /s",
            f"  Duration         : {self.duration_s:.2f} s",
        ]
        return "\n".join(lines)


@dataclass
class TensorAllocation:
    """Tracks a single live tensor allocation."""

    address: int
    size_bytes: int
    pool: MemoryPool
    tensor_name: str
    alloc_time_us: float
    device_index: int = 0
    request_id: str = ""
    layer_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Memory Profiler
# ---------------------------------------------------------------------------


class MemoryProfiler:
    """Records and analyses GPU / CPU memory allocation patterns.

    Usage::

        mp = MemoryProfiler()
        mp.start_background_snapshots(interval_s=0.1)
        # ... run workload ...
        mp.stop_background_snapshots()
        stats = mp.compute_stats()
        print(stats.summary())
    """

    def __init__(
        self,
        device_indices: Optional[List[int]] = None,
        snapshot_interval_s: float = 0.1,
        total_memory_bytes: int = 80 * (1024**3),
    ) -> None:
        self._device_indices = device_indices or [0]
        self._snapshot_interval = snapshot_interval_s
        self._total_memory_bytes = total_memory_bytes

        # per-device event logs
        self._events: Dict[int, List[MemoryEvent]] = {
            d: [] for d in self._device_indices
        }
        self._snapshots: Dict[int, List[MemorySnapshot]] = {
            d: [] for d in self._device_indices
        }

        # live allocation tracking
        self._live_tensors: Dict[int, Dict[int, TensorAllocation]] = {
            d: {} for d in self._device_indices
        }
        self._current_allocated: Dict[int, int] = {
            d: 0 for d in self._device_indices
        }
        self._pool_allocated: Dict[int, Dict[str, int]] = {
            d: {p.value: 0 for p in MemoryPool} for d in self._device_indices
        }
        self._peak_allocated: Dict[int, int] = {
            d: 0 for d in self._device_indices
        }

        # background sampling
        self._lock = threading.Lock()
        self._sampling = False
        self._thread: Optional[threading.Thread] = None

    # ---- event recording ----

    def record_event(self, event: MemoryEvent) -> None:
        """Record a memory event and update live tracking."""
        dev = event.device_index
        if dev not in self._events:
            return

        with self._lock:
            self._events[dev].append(event)

            if event.kind in (MemoryEventKind.ALLOC, MemoryEventKind.CACHE_ALLOC):
                self._current_allocated[dev] += event.size_bytes
                pool_key = event.pool.value
                self._pool_allocated[dev][pool_key] = (
                    self._pool_allocated[dev].get(pool_key, 0) + event.size_bytes
                )
                if event.address:
                    self._live_tensors[dev][event.address] = TensorAllocation(
                        address=event.address,
                        size_bytes=event.size_bytes,
                        pool=event.pool,
                        tensor_name=event.tensor_name,
                        alloc_time_us=event.timestamp_us,
                        device_index=dev,
                        request_id=event.request_id,
                        layer_index=event.layer_index,
                    )
                if self._current_allocated[dev] > self._peak_allocated[dev]:
                    self._peak_allocated[dev] = self._current_allocated[dev]

            elif event.kind in (
                MemoryEventKind.FREE,
                MemoryEventKind.CACHE_FREE,
            ):
                freed = min(event.size_bytes, self._current_allocated[dev])
                self._current_allocated[dev] -= freed
                pool_key = event.pool.value
                pool_current = self._pool_allocated[dev].get(pool_key, 0)
                self._pool_allocated[dev][pool_key] = max(
                    0, pool_current - event.size_bytes
                )
                if event.address and event.address in self._live_tensors[dev]:
                    del self._live_tensors[dev][event.address]

            elif event.kind == MemoryEventKind.SWAP_OUT:
                freed = min(event.size_bytes, self._current_allocated[dev])
                self._current_allocated[dev] -= freed
                pool_key = event.pool.value
                pool_current = self._pool_allocated[dev].get(pool_key, 0)
                self._pool_allocated[dev][pool_key] = max(
                    0, pool_current - event.size_bytes
                )

            elif event.kind == MemoryEventKind.SWAP_IN:
                self._current_allocated[dev] += event.size_bytes
                pool_key = event.pool.value
                self._pool_allocated[dev][pool_key] = (
                    self._pool_allocated[dev].get(pool_key, 0) + event.size_bytes
                )
                if self._current_allocated[dev] > self._peak_allocated[dev]:
                    self._peak_allocated[dev] = self._current_allocated[dev]

    def take_snapshot(self, device_index: int = 0) -> MemorySnapshot:
        """Take a point-in-time memory snapshot."""
        with self._lock:
            allocated = self._current_allocated.get(device_index, 0)
            free = max(0, self._total_memory_bytes - allocated)
            live = len(self._live_tensors.get(device_index, {}))

            # fragmentation = 1 - (largest_free_block / total_free)
            # Without real allocator info, approximate from pool spread
            pools = self._pool_allocated.get(device_index, {})
            pool_sizes = [v for v in pools.values() if v > 0]
            if pool_sizes and allocated > 0:
                frag = 1.0 - max(pool_sizes) / allocated
            else:
                frag = 0.0

            snap = MemorySnapshot(
                timestamp_us=time.time() * 1e6,
                device_index=device_index,
                total_allocated_bytes=allocated,
                total_reserved_bytes=self._total_memory_bytes,
                total_free_bytes=free,
                pool_usage=dict(pools),
                fragmentation_ratio=max(0.0, frag),
                num_live_tensors=live,
                peak_allocated_bytes=self._peak_allocated.get(device_index, 0),
            )
            self._snapshots[device_index].append(snap)
            return snap

    # ---- background snapshotting ----

    def start_background_snapshots(
        self, interval_s: Optional[float] = None
    ) -> None:
        """Start periodic snapshotting in a background thread."""
        if self._sampling:
            return
        self._sampling = True
        iv = interval_s or self._snapshot_interval
        self._thread = threading.Thread(
            target=self._snapshot_loop, args=(iv,), daemon=True
        )
        self._thread.start()

    def stop_background_snapshots(self) -> None:
        """Stop background snapshotting."""
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _snapshot_loop(self, interval_s: float) -> None:
        while self._sampling:
            for dev in self._device_indices:
                try:
                    self.take_snapshot(dev)
                except Exception:
                    logger.debug("Snapshot failed for device %d", dev)
            time.sleep(interval_s)

    # ---- analysis ----

    def compute_stats(self, device_index: int = 0) -> MemoryStats:
        """Aggregate events and snapshots into MemoryStats."""
        events = self._events.get(device_index, [])
        snapshots = self._snapshots.get(device_index, [])

        alloc_count = sum(
            1 for e in events if e.kind in (MemoryEventKind.ALLOC, MemoryEventKind.CACHE_ALLOC)
        )
        free_count = sum(
            1 for e in events if e.kind in (MemoryEventKind.FREE, MemoryEventKind.CACHE_FREE)
        )
        oom_count = sum(1 for e in events if e.kind == MemoryEventKind.OOM)
        swap_out = sum(e.size_bytes for e in events if e.kind == MemoryEventKind.SWAP_OUT)
        swap_in = sum(e.size_bytes for e in events if e.kind == MemoryEventKind.SWAP_IN)
        defrag_count = sum(1 for e in events if e.kind == MemoryEventKind.DEFRAG)

        if snapshots:
            alloc_vals = [s.total_allocated_bytes for s in snapshots]
            kv_vals = [s.pool_usage.get(MemoryPool.KV_CACHE.value, 0) for s in snapshots]
            frag_vals = [s.fragmentation_ratio for s in snapshots]
            mean_alloc = sum(alloc_vals) / len(alloc_vals)
            peak_alloc = max(alloc_vals)
            mean_kv = sum(kv_vals) / len(kv_vals)
            peak_kv = max(kv_vals)
            mean_frag = sum(frag_vals) / len(frag_vals)
            peak_frag = max(frag_vals)
        else:
            mean_alloc = float(self._current_allocated.get(device_index, 0))
            peak_alloc = self._peak_allocated.get(device_index, 0)
            mean_kv = 0.0
            peak_kv = 0
            mean_frag = 0.0
            peak_frag = 0.0

        if events:
            ts = [e.timestamp_us for e in events]
            duration_us = max(ts) - min(ts)
        else:
            duration_us = 0.0

        duration_s = duration_us / 1e6
        alloc_rate = alloc_count / duration_s if duration_s > 0 else 0.0

        return MemoryStats(
            device_index=device_index,
            num_events=len(events),
            peak_allocated_bytes=peak_alloc,
            mean_allocated_bytes=mean_alloc,
            peak_kv_cache_bytes=peak_kv,
            mean_kv_cache_bytes=mean_kv,
            total_alloc_count=alloc_count,
            total_free_count=free_count,
            total_oom_count=oom_count,
            total_swap_out_bytes=swap_out,
            total_swap_in_bytes=swap_in,
            total_defrag_count=defrag_count,
            mean_fragmentation=mean_frag,
            peak_fragmentation=peak_frag,
            allocation_rate_per_s=alloc_rate,
            duration_s=duration_s,
        )

    def pool_breakdown(self, device_index: int = 0) -> Dict[str, int]:
        """Return current allocation per memory pool."""
        with self._lock:
            return dict(self._pool_allocated.get(device_index, {}))

    def live_tensor_report(self, device_index: int = 0) -> List[Dict]:
        """Return list of currently live tensors sorted by size desc."""
        with self._lock:
            tensors = list(self._live_tensors.get(device_index, {}).values())
        tensors.sort(key=lambda t: t.size_bytes, reverse=True)
        return [
            {
                "tensor_name": t.tensor_name,
                "size_mib": t.size_bytes / (1024**2),
                "pool": t.pool.value,
                "request_id": t.request_id,
                "layer": t.layer_index,
                "age_us": time.time() * 1e6 - t.alloc_time_us,
            }
            for t in tensors
        ]

    def export_timeline(self, device_index: int = 0) -> List[Dict]:
        """Export memory events as Chrome Trace events."""
        events = self._events.get(device_index, [])
        chrome_events: List[Dict] = []
        for e in events:
            chrome_events.append(
                {
                    "name": f"{e.kind.value}:{e.pool.value}",
                    "cat": "memory",
                    "ph": "i",
                    "ts": e.timestamp_us,
                    "pid": device_index,
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "size_bytes": e.size_bytes,
                        "tensor": e.tensor_name,
                        "request_id": e.request_id,
                    },
                }
            )
        return chrome_events

    def export_snapshot_timeseries(
        self, device_index: int = 0
    ) -> List[Dict]:
        """Export snapshots as time-series dicts (for CSV / plotting)."""
        snaps = self._snapshots.get(device_index, [])
        return [
            {
                "timestamp_us": s.timestamp_us,
                "allocated_mib": s.total_allocated_bytes / (1024**2),
                "free_mib": s.total_free_bytes / (1024**2),
                "kv_cache_mib": s.pool_usage.get(MemoryPool.KV_CACHE.value, 0)
                / (1024**2),
                "fragmentation": s.fragmentation_ratio,
                "live_tensors": s.num_live_tensors,
                "utilization_pct": s.utilization_pct(),
            }
            for s in snaps
        ]

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            for d in self._device_indices:
                self._events[d].clear()
                self._snapshots[d].clear()
                self._live_tensors[d].clear()
                self._current_allocated[d] = 0
                self._pool_allocated[d] = {p.value: 0 for p in MemoryPool}
                self._peak_allocated[d] = 0
