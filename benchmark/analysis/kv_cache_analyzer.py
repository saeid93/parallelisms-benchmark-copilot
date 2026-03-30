"""KV cache analyzer for LLM serving.

Inspired by:
* vLLM PagedAttention (Kwon et al. 2023)
* SGLang RadixAttention
* Prefix caching / sharing strategies
* "Efficient Memory Management for Large Language Model Serving with PagedAttention"

Provides detailed KV cache utilization analysis, eviction policy
comparison, prefix sharing efficiency, cache fragmentation metrics,
and optimal block size recommendation.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EvictionPolicy(str, Enum):
    """KV cache eviction policies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    PRIORITY = "priority"
    SIZE_AWARE = "size_aware"


class CacheEventType(str, Enum):
    """KV cache events."""

    ALLOCATE = "allocate"
    FREE = "free"
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    SWAP_OUT = "swap_out"
    SWAP_IN = "swap_in"
    PREFIX_HIT = "prefix_hit"
    PREFIX_MISS = "prefix_miss"
    DEFRAGMENT = "defragment"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class CacheEvent:
    """A single KV cache event."""

    event_type: CacheEventType
    timestamp_us: float
    request_id: str = ""
    num_blocks: int = 0
    num_tokens: int = 0
    layer_index: int = 0
    block_size: int = 16
    prefix_length: int = 0
    eviction_policy: str = ""

    @property
    def memory_bytes(self) -> int:
        """Approximate memory for these blocks (assuming FP16, heads=32, dim=128)."""
        return self.num_blocks * self.block_size * 2 * 32 * 128 * 2  # 2 for K+V


@dataclass
class CacheSnapshot:
    """Point-in-time cache state."""

    timestamp_us: float
    total_blocks: int = 0
    used_blocks: int = 0
    free_blocks: int = 0
    utilization_pct: float = 0.0
    fragmentation_ratio: float = 0.0
    num_sequences: int = 0
    prefix_shared_blocks: int = 0


@dataclass
class PrefixSharingStats:
    """Statistics on prefix cache sharing."""

    total_prefix_lookups: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    hit_rate: float = 0.0
    tokens_saved: int = 0
    compute_savings_pct: float = 0.0
    unique_prefixes: int = 0
    mean_prefix_length: float = 0.0
    max_sharing_factor: int = 0  # max requests sharing a prefix


@dataclass
class BlockSizeAnalysis:
    """Analysis of block size impact on efficiency."""

    block_size: int
    internal_fragmentation_pct: float = 0.0  # wasted space within blocks
    num_blocks_needed: int = 0
    memory_overhead_pct: float = 0.0
    swap_granularity_tokens: int = 0


@dataclass
class KVCacheProfile:
    """Full KV cache analysis results."""

    total_events: int = 0
    total_allocations: int = 0
    total_evictions: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    mean_utilization_pct: float = 0.0
    peak_utilization_pct: float = 0.0
    mean_fragmentation: float = 0.0
    total_swap_out_blocks: int = 0
    total_swap_in_blocks: int = 0
    prefix_sharing: Optional[PrefixSharingStats] = None
    block_size_analysis: List[BlockSizeAnalysis] = field(default_factory=list)
    recommended_block_size: int = 16
    per_layer_utilization: Dict[int, float] = field(default_factory=dict)
    eviction_policy: str = ""
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "KV Cache Analysis",
            f"  Total events      : {self.total_events}",
            f"  Allocations       : {self.total_allocations}",
            f"  Evictions         : {self.total_evictions}",
            f"  Hit rate          : {self.hit_rate:.1%}",
            f"  Miss rate         : {self.miss_rate:.1%}",
            f"  Eviction rate     : {self.eviction_rate:.1%}",
            f"  Mean utilization  : {self.mean_utilization_pct:.1f}%",
            f"  Peak utilization  : {self.peak_utilization_pct:.1f}%",
            f"  Mean fragmentation: {self.mean_fragmentation:.3f}",
            f"  Swap out blocks   : {self.total_swap_out_blocks}",
            f"  Swap in blocks    : {self.total_swap_in_blocks}",
            f"  Rec. block size   : {self.recommended_block_size}",
        ]
        if self.prefix_sharing:
            ps = self.prefix_sharing
            lines.append(f"  Prefix hit rate   : {ps.hit_rate:.1%}")
            lines.append(f"  Tokens saved      : {ps.tokens_saved}")
            lines.append(f"  Compute saved     : {ps.compute_savings_pct:.1f}%")
        if self.block_size_analysis:
            lines.append("  Block size comparison:")
            for bs in self.block_size_analysis:
                lines.append(
                    f"    {bs.block_size}: frag={bs.internal_fragmentation_pct:.1f}%, "
                    f"blocks={bs.num_blocks_needed}, overhead={bs.memory_overhead_pct:.1f}%"
                )
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class KVCacheAnalyzer:
    """Analyses KV cache utilization and efficiency.

    Usage::

        ka = KVCacheAnalyzer(total_blocks=1024, block_size=16)
        ka.record(CacheEvent(...))
        ka.record_snapshot(CacheSnapshot(...))
        profile = ka.analyse()
        print(profile.summary())
    """

    def __init__(
        self,
        total_blocks: int = 1024,
        block_size: int = 16,
        num_layers: int = 32,
    ) -> None:
        self._total_blocks = total_blocks
        self._block_size = block_size
        self._num_layers = num_layers
        self._events: List[CacheEvent] = []
        self._snapshots: List[CacheSnapshot] = []
        self._prefix_lookups: List[CacheEvent] = []

    def record(self, event: CacheEvent) -> None:
        self._events.append(event)
        if event.event_type in (CacheEventType.PREFIX_HIT, CacheEventType.PREFIX_MISS):
            self._prefix_lookups.append(event)

    def record_batch(self, events: List[CacheEvent]) -> None:
        for e in events:
            self.record(e)

    def record_snapshot(self, snapshot: CacheSnapshot) -> None:
        self._snapshots.append(snapshot)

    def _analyse_prefix_sharing(self) -> Optional[PrefixSharingStats]:
        if not self._prefix_lookups:
            return None

        hits = sum(1 for e in self._prefix_lookups if e.event_type == CacheEventType.PREFIX_HIT)
        misses = sum(1 for e in self._prefix_lookups if e.event_type == CacheEventType.PREFIX_MISS)
        total = hits + misses
        tokens_saved = sum(
            e.prefix_length for e in self._prefix_lookups if e.event_type == CacheEventType.PREFIX_HIT
        )
        total_tokens = sum(e.num_tokens for e in self._events if e.event_type == CacheEventType.ALLOCATE)

        return PrefixSharingStats(
            total_prefix_lookups=total,
            prefix_hits=hits,
            prefix_misses=misses,
            hit_rate=hits / total if total > 0 else 0.0,
            tokens_saved=tokens_saved,
            compute_savings_pct=(
                100.0 * tokens_saved / total_tokens if total_tokens > 0 else 0.0
            ),
        )

    def _analyse_block_sizes(self) -> List[BlockSizeAnalysis]:
        """Compare different block sizes using recorded token counts."""
        token_counts = [
            e.num_tokens for e in self._events if e.event_type == CacheEventType.ALLOCATE and e.num_tokens > 0
        ]
        if not token_counts:
            return []

        analyses: List[BlockSizeAnalysis] = []
        for bs in [8, 16, 32, 64, 128]:
            total_blocks = 0
            wasted_tokens = 0
            for tc in token_counts:
                blocks_needed = (tc + bs - 1) // bs
                total_blocks += blocks_needed
                wasted = blocks_needed * bs - tc
                wasted_tokens += wasted

            total_allocated = total_blocks * bs
            frag_pct = 100.0 * wasted_tokens / total_allocated if total_allocated > 0 else 0.0
            # metadata overhead: ~8 bytes per block for pointers
            overhead = 100.0 * (total_blocks * 8) / (total_allocated * 2 * 32 * 128 * 2) if total_allocated > 0 else 0.0

            analyses.append(
                BlockSizeAnalysis(
                    block_size=bs,
                    internal_fragmentation_pct=frag_pct,
                    num_blocks_needed=total_blocks,
                    memory_overhead_pct=overhead,
                    swap_granularity_tokens=bs,
                )
            )

        return analyses

    def analyse(self) -> KVCacheProfile:
        if not self._events:
            return KVCacheProfile()

        allocs = sum(1 for e in self._events if e.event_type == CacheEventType.ALLOCATE)
        evictions = sum(1 for e in self._events if e.event_type == CacheEventType.EVICTION)
        hits = sum(1 for e in self._events if e.event_type == CacheEventType.HIT)
        misses = sum(1 for e in self._events if e.event_type == CacheEventType.MISS)
        swap_out = sum(e.num_blocks for e in self._events if e.event_type == CacheEventType.SWAP_OUT)
        swap_in = sum(e.num_blocks for e in self._events if e.event_type == CacheEventType.SWAP_IN)

        total_accesses = hits + misses
        hit_rate = hits / total_accesses if total_accesses > 0 else 0.0
        miss_rate = misses / total_accesses if total_accesses > 0 else 0.0
        eviction_rate = evictions / allocs if allocs > 0 else 0.0

        # Utilization from snapshots
        utils = [s.utilization_pct for s in self._snapshots]
        frags = [s.fragmentation_ratio for s in self._snapshots]

        # Per-layer utilization
        layer_allocs: Dict[int, int] = {}
        for e in self._events:
            if e.event_type == CacheEventType.ALLOCATE:
                layer_allocs[e.layer_index] = layer_allocs.get(e.layer_index, 0) + e.num_blocks
        total_layer_blocks = sum(layer_allocs.values())
        per_layer_util = {
            layer: 100.0 * blocks / total_layer_blocks
            for layer, blocks in layer_allocs.items()
        } if total_layer_blocks > 0 else {}

        # Block size analysis
        block_analyses = self._analyse_block_sizes()

        # Recommended block size: lowest fragmentation
        rec_bs = self._block_size
        if block_analyses:
            best = min(block_analyses, key=lambda b: b.internal_fragmentation_pct)
            rec_bs = best.block_size

        # Prefix sharing
        prefix = self._analyse_prefix_sharing()

        recs = self._generate_recommendations(hit_rate, eviction_rate, utils, prefix, block_analyses)

        return KVCacheProfile(
            total_events=len(self._events),
            total_allocations=allocs,
            total_evictions=evictions,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            eviction_rate=eviction_rate,
            mean_utilization_pct=statistics.mean(utils) if utils else 0.0,
            peak_utilization_pct=max(utils) if utils else 0.0,
            mean_fragmentation=statistics.mean(frags) if frags else 0.0,
            total_swap_out_blocks=swap_out,
            total_swap_in_blocks=swap_in,
            prefix_sharing=prefix,
            block_size_analysis=block_analyses,
            recommended_block_size=rec_bs,
            per_layer_utilization=per_layer_util,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        hit_rate: float,
        eviction_rate: float,
        utils: List[float],
        prefix: Optional[PrefixSharingStats],
        block_analyses: List[BlockSizeAnalysis],
    ) -> List[str]:
        recs: List[str] = []

        if eviction_rate > 0.1:
            recs.append(
                f"High eviction rate ({eviction_rate:.1%}). "
                "Consider increasing GPU memory utilization or reducing max_num_seqs."
            )

        if utils and statistics.mean(utils) > 90:
            recs.append(
                "KV cache utilization is near capacity. "
                "Consider FP8 KV cache to double effective capacity."
            )

        if hit_rate < 0.5 and (prefix and prefix.total_prefix_lookups > 0):
            recs.append(
                f"Low cache hit rate ({hit_rate:.1%}). "
                "Enable prefix caching if workload has repeated prefixes."
            )

        if prefix and prefix.hit_rate > 0.8:
            recs.append(
                f"Excellent prefix cache hit rate ({prefix.hit_rate:.1%}). "
                "Prefix caching is highly effective for this workload."
            )

        if block_analyses:
            current_frag = next(
                (b.internal_fragmentation_pct for b in block_analyses if b.block_size == self._block_size),
                0.0,
            )
            best = min(block_analyses, key=lambda b: b.internal_fragmentation_pct)
            if best.block_size != self._block_size and current_frag > best.internal_fragmentation_pct + 5:
                recs.append(
                    f"Block size {best.block_size} would reduce fragmentation from "
                    f"{current_frag:.1f}% to {best.internal_fragmentation_pct:.1f}%."
                )

        return recs

    def reset(self) -> None:
        self._events.clear()
        self._snapshots.clear()
        self._prefix_lookups.clear()
