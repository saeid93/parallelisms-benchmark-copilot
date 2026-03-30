"""Tokenizer and preprocessing profiler for LLM serving.

Inspired by:
* HuggingFace Tokenizer Benchmarks
* vLLM tokenizer integration overhead analysis
* "Efficient Tokenization for Neural Machine Translation" research

Profiles tokenization throughput, detokenization latency,
vocabulary utilization, and prompt template overhead to identify
preprocessing bottlenecks in serving pipelines.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TokenizerType(str, Enum):
    """Common tokenizer implementations."""

    SENTENCEPIECE = "sentencepiece"
    TIKTOKEN = "tiktoken"
    HF_FAST = "hf_fast"
    HF_SLOW = "hf_slow"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class TokenizationEvent:
    """A single tokenization / detokenization timing."""

    request_id: str
    direction: str  # "encode" or "decode"
    start_us: float
    end_us: float
    num_chars: int = 0
    num_tokens: int = 0
    tokenizer_type: str = "unknown"
    is_batched: bool = False
    batch_size: int = 1

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def chars_per_second(self) -> float:
        dur_s = self.duration_us / 1e6
        return self.num_chars / dur_s if dur_s > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        dur_s = self.duration_us / 1e6
        return self.num_tokens / dur_s if dur_s > 0 else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.num_chars / self.num_tokens if self.num_tokens > 0 else 0.0


@dataclass
class VocabUtilization:
    """Vocabulary utilization analysis."""

    vocab_size: int = 0
    unique_tokens_seen: int = 0
    total_tokens_processed: int = 0
    utilization_pct: float = 0.0
    top_tokens: List[Dict] = field(default_factory=list)
    rare_token_pct: float = 0.0  # tokens appearing only once
    special_token_pct: float = 0.0


@dataclass
class PromptTemplateOverhead:
    """Overhead from prompt templating / chat formatting."""

    template_name: str = ""
    original_tokens: int = 0
    templated_tokens: int = 0
    overhead_tokens: int = 0
    overhead_pct: float = 0.0
    template_latency_us: float = 0.0


@dataclass
class TokenizerStats:
    """Aggregated tokenizer performance statistics."""

    encode_count: int = 0
    decode_count: int = 0
    mean_encode_us: float = 0.0
    p50_encode_us: float = 0.0
    p99_encode_us: float = 0.0
    mean_decode_us: float = 0.0
    p50_decode_us: float = 0.0
    p99_decode_us: float = 0.0
    mean_encode_throughput_tps: float = 0.0
    mean_decode_throughput_tps: float = 0.0
    total_encode_time_us: float = 0.0
    total_decode_time_us: float = 0.0
    mean_chars_per_token: float = 0.0
    vocab_utilization: Optional[VocabUtilization] = None
    prompt_overhead: Optional[PromptTemplateOverhead] = None

    def summary(self) -> str:
        lines = [
            "Tokenizer Profile",
            f"  Encode calls   : {self.encode_count}",
            f"  Decode calls   : {self.decode_count}",
            f"  Encode mean    : {self.mean_encode_us:.1f} us (p99: {self.p99_encode_us:.1f} us)",
            f"  Decode mean    : {self.mean_decode_us:.1f} us (p99: {self.p99_decode_us:.1f} us)",
            f"  Encode thruput : {self.mean_encode_throughput_tps:.0f} tok/s",
            f"  Decode thruput : {self.mean_decode_throughput_tps:.0f} tok/s",
            f"  Total encode   : {self.total_encode_time_us / 1e3:.1f} ms",
            f"  Total decode   : {self.total_decode_time_us / 1e3:.1f} ms",
            f"  Chars/token    : {self.mean_chars_per_token:.2f}",
        ]
        if self.vocab_utilization:
            vu = self.vocab_utilization
            lines.append(f"  Vocab util     : {vu.utilization_pct:.1f}% ({vu.unique_tokens_seen}/{vu.vocab_size})")
        if self.prompt_overhead:
            po = self.prompt_overhead
            lines.append(f"  Prompt overhead: {po.overhead_pct:.1f}% (+{po.overhead_tokens} tokens)")
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


class TokenizerProfiler:
    """Profiles tokenization and detokenization performance.

    Usage::

        tp = TokenizerProfiler(vocab_size=32000)
        tp.record(TokenizationEvent(...))
        tp.record_prompt_template(original_tokens=100, templated_tokens=130, ...)
        stats = tp.compute_stats()
        print(stats.summary())
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        self._vocab_size = vocab_size
        self._events: List[TokenizationEvent] = []
        self._token_counts: Dict[int, int] = {}  # token_id -> count
        self._prompt_overheads: List[PromptTemplateOverhead] = []

    def record(self, event: TokenizationEvent) -> None:
        """Record a tokenization event."""
        self._events.append(event)

    def record_batch(self, events: List[TokenizationEvent]) -> None:
        """Record multiple events."""
        self._events.extend(events)

    def record_token_ids(self, token_ids: List[int]) -> None:
        """Record token IDs for vocabulary utilization analysis."""
        for tid in token_ids:
            self._token_counts[tid] = self._token_counts.get(tid, 0) + 1

    def record_prompt_template(
        self,
        original_tokens: int,
        templated_tokens: int,
        template_name: str = "",
        latency_us: float = 0.0,
    ) -> None:
        """Record prompt template overhead."""
        overhead = templated_tokens - original_tokens
        pct = 100.0 * overhead / original_tokens if original_tokens > 0 else 0.0
        self._prompt_overheads.append(
            PromptTemplateOverhead(
                template_name=template_name,
                original_tokens=original_tokens,
                templated_tokens=templated_tokens,
                overhead_tokens=overhead,
                overhead_pct=pct,
                template_latency_us=latency_us,
            )
        )

    def _compute_vocab_util(self) -> Optional[VocabUtilization]:
        if not self._token_counts:
            return None

        total = sum(self._token_counts.values())
        unique = len(self._token_counts)
        rare = sum(1 for c in self._token_counts.values() if c == 1)

        # top 10 tokens
        top = sorted(self._token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_list = [{"token_id": tid, "count": cnt, "pct": 100.0 * cnt / total} for tid, cnt in top]

        return VocabUtilization(
            vocab_size=self._vocab_size,
            unique_tokens_seen=unique,
            total_tokens_processed=total,
            utilization_pct=100.0 * unique / self._vocab_size if self._vocab_size > 0 else 0.0,
            top_tokens=top_list,
            rare_token_pct=100.0 * rare / unique if unique > 0 else 0.0,
        )

    def compute_stats(self) -> TokenizerStats:
        """Compute aggregated tokenizer statistics."""
        encodes = [e for e in self._events if e.direction == "encode"]
        decodes = [e for e in self._events if e.direction == "decode"]

        enc_durations = [e.duration_us for e in encodes]
        dec_durations = [e.duration_us for e in decodes]
        enc_tps = [e.tokens_per_second for e in encodes]
        dec_tps = [e.tokens_per_second for e in decodes]
        cpt = [e.chars_per_token for e in self._events if e.num_tokens > 0]

        # prompt template
        po = None
        if self._prompt_overheads:
            avg_overhead = statistics.mean([p.overhead_tokens for p in self._prompt_overheads])
            avg_pct = statistics.mean([p.overhead_pct for p in self._prompt_overheads])
            avg_latency = statistics.mean([p.template_latency_us for p in self._prompt_overheads])
            po = PromptTemplateOverhead(
                template_name="aggregate",
                original_tokens=0,
                templated_tokens=0,
                overhead_tokens=int(avg_overhead),
                overhead_pct=avg_pct,
                template_latency_us=avg_latency,
            )

        return TokenizerStats(
            encode_count=len(encodes),
            decode_count=len(decodes),
            mean_encode_us=statistics.mean(enc_durations) if enc_durations else 0.0,
            p50_encode_us=_percentile(enc_durations, 50),
            p99_encode_us=_percentile(enc_durations, 99),
            mean_decode_us=statistics.mean(dec_durations) if dec_durations else 0.0,
            p50_decode_us=_percentile(dec_durations, 50),
            p99_decode_us=_percentile(dec_durations, 99),
            mean_encode_throughput_tps=statistics.mean(enc_tps) if enc_tps else 0.0,
            mean_decode_throughput_tps=statistics.mean(dec_tps) if dec_tps else 0.0,
            total_encode_time_us=sum(enc_durations),
            total_decode_time_us=sum(dec_durations),
            mean_chars_per_token=statistics.mean(cpt) if cpt else 0.0,
            vocab_utilization=self._compute_vocab_util(),
            prompt_overhead=po,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._events.clear()
        self._token_counts.clear()
        self._prompt_overheads.clear()

    def __len__(self) -> int:
        return len(self._events)
