"""Playback accounting for barge-in.

When the caller interrupts, the model's conversation history must contain only what the
caller actually *heard*, otherwise the agent "remembers" saying things that were cut off.

The tracker models the far end as a continuous jitter buffer: audio we send plays back-to-back
starting no earlier than when it was sent (+ a fixed output latency). At the moment of
interruption we can then work out which chunks finished playing and how far into the
current chunk playback got. A transport that echoes playback marks (Twilio `mark`) corrects
that estimate: until the first mark returns, the clock still applies; after it, a chunk counts
as heard only when its mark was acked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class _Chunk:
    text: str
    dur_ms: float
    start_ms: float


@dataclass
class PlaybackTracker:
    clock: Callable[[], float]          # ms, monotonic
    output_latency_ms: float = 0.0      # network + device delay before the caller hears audio
    _chunks: list[_Chunk] = field(default_factory=list)
    _free_at: float = 0.0
    pending_marks: dict[str, int] = field(default_factory=dict)  # unacked mark name -> chunk index
    acked_through: int = -1             # highest acked chunk index; -1 until the first ack
    acked_ms: float = 0.0               # end time (clock domain) of the furthest acked chunk
    _acked: set[int] = field(default_factory=set)

    def on_sent(self, text: str, dur_ms: float) -> None:
        now = self.clock() + self.output_latency_ms
        start = max(now, self._free_at)
        self._chunks.append(_Chunk(text, dur_ms, start))
        self._free_at = start + dur_ms

    def arm_mark(self, name: str) -> None:
        """Bind a transport playback mark to the chunk just passed to on_sent."""
        if not name or not self._chunks:
            return
        self.pending_marks[name] = len(self._chunks) - 1

    def note_mark(self, name: str) -> None:
        """Twilio (or another transport) echoed `name`: that chunk has been played.

        Unknown names are ignored. After `clear`, the tracker is reset, so a late echo for a
        flushed chunk must not extend what the caller heard.
        """
        idx = self.pending_marks.pop(name, None)
        if idx is None or idx >= len(self._chunks):
            return
        self._acked.add(idx)
        self.acked_through = max(self._acked)
        self.acked_ms = max(self._chunks[i].start_ms + self._chunks[i].dur_ms for i in self._acked)

    def ends_at(self) -> float:
        """Clock-estimate of when the audio sent so far finishes playing."""
        return self._free_at

    def _trust_marks(self) -> bool:
        return bool(self._acked)

    def buffered_ms(self) -> float:
        """Audio already sent but not yet played (how far ahead of the listener we are)."""
        if self._trust_marks():
            return sum(c.dur_ms for i, c in enumerate(self._chunks) if i not in self._acked)
        return max(0.0, self._free_at - self.clock())

    def is_playing(self) -> bool:
        if self.pending_marks or self._acked:
            return bool(self.pending_marks)   # an armed mark that has not been acked
        return bool(self._chunks) and self.clock() < self._free_at

    def reset(self) -> None:
        self._chunks.clear()
        self._free_at = 0.0
        self.pending_marks.clear()
        self._acked.clear()
        self.acked_through = -1
        self.acked_ms = 0.0

    def heard(self) -> tuple[str, bool]:
        """Return (text the caller heard so far, whether the last chunk was cut mid-way)."""
        if self._trust_marks():
            parts = [c.text for i, c in enumerate(self._chunks) if i in self._acked and c.text]
            cut = len(self._acked) < len(self._chunks)
            return " ".join(parts).strip(), cut
        now = self.clock()
        parts: list[str] = []
        cut = False
        for c in self._chunks:
            if now >= c.start_ms + c.dur_ms:
                parts.append(c.text)
            elif now > c.start_ms:
                frac = (now - c.start_ms) / c.dur_ms
                words = c.text.split()
                k = int(len(words) * frac)
                if k > 0:
                    parts.append(" ".join(words[:k]))
                cut = True
                break
            else:
                cut = True  # queued but not yet audible
                break
        return " ".join(parts).strip(), cut


def trim_history(messages: list[dict], heard_text: str, cut: bool) -> list[dict]:
    """Replace the last assistant message with what was actually heard.

    If nothing was heard the assistant turn is dropped entirely.
    """
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            if heard_text:
                content = heard_text + (" [interrupted]" if cut else "")
                msgs[i] = {**msgs[i], "content": content}
            else:
                del msgs[i]
            break
    return msgs
