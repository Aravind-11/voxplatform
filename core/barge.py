"""Playback accounting for barge-in.

When the caller interrupts, the model's conversation history must contain only what the
caller actually *heard*, otherwise the agent "remembers" saying things that were cut off.

The tracker models the far end as a continuous jitter buffer: audio we send plays back-to-back
starting no earlier than when it was sent (+ a fixed output latency). At the moment of
interruption we can then work out which chunks finished playing and how far into the
current chunk playback got. Transports with real playback acknowledgements (Twilio `mark`)
can later correct the model; the estimate alone is already usable.
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

    def on_sent(self, text: str, dur_ms: float) -> None:
        now = self.clock() + self.output_latency_ms
        start = max(now, self._free_at)
        self._chunks.append(_Chunk(text, dur_ms, start))
        self._free_at = start + dur_ms

    def buffered_ms(self) -> float:
        """Audio already sent but not yet played (how far ahead of the listener we are)."""
        return max(0.0, self._free_at - self.clock())

    def is_playing(self) -> bool:
        return bool(self._chunks) and self.clock() < self._free_at

    def reset(self) -> None:
        self._chunks.clear()
        self._free_at = 0.0

    def heard(self) -> tuple[str, bool]:
        """Return (text the caller heard so far, whether the last chunk was cut mid-way)."""
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
