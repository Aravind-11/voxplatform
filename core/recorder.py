"""Call recording: stereo WAV, caller on the left channel and the agent on the right, both at 16 kHz, aligned to
the wall clock. The agent track is laid out as it plays: audio is queued back to back and an interruption
truncates what had not played yet. Recording is opt-in per agent and needs the callers' consent (see docs)."""
from __future__ import annotations

import os
import time
import wave
from typing import Callable, Optional

import numpy as np

from .audio import resample

RATE = 16000


class CallRecorder:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock                 # seconds
        self.t0 = clock()
        self.left: list[np.ndarray] = []   # caller
        self.right = np.zeros(0, dtype=np.int16)
        self._left_n = 0
        self._agent_end = 0                # sample index where queued agent audio ends

    def _now(self) -> int:
        return int((self.clock() - self.t0) * RATE)

    def mic(self, frame: np.ndarray) -> None:
        self.left.append(np.asarray(frame, dtype=np.int16))
        self._left_n += len(frame)

    def agent(self, pcm: np.ndarray, rate: int) -> None:
        x = resample(np.asarray(pcm, dtype=np.int16), rate, RATE) if rate != RATE else np.asarray(pcm, dtype=np.int16)
        start = max(self._agent_end, self._now())
        if len(self.right) < start:
            self.right = np.concatenate([self.right, np.zeros(start - len(self.right), np.int16)])
        self.right = np.concatenate([self.right[:start], x])
        self._agent_end = start + len(x)

    def clear(self) -> None:
        """The far end flushed its buffer: whatever had not played by now never will."""
        now = self._now()
        if self._agent_end > now:
            self.right = self.right[:now]
            self._agent_end = now

    def stereo(self) -> np.ndarray:
        l = np.concatenate(self.left) if self.left else np.zeros(0, np.int16)
        n = max(len(l), len(self.right))
        out = np.zeros((n, 2), dtype=np.int16)
        out[:len(l), 0] = l
        out[:len(self.right), 1] = self.right
        return out

    def save(self, path: str) -> float:
        data = self.stereo()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(data.astype("<i2").tobytes())
        os.replace(tmp, path)
        return len(data) / RATE


def write_wav16(path: str, audio: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.asarray(audio, dtype="<i2").tobytes())
