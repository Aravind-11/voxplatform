"""Call recording: stereo WAV, caller on the left channel and the agent on the right, both at 16 kHz, aligned to
the wall clock. The agent track is laid out as it plays: audio is queued back to back and an interruption
truncates what had not played yet. Recording is opt-in per agent and needs the callers' consent (see docs).

Pass `path` to append PCM to a WAV as the call happens. Without it, samples stay in memory until `save`.
"""
from __future__ import annotations

import os
import struct
import time
import wave
from typing import Callable, Optional

import numpy as np

from .audio import resample

RATE = 16000
_HEADER = 44
# RIFF and data sizes are uint32. 1073741815 stereo frames overflows the RIFF field.
_MAX_WAV_FRAMES = (2**32 - 1 - 36) // 4


def _wav_header(nframes: int) -> bytes:
    data_bytes = nframes * 4  # stereo int16
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE", b"fmt ",
        16, 1, 2, RATE, RATE * 4, 4, 16, b"data", data_bytes,
    )


class CallRecorder:
    def __init__(self, clock: Callable[[], float] = time.monotonic, path: Optional[str] = None):
        self.clock = clock                 # seconds
        self.t0 = clock()
        self.left: list[np.ndarray] = []   # caller; streaming keeps only the latest frame
        self.right = np.zeros(0, dtype=np.int16)
        self._left_n = 0
        self._agent_end = 0                # sample index where queued agent audio ends
        self._path = path
        self._n = 0                         # stereo frames committed to the file
        self._fh = None
        self._failed = False                # torn write: stop patching until save publishes a consistent file
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "w+b")
            self._write_header()

    def _now(self) -> int:
        return int((self.clock() - self.t0) * RATE)

    def _streaming(self) -> bool:
        return self._path is not None

    def _write_header(self) -> None:
        """RIFF sizes must match bytes already written. At least every 50 mic frames, and on clear and save."""
        if self._fh is None:
            return
        self._fh.seek(0)
        self._fh.write(_wav_header(self._n))
        self._fh.flush()

    def _align_to_disk(self) -> None:
        """Header names only whole frames that were committed and that still fit in a uint32 WAV."""
        if self._fh is None:
            return
        self._fh.flush()
        size = self._fh.seek(0, os.SEEK_END)
        disk_frames = 0 if size <= _HEADER else (size - _HEADER) // 4
        frames = min(self._n, disk_frames, _MAX_WAV_FRAMES)
        self._fh.truncate(_HEADER + frames * 4)
        self._n = frames
        self._write_header()

    def _ensure(self, nframes: int) -> None:
        if nframes <= self._n:
            return
        self._fh.seek(_HEADER + self._n * 4)
        self._fh.write(b"\x00" * ((nframes - self._n) * 4))
        self._n = nframes

    def _patch(self, channel: int, start: int, samples: np.ndarray) -> None:
        """Overwrite one channel (0 caller, 1 agent). The other channel in the same frames stays."""
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        n = int(samples.shape[0])
        end = start + n
        if n and start == self._n:
            blank = np.zeros((n, 2), dtype="<i2")
            blank[:, channel] = samples
            self._fh.seek(_HEADER + self._n * 4)
            self._fh.write(blank.tobytes())
            self._n = end
            return
        self._ensure(end)
        if n == 0:
            return
        self._fh.seek(_HEADER + start * 4)
        stereo = np.frombuffer(self._fh.read(n * 4), dtype="<i2").reshape(-1, 2).copy()
        stereo[:, channel] = samples
        self._fh.seek(_HEADER + start * 4)
        self._fh.write(np.ascontiguousarray(stereo, dtype="<i2").tobytes())

    def mic(self, frame: np.ndarray) -> None:
        x = np.asarray(frame, dtype=np.int16)
        if not self._streaming():
            self.left.append(x)
            self._left_n += len(x)
            return
        if self._failed:
            return
        try:
            self.left = [x]                # the earlier caller audio is in the file, not this list
            self._patch(0, self._left_n, x)
            self._left_n += len(x)
            self._write_header()           # every frame, so at least every 50, and a reader sees matching sizes
        except (OSError, struct.error):
            self._failed = True
            raise

    def agent(self, pcm: np.ndarray, rate: int) -> None:
        x = resample(np.asarray(pcm, dtype=np.int16), rate, RATE) if rate != RATE else np.asarray(pcm, dtype=np.int16)
        start = max(self._agent_end, self._now())
        if self._streaming():
            if self._failed:
                return
            try:
                self._patch(1, start, x)
                self._agent_end = start + len(x)
                self._write_header()
            except (OSError, struct.error):
                self._failed = True
                raise
            return
        if len(self.right) < start:
            self.right = np.concatenate([self.right, np.zeros(start - len(self.right), np.int16)])
        self.right = np.concatenate([self.right[:start], x])
        self._agent_end = start + len(x)

    def clear(self) -> None:
        """The far end flushed its buffer: whatever had not played by now never will."""
        now = self._now()
        if not self._streaming():
            if self._agent_end > now:
                self.right = self.right[:now]
                self._agent_end = now
            return
        if self._agent_end > now:
            new_len = max(self._left_n, now)
            if now < new_len and now < self._n:
                # Caller samples past `now` stay; agent samples that never played do not.
                self._patch(1, now, np.zeros(min(new_len, self._n) - now, dtype=np.int16))
            if new_len < self._n:
                self._fh.truncate(_HEADER + new_len * 4)
                self._n = new_len
            # After the file drop. A failed truncate or zero-patch must leave _agent_end so another clear retries.
            self._agent_end = now
        self._write_header()

    def stereo(self) -> np.ndarray:
        if self._streaming():
            if self._fh is None:
                with open(self._path, "rb") as f:
                    f.seek(_HEADER)
                    raw = f.read()
            else:
                self._write_header()
                self._fh.seek(_HEADER)
                raw = self._fh.read(self._n * 4)
            if not raw:
                return np.zeros((0, 2), dtype=np.int16)
            return np.frombuffer(raw, dtype="<i2").reshape(-1, 2).copy()
        l = np.concatenate(self.left) if self.left else np.zeros(0, np.int16)
        n = max(len(l), len(self.right))
        out = np.zeros((n, 2), dtype=np.int16)
        out[:len(l), 0] = l
        out[:len(self.right), 1] = self.right
        return out

    def save(self, path: str) -> float:
        if not self._streaming():
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
        try:
            self._align_to_disk()
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.abspath(path) != os.path.abspath(self._path or path):
            os.replace(self._path, path)
            self._path = path
        return self._n / RATE


def write_wav16(path: str, audio: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.asarray(audio, dtype="<i2").tobytes())
