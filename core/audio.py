"""Audio helpers: G.711 mu-law codec and streaming resampling.

Telephony edge audio is 8 kHz mu-law. Internally we use 16 kHz mono int16 PCM for
VAD / the speech model and whatever rate the TTS emits (Kokoro: 24 kHz) for output.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

_BIAS = 0x84
_CLIP = 32635


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM samples to G.711 mu-law bytes (vectorised)."""
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0).astype(np.int32)
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    # exponent = position of the highest set bit above bit 7 (0..7)
    exponent = np.clip(np.floor(np.log2(x)).astype(np.int32) - 7, 0, 7)
    mantissa = (x >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to int16 PCM samples."""
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """One-shot polyphase resample of int16 or float audio. Returns same dtype family."""
    if src_rate == dst_rate:
        return x
    g = gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    y = resample_poly(x.astype(np.float64), up, down)
    if x.dtype == np.int16:
        return np.clip(np.rint(y), -32768, 32767).astype(np.int16)
    return y.astype(x.dtype)


class StreamResampler:
    """Chunked polyphase resampler without chunk-boundary artifacts.

    Keeps `ctx` samples of left context and withholds `lookahead` samples of right context
    so every emitted sample is computed with real neighbours on both sides. The cost is
    `lookahead` input samples of latency (96 samples = 6 ms at 16 kHz, 4 ms at 24 kHz).
    Call flush() at end of stream to emit the withheld tail.
    """

    CTX = 96  # multiples of 1,2,3,6 so integer ratios divide evenly

    def __init__(self, src_rate: int, dst_rate: int):
        g = gcd(src_rate, dst_rate)
        self.up, self.down = dst_rate // g, src_rate // g
        self.identity = src_rate == dst_rate
        self._ctx = np.zeros(0, dtype=np.float64)
        self._buf = np.zeros(0, dtype=np.float64)
        self._total_in = 0
        self._total_out = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.identity:
            return x
        self._total_in += len(x)
        self._buf = np.concatenate([self._buf, x.astype(np.float64)])
        n_emit = ((len(self._buf) - self.CTX) // self.down) * self.down
        if n_emit <= 0:
            return np.zeros(0, dtype=np.int16)
        out = self._emit(n_emit)
        self._total_out += len(out)
        return out

    def flush(self) -> np.ndarray:
        """Emit the withheld tail; total output length matches a one-shot resample."""
        if self.identity or len(self._buf) == 0:
            return np.zeros(0, dtype=np.int16)
        pad = (-len(self._buf)) % self.down  # zero-pad like the one-shot filter does
        self._buf = np.concatenate([self._buf, np.zeros(pad)])
        out = self._emit(len(self._buf))
        want = -(-self._total_in * self.up // self.down)  # ceil
        out = out[: max(0, want - self._total_out)]
        self._total_out += len(out)
        self._buf = np.zeros(0, dtype=np.float64)
        return out

    def _emit(self, n_emit: int) -> np.ndarray:
        data = np.concatenate([self._ctx, self._buf])
        y = resample_poly(data, self.up, self.down)
        start = len(self._ctx) * self.up // self.down
        end = start + n_emit * self.up // self.down
        out = y[start:end]
        self._ctx = data[: len(self._ctx) + n_emit][-self.CTX :]
        self._buf = self._buf[n_emit:]
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def pcm_duration_ms(n_samples: int, rate: int) -> float:
    return 1000.0 * n_samples / rate
