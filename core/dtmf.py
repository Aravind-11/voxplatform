"""DTMF tone generation (ITU-T Q.23 frequencies) for the playDtmf tool."""
from __future__ import annotations

import numpy as np

LOW = {"1": 697, "2": 697, "3": 697, "A": 697, "4": 770, "5": 770, "6": 770, "B": 770,
       "7": 852, "8": 852, "9": 852, "C": 852, "*": 941, "0": 941, "#": 941, "D": 941}
HIGH = {"1": 1209, "4": 1209, "7": 1209, "*": 1209, "2": 1336, "5": 1336, "8": 1336, "0": 1336,
        "3": 1477, "6": 1477, "9": 1477, "#": 1477, "A": 1633, "B": 1633, "C": 1633, "D": 1633}


def tone(digit: str, rate: int = 8000, ms: int = 120, amp: float = 0.35) -> np.ndarray:
    d = digit.upper()
    t = np.arange(int(rate * ms / 1000)) / rate
    x = np.sin(2 * np.pi * LOW[d] * t) + np.sin(2 * np.pi * HIGH[d] * t)
    ramp = min(len(x) // 8, int(rate * 0.005))             # 5 ms fade avoids clicks
    if ramp:
        x[:ramp] *= np.linspace(0, 1, ramp)
        x[-ramp:] *= np.linspace(1, 0, ramp)
    return (x / 2 * amp * 32767).astype(np.int16)


def sequence(digits: str, rate: int = 8000, tone_ms: int = 120, gap_ms: int = 80) -> np.ndarray:
    """Digits 0-9 * # A-D; 'w' is a 500 ms pause."""
    gap = np.zeros(int(rate * gap_ms / 1000), np.int16)
    parts: list[np.ndarray] = []
    for ch in digits:
        if ch in "wW":
            parts.append(np.zeros(int(rate * 0.5), np.int16))
        else:
            parts += [tone(ch, rate, tone_ms), gap]
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def detect(pcm: np.ndarray, rate: int = 8000) -> str:
    """Goertzel-based detector for one tone burst; used by tests to prove the generated tones decode."""
    n = len(pcm)
    x = pcm.astype(np.float64)

    def power(f: float) -> float:
        k = 2 * np.cos(2 * np.pi * f / rate)
        s1 = s2 = 0.0
        for v in x:
            s1, s2 = v + k * s1 - s2, s1
        return s1 * s1 + s2 * s2 - k * s1 * s2

    lo = max(set(LOW.values()), key=power)
    hi = max(set(HIGH.values()), key=power)
    for d in LOW:
        if LOW[d] == lo and HIGH[d] == hi:
            return d
    return ""
