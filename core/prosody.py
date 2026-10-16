"""Acoustic end-of-turn cues, no model and no GPU: how the caller's voice ends, not just that it stopped.

Cues, measured on the last ~0.5 s of speech before the pause:
  * pitch fall      declarative endings drop in pitch; flat or rising endings often continue
  * energy decay    people soften as they finish; a cut-off clause stops at full volume
  * filler hum      a steady, level voiced sound ("um", "uh", "so...") means they are still thinking
Weights are hand-set priors, not fitted. Limits: it cannot tell a rising question (done) from a rising
"and then..." (not done), and it knows nothing about the words. tests/bringup/eot_eval.py reports how well it
actually separates finished from unfinished speech next to the model-based verdict.
"""
from __future__ import annotations

import numpy as np

SR = 16000
WIN, HOP = 1024, 256            # 64 ms window, 16 ms hop
FMIN, FMAX = 70.0, 400.0


def _f0(x: np.ndarray) -> float:
    """Autocorrelation pitch of one window in Hz, 0.0 if unvoiced."""
    x = x - x.mean()
    e = float(x @ x)
    if e < 1e-6:
        return 0.0
    n = len(x)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    lo, hi = int(SR / FMAX), int(SR / FMIN)
    k = lo + int(np.argmax(ac[lo:hi]))
    return SR / k if ac[k] / ac[0] > 0.35 else 0.0


def features(audio: np.ndarray) -> dict | None:
    """Pitch slope (semitones/s), energy decay ratio and filler flag for the end of the speech; None if too short."""
    x = audio.astype(np.float32) / 32768.0
    if len(x) < WIN * 3:
        return None
    starts = np.arange(0, len(x) - WIN, HOP)
    rms = np.array([np.sqrt(np.mean(x[s:s + WIN] ** 2)) for s in starts])
    if rms.max() < 1e-3:
        return None
    speech = np.where(rms > 0.12 * rms.max())[0]
    if len(speech) < 8:
        return None
    last = speech[-1]
    tail = np.arange(max(speech[0], last - 31), last + 1)          # ~0.5 s of speech ending at the last speech hop
    f0 = np.array([_f0(x[starts[i]:starts[i] + WIN]) if rms[i] > 0.12 * rms.max() else 0.0 for i in tail])
    voiced = f0 > 0
    semis = 12 * np.log2(np.maximum(f0, 1e-3) / 100.0)
    slope = 0.0
    if voiced.sum() >= 5:
        t = tail[voiced] * HOP / SR
        slope = float(np.polyfit(t, semis[voiced], 1)[0])           # semitones per second
    body = rms[speech[-min(len(speech), 64):]]
    decay = float(np.mean(rms[max(0, last - 5):last + 1]) / (np.median(body) + 1e-9))
    steady = bool(voiced.sum() >= 18 and np.std(semis[voiced]) < 0.8 and np.std(rms[tail]) / (np.mean(rms[tail]) + 1e-9) < 0.25)
    return {"slope": slope, "decay": decay, "filler": steady, "voiced": int(voiced.sum())}


def eot_prosody(audio: np.ndarray, history=None) -> float:
    """P(caller is done) from acoustic cues alone, 0..1. 0.5 = no evidence (too little speech)."""
    f = features(audio)
    if f is None:
        return 0.5
    fall = float(np.clip(-f["slope"] / 6.0, -1.0, 1.0))
    soft = float(np.clip((0.75 - f["decay"]) / 0.5, -1.0, 1.0))
    z = 0.3 + 1.5 * fall + 0.9 * soft - 1.8 * float(f["filler"])
    return float(1.0 / (1.0 + np.exp(-z)))
