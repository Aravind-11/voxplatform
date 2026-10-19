"""Logistic regression on the acoustic features from core.prosody.

Default coefficients are the hand priors in eot_prosody (bias 0.3, slope 1.5, decay 0.9,
filler -1.8, voiced 0). A missing feature vector returns 0.5, the same "no evidence" answer.
This does not solve end-of-turn detection: the priors still cannot tell a finished question
from a trailing "um", and a fitted VOX_EOT_MODEL only retunes the same four cues.
"""
from __future__ import annotations

import json
import math

import numpy as np

from core.prosody import features

DEFAULT_WEIGHTS = {"bias": 0.3, "slope": 1.5, "decay": 0.9, "filler": -1.8, "voiced": 0.0}
_KEYS = tuple(DEFAULT_WEIGHTS)


class EotClassifier:
    def __init__(self, weights: dict | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        for k, v in (weights or {}).items():
            if k in self.weights and v is not None:
                self.weights[k] = float(v)

    @classmethod
    def load(cls, path: str) -> "EotClassifier":
        """JSON {"bias", "slope", "decay", "filler", "voiced"}. A missing or unreadable file keeps the priors."""
        if not path:
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        picked = {}
        for k in _KEYS:
            if k in data and data[k] is not None:
                try:
                    value = float(data[k])
                except (TypeError, ValueError, OverflowError):
                    return cls()
                # 1e309 and JSON Infinity/NaN become inf/nan without raising.
                if not math.isfinite(value):
                    return cls()
                picked[k] = value
        return cls(picked)

    def score_features(self, feats: dict) -> float:
        """Sigmoid of the linear score. slope and decay use the eot_prosody clips; filler is 0/1."""
        w = self.weights
        filler = 1.0 if feats.get("filler") else 0.0
        slope = float(feats.get("slope") or 0.0)
        decay = float(feats.get("decay") or 0.0)
        fall = float(np.clip(-slope / 6.0, -1.0, 1.0))
        soft = float(np.clip((0.75 - decay) / 0.5, -1.0, 1.0))
        z = (w["bias"]
             + w["slope"] * fall
             + w["decay"] * soft
             + w["filler"] * filler
             + w["voiced"] * float(feats.get("voiced") or 0.0))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))

    def score(self, audio) -> float:
        feats = features(audio)
        if feats is None:
            return 0.5
        return self.score_features(feats)
