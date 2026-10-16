"""Silero VAD adapter (one instance per call: the model is stateful across frames).

Frame contract: 512 int16 samples at 16 kHz -> speech probability in [0, 1].
Imports are lazy so the rest of the package (and its tests) run without torch.
"""
from __future__ import annotations

import numpy as np


class SileroVAD:
    def __init__(self) -> None:
        import torch

        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        except ImportError:  # fall back to the torch.hub entry point
            self._model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._torch = torch
        torch.set_num_threads(1)

    def prob(self, frame: np.ndarray) -> float:
        x = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        with self._torch.no_grad():
            return float(self._model(x, 16000).item())

    def reset(self) -> None:
        self._model.reset_states()
