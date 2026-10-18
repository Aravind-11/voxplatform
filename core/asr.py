"""Optional HTTP speech-to-text so a tool round does not spend another Ultravox pass on audio.

The vLLM server accepts one audio clip per request (`--limit-mm-per-prompt`). When `VOX_ASR_URL`
is set, caller transcripts come from this client instead.
"""
from __future__ import annotations

import asyncio
import io
import wave

import numpy as np


def _wav_bytes(audio: np.ndarray, rate: int = 16000) -> bytes:
    pcm = np.asarray(audio, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class HttpASR:
    """OpenAI-compatible `POST {base_url}/audio/transcriptions` (multipart file + model)."""

    def __init__(self, base_url: str, client, model: str = "whisper-1") -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.model = model

    async def transcribe(self, audio: np.ndarray) -> str:
        resp = await self.client.post(
            f"{self.base_url}/audio/transcriptions",
            files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")},
            data={"model": self.model},
        )
        body = resp.json()
        if asyncio.iscoroutine(body):
            body = await body
        if isinstance(body, dict):
            return str(body.get("text") or "").strip()
        return str(body or "").strip()
