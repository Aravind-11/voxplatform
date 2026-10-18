"""OpenAI-compatible speech-to-text for caller transcripts.

Ultravox is served with one audio clip per prompt (`--limit-mm-per-prompt audio:1`),
so earlier turns cannot be sent back as audio. When `VOX_ASR_URL` is set, those
history transcripts come from this client instead of a second Ultravox pass.
The URL includes `/v1`, same shape as `VOX_VLLM_URL`.
"""
from __future__ import annotations

import io
import wave

import numpy as np


class HttpASR:
    def __init__(self, base_url: str, client, model: str = "whisper-1"):
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.model = model or "whisper-1"

    async def transcribe(self, audio) -> str:
        # The pooled client reads for 5s. A slower transcript must not become
        # [inaudible], and timeout=None would stall the next turn that awaits it.
        response = await self.client.post(
            f"{self.base_url}/audio/transcriptions",
            files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")},
            data={"model": self.model},
            timeout=60)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        body = response.json()
        text = body.get("text") if isinstance(body, dict) else None
        return text.strip() if isinstance(text, str) else ""


def _wav_bytes(audio) -> bytes:
    pcm = np.asarray(audio, dtype=np.int16).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class AsrLLM:
    """Session-facing LLM: `transcribe` uses HttpASR; `stream` and `eot_score` stay on the inner client.

    Do not install this on `Runtime.llm`. End-of-turn looks up `eot_score` on the unwrapped
    Ultravox client, and an unset `VOX_ASR_URL` must keep `VllmClient.transcribe`.
    """

    def __init__(self, inner, asr: HttpASR):
        self._inner = inner
        self._asr = asr

    async def transcribe(self, audio) -> str:
        return await self._asr.transcribe(audio)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
