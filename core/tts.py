"""Kokoro-82M TTS adapter (Apache-2.0 weights; needs `pip install kokoro` + apt `espeak-ng`).

KPipeline is a blocking generator, so synthesis runs in a worker thread and PCM pieces are
handed back through an asyncio queue. Output: int16 mono at 24 kHz.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np


def make_pipeline(lang_code: str = "a"):
    """Loads Kokoro once. VOX_NO_ESPEAK=1 replaces the grapheme-to-phoneme stage with misaki's dictionary-only
    English G2P (no espeak-ng fallback), so no GPL-3.0 code is used at run time. Out-of-dictionary words
    (rare names, codes) are then pronounced less well. Written, not yet run on the GPU box: see docs/LICENSES.md."""
    import os
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code=lang_code)
    if os.environ.get("VOX_NO_ESPEAK") == "1" and lang_code in ("a", "b"):
        from misaki import en
        pipeline.g2p = en.G2P(trf=False, british=(lang_code == "b"), fallback=None)
    return pipeline


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, lang_code: str = "a",
                 pipeline=None):
        if pipeline is None:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code=lang_code)
        self.pipeline = pipeline
        self.voice = voice
        self.speed = speed

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        voice, speed = self.voice, self.speed

        def work() -> None:
            try:
                for _gs, _ps, audio in self.pipeline(text, voice=voice, speed=speed):
                    arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                    loop.call_soon_threadsafe(q.put_nowait, arr)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, work)
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield np.clip(np.rint(item * 32767.0), -32768, 32767).astype(np.int16)
        finally:
            await asyncio.shield(fut)
