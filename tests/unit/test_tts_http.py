"""HttpTTS decodes one WAV (or streams client chunks). LimitedTTS yields without buffering the sentence."""
import asyncio
import io
import wave

import numpy as np

from api.runtime import LimitedTTS
from core.tts import HttpTTS


def _wav(samples, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class _Body:
    def __init__(self, content: bytes):
        self.content = content


class _Client:
    def __init__(self, body):
        self.body = body
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _Body(self.body)


async def test_http_tts_decodes_wav_and_takes_its_sample_rate():
    client = _Client(_wav([1, 2, 3, 4], 16000))
    tts = HttpTTS("http://tts.example/v1/", client, model="tts-1", voice="af_heart", api_key="k")
    assert tts.sample_rate == 24000
    chunks = [c async for c in tts.synthesize("hi")]
    assert len(chunks) == 1 and chunks[0].tolist() == [1, 2, 3, 4]
    assert tts.sample_rate == 16000
    assert client.calls[0]["url"] == "http://tts.example/v1/audio/speech"
    assert client.calls[0]["json"] == {"model": "tts-1", "voice": "af_heart", "input": "hi"}
    assert client.calls[0]["headers"]["Authorization"] == "Bearer k"


class _StreamClient:
    def post(self, url, json=None, headers=None):
        async def chunks():
            yield np.array([5, 6], dtype=np.int16)
            yield np.array([7], dtype=np.int16)

        return chunks()


async def test_http_tts_yields_each_piece_when_the_client_streams():
    tts = HttpTTS("http://tts.example/v1", _StreamClient())
    chunks = [c async for c in tts.synthesize("a")]
    assert [c.tolist() for c in chunks] == [[5, 6], [7]]
    assert tts.sample_rate == 24000


async def test_limited_tts_yields_each_chunk_while_holding_the_semaphore():
    old = LimitedTTS._sem
    LimitedTTS._sem = None
    gate = asyncio.Event()

    class Inner:
        sample_rate = 8000

        async def synthesize(self, text):
            yield np.array([1], dtype=np.int16)
            await gate.wait()
            yield np.array([2], dtype=np.int16)

    try:
        lim = LimitedTTS(Inner(), 1)

        async def consume():
            agen = lim.synthesize("hi")
            first = await agen.__anext__()
            assert LimitedTTS._sem.locked()
            gate.set()
            second = await agen.__anext__()
            await agen.aclose()
            return first.tolist(), second.tolist()

        assert await asyncio.wait_for(consume(), 1) == ([1], [2])
    finally:
        gate.set()
        LimitedTTS._sem = old


async def test_limited_tts_does_not_buffer_before_the_first_yield():
    """Buffering every chunk before yielding deadlocks: the second chunk waits on the first send."""
    old = LimitedTTS._sem
    LimitedTTS._sem = None
    gate = asyncio.Event()

    class Inner:
        sample_rate = 8000

        async def synthesize(self, text):
            yield np.array([9], dtype=np.int16)
            await gate.wait()
            yield np.array([8], dtype=np.int16)

    try:
        lim = LimitedTTS(Inner(), 2)

        async def consume():
            out = []
            async for p in lim.synthesize("hi"):
                out.append(int(p[0]))
                gate.set()
            return out

        assert await asyncio.wait_for(consume(), 1) == [9, 8]
    finally:
        gate.set()
        LimitedTTS._sem = old
