"""HttpTTS decodes one WAV (or streams client chunks). LimitedTTS yields without buffering the sentence."""
import asyncio
import io
import wave

import numpy as np
import pytest

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

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
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
    # The shared client stays at 5s for tools; a full utterance needs a longer read.
    timeout = client.calls[0]["timeout"]
    assert timeout.connect == 5.0 and timeout.read == 60.0


class _StreamClient:
    def post(self, url, json=None, headers=None, timeout=None):
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


async def test_limited_tts_follows_a_wav_sample_rate_change():
    """A rate copied once at wrap time resamples a 16 kHz WAV as 24 kHz for the rest of the call."""
    old = LimitedTTS._sem
    LimitedTTS._sem = None
    try:
        inner = HttpTTS("http://tts.example/v1", _Client(_wav([1, 2, 3], 16000)))
        lim = LimitedTTS(inner, 1)
        assert lim.sample_rate == 24000
        chunks = [c async for c in lim.synthesize("hi")]
        assert chunks[0].tolist() == [1, 2, 3]
        assert lim.sample_rate == 16000
    finally:
        LimitedTTS._sem = old


async def test_http_tts_does_not_play_an_error_document_as_pcm():
    class _Err:
        def __init__(self, status, content, content_type=None):
            self.status_code = status
            self.content = content
            self.headers = {"content-type": content_type} if content_type else {}

    class _ClientErr:
        def __init__(self, status, content, content_type=None):
            self.status, self.content, self.content_type = status, content, content_type

        async def post(self, url, json=None, headers=None, timeout=None):
            return _Err(self.status, self.content, self.content_type)

    with pytest.raises(RuntimeError):
        _ = [c async for c in HttpTTS("http://tts.example/v1", _ClientErr(500, b'{"error":"no"}')).synthesize("hi")]
    with pytest.raises(ValueError):
        _ = [c async for c in HttpTTS("http://tts.example/v1", _ClientErr(200, b'{"error":"no"}')).synthesize("hi")]
    with pytest.raises(ValueError):
        _ = [c async for c in HttpTTS("http://tts.example/v1", _ClientErr(200, b'\n{"error":"no"}\n')).synthesize("hi")]
    played: list = []
    with pytest.raises(ValueError):
        async for c in HttpTTS("http://tts.example/v1", _byte_stream(b'\n{"error":"no"}', 3)).synthesize("hi"):
            played.append(c)
    assert played == []
    with pytest.raises(ValueError):
        _ = [c async for c in HttpTTS(
            "http://tts.example/v1", _ClientErr(200, b"Internal Server Error", "text/plain")).synthesize("hi")]
    # A 200 plain-text body is not audio, with or without a PCM/WAV content type.
    for ctype in (None, "application/octet-stream", "audio/wav"):
        played = []
        with pytest.raises(ValueError):
            async for c in HttpTTS(
                    "http://tts.example/v1",
                    _ClientErr(200, b"Internal Server Error", ctype)).synthesize("hi"):
                played.append(c)
        assert played == []
    for body in (b"null", b"true", b'"error"'):
        with pytest.raises(ValueError):
            _ = [c async for c in HttpTTS(
                "http://tts.example/v1", _ClientErr(200, body)).synthesize("hi")]
    played = []
    with pytest.raises(ValueError):
        async for c in HttpTTS(
                "http://tts.example/v1", _byte_stream(b"Internal Server Error", 4)).synthesize("hi"):
            played.append(c)
    assert played == []
    # A finished body of only spaces or newlines is not s16le (0x20 would play as a sample).
    for body in (b"    ", b"\n", b"\r\n", b"\n\n", b" \t\r\n"):
        played = []
        with pytest.raises(ValueError):
            async for c in HttpTTS("http://tts.example/v1", _ClientErr(200, body)).synthesize("hi"):
                played.append(c)
        assert played == []
        played = []
        with pytest.raises(ValueError):
            async for c in HttpTTS("http://tts.example/v1", _byte_stream(body, 1)).synthesize("hi"):
                played.append(c)
        assert played == []
    # An empty body is silence, not an error document.
    assert [c async for c in HttpTTS("http://tts.example/v1", _ClientErr(200, b"")).synthesize("hi")] == []
    assert [c async for c in HttpTTS("http://tts.example/v1", _byte_stream(b"", 1)).synthesize("hi")] == []
    # audio/wav that is not a WAV must not be decoded as s16le.
    pcm_body = np.asarray([1, 2, 3, 4], dtype="<i2").tobytes()
    with pytest.raises(ValueError):
        _ = [c async for c in HttpTTS(
            "http://tts.example/v1", _ClientErr(200, pcm_body, "audio/wav")).synthesize("hi")]
    # A raw sample whose low byte is '<' / '[' / '{' is audio, not an error document.
    for lead, sample in ((b"<", 60), (b"[", 91), (b"{", 123)):
        raw = lead + b"\x00" + np.asarray([1000, -1000], dtype="<i2").tobytes()
        for ctype in (None, "audio/L16", "application/octet-stream"):
            got = [c async for c in HttpTTS(
                "http://tts.example/v1", _ClientErr(200, raw, ctype), sample_rate=24000).synthesize("hi")]
            assert np.concatenate(got).tolist() == [sample, 1000, -1000]
        streamed = [c async for c in HttpTTS(
            "http://tts.example/v1", _byte_stream(raw, 1), sample_rate=24000).synthesize("hi")]
        assert np.concatenate(streamed).tolist() == [sample, 1000, -1000]


def _byte_stream(body: bytes, step: int):
    class Client:
        def post(self, url, json=None, headers=None, timeout=None):
            async def chunks():
                for i in range(0, len(body), step):
                    yield body[i:i + step]

            return chunks()

    return Client()


async def test_split_wav_keeps_its_sample_rate_and_every_sample():
    """A WAV header cut across short reads is not raw 24 kHz, and a 1-byte read is not dropped."""
    samples = list(range(20))
    body = _wav(samples, 16000)
    for step in (1, 7, 13):
        tts = HttpTTS("http://tts.example/v1", _byte_stream(body, step), sample_rate=24000)
        got = [c async for c in tts.synthesize("hi")]
        pcm = np.concatenate(got) if got else np.zeros(0, np.int16)
        assert tts.sample_rate == 16000
        assert pcm.tolist() == samples


def _wav_with_trailer(samples, trailer: bytes) -> bytes:
    """A sized data chunk, then extra bytes. wave.open does not play the extra bytes."""
    pcm = np.asarray(samples, dtype="<i2").tobytes()
    fact = (len(samples)).to_bytes(4, "little")
    fmt = b"".join([
        (1).to_bytes(2, "little"),
        (1).to_bytes(2, "little"),
        (16000).to_bytes(4, "little"),
        (32000).to_bytes(4, "little"),
        (2).to_bytes(2, "little"),
        (16).to_bytes(2, "little"),
    ])

    def chunk(cid: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) & 1 else b""
        return cid + len(payload).to_bytes(4, "little") + payload + pad

    body = chunk(b"fact", fact) + chunk(b"fmt ", fmt) + chunk(b"data", pcm) + trailer
    return b"RIFF" + (4 + len(body)).to_bytes(4, "little") + b"WAVE" + body


async def test_wav_trailer_is_not_played_and_does_not_abort():
    """LIST or text after the data chunk is not more PCM, and it must not fail the utterance."""
    samples = [1000, -1000, 2000, -2000, 7]
    bodies = [
        _wav_with_trailer(samples, b"LIST" + (4).to_bytes(4, "little") + b"INFO"),
        _wav_with_trailer(samples, b"<INFO>tail</INFO>"),
    ]
    for body in bodies:
        buffered = HttpTTS("http://tts.example/v1", _Client(body), sample_rate=24000)
        whole = [c async for c in buffered.synthesize("hi")]
        assert buffered.sample_rate == 16000
        assert np.concatenate(whole).tolist() == samples
        for step in (1, 8, len(body)):
            tts = HttpTTS("http://tts.example/v1", _byte_stream(body, step), sample_rate=24000)
            got = [c async for c in tts.synthesize("hi")]
            pcm = np.concatenate(got) if got else np.zeros(0, np.int16)
            assert tts.sample_rate == 16000
            assert pcm.tolist() == samples


async def test_a_pcm_sample_split_across_chunks_is_not_dropped():
    raw = np.asarray([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
    tts = HttpTTS("http://tts.example/v1", _byte_stream(raw, 3), sample_rate=24000)
    got = [c async for c in tts.synthesize("hi")]
    pcm = np.concatenate(got)
    assert pcm.tolist() == [1000, -1000, 2000, -2000]
    assert tts.sample_rate == 24000


def test_claim_words_so_far_does_not_mark_the_rest_of_the_sentence_heard():
    """A barge-in during the first chunk must not be told the caller heard the whole sentence."""
    from core.session import _WordSlice, _claim_words_so_far, _fill_word_slices

    sentence = "one two three four five six"
    one = _WordSlice("SHOULD NOT STICK")
    _claim_words_so_far([one], [1600], sentence, 8000)  # 200 ms at 8 kHz -> one word
    assert one.text == "one"
    a, b = _WordSlice(""), _WordSlice("")
    _claim_words_so_far([a, b], [2880, 2880], sentence, 8000)  # 360 ms + 360 ms -> four words
    assert (a.text + " " + b.text).split() == ["one", "two", "three", "four"]
    assert sentence not in a.text and sentence not in b.text
    _fill_word_slices([a, b], [2880, 2880], sentence)
    assert (a.text + " " + b.text).split() == sentence.split()
