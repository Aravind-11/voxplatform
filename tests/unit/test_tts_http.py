"""HttpTTS decodes one WAV (or streams client chunks) and can replace Kokoro.

LimitedTTS yields each chunk without buffering the sentence. When VOX_TTS_URL is set, Kokoro
is not loaded. Streaming reads and a single PCM/WAV body are both one logical response.
"""
import asyncio
import io
import wave

import numpy as np
import pytest

from api.runtime import LimitedTTS
from core.audio import resample
from core.session import Session, SessionConfig
from core.tts import HttpTTS, KokoroTTS, make_tts_factory

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
    # response_format is pcm so an OpenAI-compatible server does not default to mp3.
    assert client.calls[0]["json"] == {
        "model": "tts-1", "voice": "af_heart", "input": "hi", "response_format": "pcm"}
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


async def test_limited_tts_releases_the_slot_after_the_utterance():
    """A finished utterance with no detached worker must return the process-wide permit.

    The next call shares that semaphore. Dropping the empty-pending release leaves it at zero
    and this wait times out.
    """
    old = LimitedTTS._sem
    LimitedTTS._sem = None

    class Inner:
        sample_rate = 8000

        async def synthesize(self, text):
            yield np.array([7], dtype=np.int16)

    try:
        lim = LimitedTTS(Inner(), 1)
        first = [c async for c in lim.synthesize("one")]
        assert first[0].tolist() == [7]
        sem = LimitedTTS._sem

        async def second():
            got = [c async for c in lim.synthesize("two")]
            return got[0].tolist()

        assert await asyncio.wait_for(second(), 1) == [7]
        assert LimitedTTS._sem is sem
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


def test_build_selects_http_tts_from_the_environment(monkeypatch, tmp_path):
    """Deleting the VOX_TTS_URL branch, or a model/key/rate default, must fail. Import runs build()."""
    import sys

    from core.metrics import METRICS

    monkeypatch.setenv("VOX_TTS_URL", "http://tts.example/v1")
    monkeypatch.setenv("VOX_TTS_MODEL", "tts-1-hd")
    monkeypatch.setenv("VOX_TTS_API_KEY", "secret")
    monkeypatch.setenv("VOX_TTS_RATE", "16000")
    monkeypatch.setenv("VOX_DB", str(tmp_path / "vox.db"))
    monkeypatch.delenv("VOX_REDIS_URL", raising=False)
    monkeypatch.delenv("VOX_EMBED_URL", raising=False)
    assert "api.main" not in sys.modules
    gauges = dict(METRICS.gauges)
    import api.main
    rt = api.main.app.state.runtime
    try:
        tts = rt.make_tts("af_heart")
        assert isinstance(tts, HttpTTS)
        assert tts.base_url == "http://tts.example/v1"
        assert tts.model == "tts-1-hd"
        assert tts.voice == "af_heart"
        assert tts.api_key == "secret"
        assert tts.sample_rate == 16000
        assert tts.client is rt.http
    finally:
        METRICS.gauges.clear()
        METRICS.gauges.update(gauges)
        rt.store.db.close()


def _speech_wav(pcm: np.ndarray, rate: int) -> bytes:
    return _wave_bytes(np.asarray(pcm, dtype=np.int16).tobytes(), rate, channels=1, width=2)


def _wave_bytes(frames: bytes, rate: int, channels: int, width: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


class _Resp:
    def __init__(self, content: bytes, status: int = 200, headers=None):
        self.content, self.status, self.headers = content, status, headers

    def raise_for_status(self):
        # httpx raises for every status outside 2xx, including redirects.
        if self.status < 200 or self.status >= 300:
            raise RuntimeError(f"status {self.status}")


class _SpeechClient:
    def __init__(self, content: bytes, headers=None, status: int = 200):
        self.content, self.headers, self.status, self.posts = content, headers, status, []

    async def post(self, url, json=None, headers=None, **kwargs):
        self.posts.append({"url": url, "json": json, "headers": headers, **kwargs})
        return _Resp(self.content, status=self.status, headers=self.headers)


def _read_timeout_s(timeout) -> float:
    """Read budget on the speech POST. A bare number applies to every phase, including read."""
    if isinstance(timeout, (int, float)):
        return float(timeout)
    read = getattr(timeout, "read", None)
    if read is None:
        raise AssertionError(f"speech POST read timeout is not finite: {timeout!r}")
    return float(read)


class _Scripted:
    """One body per POST, in order. Records keyword arguments the same way `_Client` does."""

    def __init__(self, bodies: list[bytes]):
        self.bodies, self.posts = list(bodies), []

    async def post(self, url, json=None, headers=None, **kwargs):
        self.posts.append({"url": url, "json": json, "headers": headers, **kwargs})
        return _Resp(self.bodies.pop(0))


async def test_http_tts_decodes_a_wav_to_int16():
    pcm = np.array([0, 100, -200, 32767], np.int16)
    client = _SpeechClient(_speech_wav(pcm, 22050))
    tts = HttpTTS("af_heart", "http://tts.local/v1/", client, model="tts-1", api_key="sk-test")
    out = [x async for x in tts.synthesize("hello")]
    assert len(out) == 1 and out[0].dtype == np.int16 and out[0].tolist() == pcm.tolist()
    assert tts.sample_rate == 22050
    post = client.posts[0]
    assert post["url"] == "http://tts.local/v1/audio/speech"
    assert post["json"] == {
        "model": "tts-1", "voice": "af_heart", "input": "hello", "response_format": "pcm"}
    assert post["headers"]["Authorization"] == "Bearer sk-test"


async def test_http_tts_treats_a_headerless_body_as_raw_s16le():
    pcm = np.array([1, -1, 2], np.int16)
    client = _SpeechClient(pcm.tobytes())
    tts = HttpTTS("af_bella", "http://tts.local/v1", client, rate=8000)
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].dtype == np.int16 and out[0].tolist() == [1, -1, 2]
    assert tts.sample_rate == 8000
    assert client.posts[0]["headers"] is None
    assert client.posts[0]["json"]["voice"] == "af_bella"


async def test_http_tts_does_not_play_a_json_error_body():
    client = _SpeechClient(b'{"error":"nope"}')
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_does_not_play_a_bom_prefixed_json_error():
    client = _SpeechClient(b'\xef\xbb\xbf{"error":"nope"}')
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


@pytest.mark.parametrize("media", [
    "audio/mpeg", "audio/mp3", "audio/opus", "audio/ogg", "audio/flac", "audio/aac", "audio/mp4",
    "audio/mpeg; charset=binary",
])
async def test_http_tts_does_not_play_compressed_audio(media):
    # No ID3 tag: the media type alone must refuse the body. A bare frame sync is valid PCM.
    client = _SpeechClient(b"\xff\xfb\x00\x01" * 8, headers={"Content-Type": media})
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_does_not_play_an_id3_body_without_a_media_type():
    client = _SpeechClient(b"ID3" + b"\x00" * 40)
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_keeps_pcm_that_starts_like_an_mp3_frame():
    pcm = np.array([-1025, 256], np.int16)  # ff fb 00 01, a common MP3 frame sync in s16le
    assert pcm.tobytes()[:2] == b"\xff\xfb"
    client = _SpeechClient(pcm.tobytes())
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].dtype == np.int16 and out[0].tolist() == [-1025, 256]
    assert tts.sample_rate == 24000


async def test_wrapped_http_tts_is_resampled_at_the_wav_rate():
    """VOX_TTS_CONCURRENCY wraps TTS, and that wrapper used to snapshot sample_rate at init.
    A 16 kHz WAV was then resampled as 24 kHz."""
    old_sem = LimitedTTS._sem
    LimitedTTS._sem = None
    try:
        pcm = np.arange(160, dtype=np.int16)
        client = _SpeechClient(_speech_wav(pcm, 16000))
        inner = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
        wrapped = LimitedTTS(inner, 4)
        sent = []

        async def send(audio, rate):
            sent.append((np.array(audio, copy=True), rate))

        async def clear():
            return None

        class VAD:
            def prob(self, frame):
                return 0.0

        class LLM:
            async def stream(self, history, audio, followups, tools):
                if False:
                    yield ""

        sess = Session(llm=LLM(), tts=wrapped, vad=VAD(), send_audio=send, clear_audio=clear,
                       clock=lambda: 0.0, cfg=SessionConfig(out_rate=8000, tts_poll_s=0))
        await sess.say("Hello.")
        gen = sess._gen
        assert gen is not None
        await gen.tasks[-1]
        assert wrapped.sample_rate == 16000
        assert len(sent) == 1
        assert sent[0][1] == 8000
        assert sent[0][0].tolist() == resample(pcm, 16000, 8000).tolist()
        assert len(sent[0][0]) != len(resample(pcm, 24000, 8000))
    finally:
        LimitedTTS._sem = old_sem


def test_tts_url_selects_http_tts_for_every_voice():
    client = _SpeechClient(b"")
    env = {"VOX_TTS_URL": "http://tts.local/v1/", "VOX_TTS_MODEL": "custom-voice", "VOX_TTS_API_KEY": "k",
           "VOX_TTS_RATE": "16000"}
    factory = make_tts_factory(env, client)
    a, b = factory("af_heart"), factory("af_bella")
    assert isinstance(a, HttpTTS) and isinstance(b, HttpTTS)
    assert a.url == "http://tts.local/v1" and a.model == "custom-voice" and a.api_key == "k"
    assert a.sample_rate == 16000 and b.voice == "af_bella" and a.client is client


async def test_http_tts_speech_post_waits_longer_than_the_shared_client_default():
    """api.main builds one httpx.AsyncClient() with a 5s read timeout. A sentence whose first
    byte takes longer than that must not be a ReadTimeout, and a dead peer must still unblock."""
    pcm = np.array([1, -1], np.int16)
    client = _SpeechClient(pcm.tobytes())
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    out = [x async for x in tts.synthesize("hello")]
    assert out[0].tolist() == [1, -1]
    assert _read_timeout_s(client.posts[0]["timeout"]) >= 30


async def test_empty_speech_body_is_not_stored_as_spoken_and_the_next_sentence_plays():
    """A 2xx body with no samples is not audio the caller heard. Dropping it must not cancel
    the rest of the reply the way a synthesize exception does."""
    pcm = np.array([100, -100, 50], np.int16)
    client = _Scripted([b"", pcm.tobytes()])
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    events, sent = [], []

    async def send(audio, rate):
        sent.append((np.array(audio, copy=True), rate))

    async def clear():
        return None

    class VAD:
        def prob(self, frame):
            return 0.0

    class LLM:
        async def stream(self, history, audio, followups, tools):
            if False:
                yield ""

    sess = Session(llm=LLM(), tts=tts, vad=VAD(), send_audio=send, clear_audio=clear,
                   clock=lambda: 0.0, cfg=SessionConfig(out_rate=8000, tts_poll_s=0),
                   on_event=events.append)
    await sess.say("Hello there friend. Second sentence is fine.")
    gen = sess._gen
    assert gen is not None
    await gen.tasks[-1]
    assert [p["json"]["input"] for p in client.posts] == [
        "Hello there friend.", "Second sentence is fine."]
    assert len(sent) == 1 and sent[0][1] == 8000
    assert sent[0][0].tolist() == resample(pcm, 24000, 8000).tolist()
    assert [e["text"] for e in events if e["type"] == "agent_speech"] == ["Second sentence is fine."]
    assert gen.spoken == ["Second sentence is fine."]
    assert sess.history[-1] == {"role": "assistant", "content": "Second sentence is fine."}
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and errors[0]["where"] == "tts"


class _PipelineSet:
    """Stands in for Kokoro's PipelineSet. A real one imports kokoro, which is not installed."""

    instances: list["_PipelineSet"] = []

    def __init__(self, factory=None, preload: str = "a"):
        self.factory, self.preload, self.sentinels = factory, preload, {}
        _PipelineSet.instances.append(self)

    def for_voice(self, voice: str):
        self.sentinels.setdefault(voice, object())
        return self.sentinels[voice]


@pytest.mark.parametrize("env", [
    {},
    {"VOX_TTS_URL": ""},
    {"VOX_TTS_URL": " \t "},
], ids=["missing", "empty", "whitespace"])
def test_blank_tts_url_keeps_one_shared_kokoro_pipeline(env, monkeypatch):
    """An unset, blank, or whitespace URL stays on Kokoro. Every voice shares that one set."""
    monkeypatch.setattr("core.tts.PipelineSet", _PipelineSet)
    _PipelineSet.instances.clear()
    factory = make_tts_factory(dict(env), object())
    heart, bella = factory("af_heart"), factory("af_bella")
    assert isinstance(heart, KokoroTTS) and isinstance(bella, KokoroTTS)
    assert len(_PipelineSet.instances) == 1
    pipelines = _PipelineSet.instances[0]
    assert pipelines.preload == "a"
    assert heart.pipeline is pipelines.sentinels["af_heart"]
    assert bella.pipeline is pipelines.sentinels["af_bella"]
    assert heart.pipeline is not bella.pipeline


@pytest.mark.parametrize("model", [None, "", " \t"])
async def test_factory_posts_tts_1_when_model_is_omitted_or_blank(model):
    client = _SpeechClient(np.array([7, -7], np.int16).tobytes())
    env = {"VOX_TTS_URL": "http://tts.local/v1"}
    if model is not None:
        env["VOX_TTS_MODEL"] = model
    tts = make_tts_factory(env, client)("af_heart")
    assert isinstance(tts, HttpTTS) and tts.model == "tts-1"
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].tolist() == [7, -7]
    assert client.posts[0]["json"]["model"] == "tts-1"


async def test_blank_model_argument_posts_tts_1():
    """HttpTTS(model='') is the same default the factory passes through after strip()."""
    client = _SpeechClient(np.array([1, -1], np.int16).tobytes())
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, model="")
    assert tts.model == "tts-1"
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].tolist() == [1, -1]
    assert client.posts[0]["json"]["model"] == "tts-1"


async def test_factory_posts_the_configured_model():
    client = _SpeechClient(np.array([3], np.int16).tobytes())
    tts = make_tts_factory(
        {"VOX_TTS_URL": "http://tts.local/v1/", "VOX_TTS_MODEL": "licensed-voice"}, client)("af_bella")
    assert isinstance(tts, HttpTTS) and tts.model == "licensed-voice"
    out = [x async for x in tts.synthesize("yo")]
    assert out[0].tolist() == [3]
    assert client.posts[0]["json"]["model"] == "licensed-voice"
    assert client.posts[0]["json"]["voice"] == "af_bella"
    assert client.posts[0]["json"]["input"] == "yo"


@pytest.mark.parametrize("rate", [None, ""])
def test_factory_default_sample_rate_is_24000(rate):
    env = {"VOX_TTS_URL": "http://tts.local/v1"}
    if rate is not None:
        env["VOX_TTS_RATE"] = rate
    tts = make_tts_factory(env, _SpeechClient(b""))("af_heart")
    assert isinstance(tts, HttpTTS) and tts.sample_rate == 24000


@pytest.mark.parametrize("media,body", [
    ("text/plain", b"this is not audio!!"),
    ("text/html", b"<html>nope</html>!!"),
    ("application/json", b"not-json"),
])
async def test_http_tts_does_not_play_text_or_a_non_document_json_body(media, body):
    client = _SpeechClient(body, headers={"Content-Type": media})
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


@pytest.mark.parametrize("status", [302, 500])
async def test_http_tts_does_not_play_a_non_success_status(status):
    pcm = np.array([9, -9, 8, -8], np.int16)
    client = _SpeechClient(pcm.tobytes(), status=status)
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(RuntimeError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_refuses_an_8bit_wav():
    client = _SpeechClient(_wave_bytes(bytes([0, 255, 128, 64]), 22050, channels=1, width=1))
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_keeps_channel_0_of_a_stereo_wav():
    left = np.array([1, 2, 3, 4], np.int16)
    right = np.array([-100, -200, -300, -400], np.int16)
    stereo = np.empty(left.size * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right
    client = _SpeechClient(_wave_bytes(stereo.tobytes(), 16000, channels=2, width=2))
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
    out = [x async for x in tts.synthesize("hello")]
    assert len(out) == 1 and out[0].dtype == np.int16 and out[0].tolist() == left.tolist()
    assert tts.sample_rate == 16000
