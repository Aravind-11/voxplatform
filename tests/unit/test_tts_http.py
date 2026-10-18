"""HttpTTS speaks from a fake OpenAI-compatible endpoint. Kokoro is not loaded."""
import io
import wave

import numpy as np
import pytest

from api.runtime import LimitedTTS
from core.audio import resample
from core.session import Session, SessionConfig
from core.tts import HttpTTS, KokoroTTS, make_tts_factory


def _wav(pcm: np.ndarray, rate: int) -> bytes:
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


class _Client:
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
    client = _Client(_wav(pcm, 22050))
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
    client = _Client(pcm.tobytes())
    tts = HttpTTS("af_bella", "http://tts.local/v1", client, rate=8000)
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].dtype == np.int16 and out[0].tolist() == [1, -1, 2]
    assert tts.sample_rate == 8000
    assert client.posts[0]["headers"] is None
    assert client.posts[0]["json"]["voice"] == "af_bella"


async def test_http_tts_does_not_play_a_json_error_body():
    client = _Client(b'{"error":"nope"}')
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_does_not_play_a_bom_prefixed_json_error():
    client = _Client(b'\xef\xbb\xbf{"error":"nope"}')
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
    client = _Client(b"\xff\xfb\x00\x01" * 8, headers={"Content-Type": media})
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_does_not_play_an_id3_body_without_a_media_type():
    client = _Client(b"ID3" + b"\x00" * 40)
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_keeps_pcm_that_starts_like_an_mp3_frame():
    pcm = np.array([-1025, 256], np.int16)  # ff fb 00 01, a common MP3 frame sync in s16le
    assert pcm.tobytes()[:2] == b"\xff\xfb"
    client = _Client(pcm.tobytes())
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].dtype == np.int16 and out[0].tolist() == [-1025, 256]
    assert tts.sample_rate == 24000


async def test_wrapped_http_tts_is_resampled_at_the_wav_rate():
    """VOX_TTS_CONCURRENCY wraps TTS, and that wrapper used to snapshot sample_rate at init.
    A 16 kHz WAV was then resampled as 24 kHz."""
    pcm = np.arange(160, dtype=np.int16)
    client = _Client(_wav(pcm, 16000))
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
    sess.say("Hello.")
    gen = sess._gen
    assert gen is not None
    await gen.tasks[-1]
    assert wrapped.sample_rate == 16000
    assert len(sent) == 1
    assert sent[0][1] == 8000
    assert sent[0][0].tolist() == resample(pcm, 16000, 8000).tolist()
    assert len(sent[0][0]) != len(resample(pcm, 24000, 8000))


def test_tts_url_selects_http_tts_for_every_voice():
    client = _Client(b"")
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
    client = _Client(pcm.tobytes())
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
    sess.say("Hello there friend. Second sentence is fine.")
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
    client = _Client(np.array([7, -7], np.int16).tobytes())
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
    client = _Client(np.array([1, -1], np.int16).tobytes())
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, model="")
    assert tts.model == "tts-1"
    out = [x async for x in tts.synthesize("hi")]
    assert out[0].tolist() == [1, -1]
    assert client.posts[0]["json"]["model"] == "tts-1"


async def test_factory_posts_the_configured_model():
    client = _Client(np.array([3], np.int16).tobytes())
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
    tts = make_tts_factory(env, _Client(b""))("af_heart")
    assert isinstance(tts, HttpTTS) and tts.sample_rate == 24000


@pytest.mark.parametrize("media,body", [
    ("text/plain", b"this is not audio!!"),
    ("text/html", b"<html>nope</html>!!"),
    ("application/json", b"not-json"),
])
async def test_http_tts_does_not_play_text_or_a_non_document_json_body(media, body):
    client = _Client(body, headers={"Content-Type": media})
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(ValueError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


@pytest.mark.parametrize("status", [302, 500])
async def test_http_tts_does_not_play_a_non_success_status(status):
    pcm = np.array([9, -9, 8, -8], np.int16)
    client = _Client(pcm.tobytes(), status=status)
    tts = HttpTTS("af_heart", "http://tts.local/v1", client)
    got = []
    with pytest.raises(RuntimeError):
        async for chunk in tts.synthesize("hello"):
            got.append(chunk)
    assert got == []


async def test_http_tts_refuses_an_8bit_wav():
    client = _Client(_wave_bytes(bytes([0, 255, 128, 64]), 22050, channels=1, width=1))
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
    client = _Client(_wave_bytes(stereo.tobytes(), 16000, channels=2, width=2))
    tts = HttpTTS("af_heart", "http://tts.local/v1", client, rate=24000)
    out = [x async for x in tts.synthesize("hello")]
    assert len(out) == 1 and out[0].dtype == np.int16 and out[0].tolist() == left.tolist()
    assert tts.sample_rate == 16000
