"""HttpTTS speaks from a fake OpenAI-compatible endpoint. Kokoro is not loaded."""
import io
import wave

import numpy as np
import pytest

from api.runtime import LimitedTTS
from core.audio import resample
from core.session import Session, SessionConfig
from core.tts import HttpTTS, make_tts_factory


def _wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


class _Resp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"status {self.status}")


class _Client:
    def __init__(self, content: bytes):
        self.content, self.posts = content, []

    async def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _Resp(self.content)


async def test_http_tts_decodes_a_wav_to_int16():
    pcm = np.array([0, 100, -200, 32767], np.int16)
    client = _Client(_wav(pcm, 22050))
    tts = HttpTTS("af_heart", "http://tts.local/v1/", client, model="tts-1", api_key="sk-test")
    out = [x async for x in tts.synthesize("hello")]
    assert len(out) == 1 and out[0].dtype == np.int16 and out[0].tolist() == pcm.tolist()
    assert tts.sample_rate == 22050
    post = client.posts[0]
    assert post["url"] == "http://tts.local/v1/audio/speech"
    assert post["json"] == {"model": "tts-1", "voice": "af_heart", "input": "hello"}
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
