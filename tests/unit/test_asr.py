"""HttpASR posts OpenAI multipart audio; Runtime uses it only when VOX_ASR_URL is set."""
import numpy as np

from api.runtime import Runtime, _LlmWithASR
from api.store import Store
from core.asr import HttpASR


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class _Client:
    def __init__(self, body):
        self.body = body
        self.posts = []

    async def post(self, url, files=None, data=None):
        self.posts.append({"url": url, "files": files, "data": data})
        return _Resp(self.body)


async def test_http_asr_posts_multipart_and_returns_text():
    client = _Client({"text": "  hello there  "})
    asr = HttpASR("http://asr.example/v1/", client)
    text = await asr.transcribe(np.array([0, 100, -100], dtype=np.int16))
    assert text == "hello there"
    assert len(client.posts) == 1
    post = client.posts[0]
    assert post["url"] == "http://asr.example/v1/audio/transcriptions"
    assert post["data"] == {"model": "whisper-1"}
    name, wav, mime = post["files"]["file"]
    assert (name, mime) == ("audio.wav", "audio/wav")
    assert wav[:4] == b"RIFF" and b"WAVE" in wav[:16]


class _LLM:
    def __init__(self):
        self.transcribed = 0
        self.streamed = 0

    def stream(self, history, audio, followups, tools):
        self.streamed += 1

        async def gen():
            yield "ok"

        return gen()

    def eot_score(self, history, audio):
        return 0.25

    async def transcribe(self, audio):
        self.transcribed += 1
        return "from-llm"


class _TTS:
    sample_rate = 8000


class _Transport:
    out_rate = 8000

    async def send_audio(self, pcm, rate): ...

    async def clear(self): ...

    async def end(self): ...


def _session(rt):
    store = rt.store
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "webSocket", {"systemPrompt": "hi"})
    return rt.build_session(call, _Transport())


async def test_runtime_uses_asr_only_when_configured(monkeypatch):
    monkeypatch.delenv("VOX_ASR_URL", raising=False)
    llm = _LLM()
    client = _Client({"text": "from-asr"})
    rt = Runtime(llm, lambda v: _TTS(), lambda: None, Store(), http=client)
    sess = _session(rt)
    assert sess.llm is llm
    assert await sess.llm.transcribe(np.zeros(160, np.int16)) == "from-llm"
    assert llm.transcribed == 1

    monkeypatch.setenv("VOX_ASR_URL", "http://asr.example/v1")
    sess2 = _session(rt)
    assert isinstance(sess2.llm, _LlmWithASR)
    assert sess2.llm._inner is llm
    assert await sess2.llm.transcribe(np.zeros(160, np.int16)) == "from-asr"
    assert llm.transcribed == 1
    assert client.posts[0]["url"] == "http://asr.example/v1/audio/transcriptions"
    assert sess2.llm.eot_score([], np.zeros(0, np.int16)) == 0.25
    chunks = [c async for c in sess2.llm.stream([], np.zeros(0, np.int16), [], [])]
    assert chunks == ["ok"] and llm.streamed == 1
