"""Separate ASR for history transcripts, and the restateSlots prompt reminder."""
import io
import wave

import numpy as np

from api.runtime import RESTATE_SLOTS, Runtime
from api.store import Store
from core.asr import AsrLLM, HttpASR
from tests.unit.test_api import FakeTTS, FakeVAD, make, tenant


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"status {self.status}")


class _Client:
    def __init__(self, body, status=200):
        self.body, self.status, self.posts = body, status, []

    async def post(self, url, files=None, data=None, timeout=None):
        self.posts.append({"url": url, "files": files, "data": data, "timeout": timeout})
        # The process-wide client reads for 5s. Omitting timeout, or setting it under 60s,
        # aborts a slow transcript and the session stores [inaudible].
        if not isinstance(timeout, (int, float)) or float(timeout) < 60:
            raise RuntimeError(f"ASR timeout {timeout!r} is missing or below 60s")
        return _Resp(self.body, self.status)


class _Inner:
    def __init__(self):
        self.streams = 0
        self.transcripts = 0

    async def stream(self, history, audio, followups, tools):
        self.streams += 1
        yield "hello"

    async def transcribe(self, audio):
        self.transcripts += 1
        return "from-ultravox"

    async def eot_score(self, history, audio):
        return 0.4


class _T:
    out_rate = 24000

    async def send_audio(self, b, rate=24000): ...
    async def clear(self): ...
    async def end(self): ...


def _pcm(payload: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(payload), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


async def test_http_asr_posts_wav_and_returns_text():
    audio = np.array([0, 1000, -1000, 50], np.int16)
    client = _Client({"text": "  tuesday at eleven  "})
    asr = HttpASR("http://asr.local/v1/", client, model="whisper-test")
    assert await asr.transcribe(audio) == "tuesday at eleven"
    post = client.posts[0]
    assert post["url"] == "http://asr.local/v1/audio/transcriptions"
    assert post["data"] == {"model": "whisper-test"}
    name, payload, mime = post["files"]["file"]
    assert name == "audio.wav" and mime == "audio/wav"
    assert post["timeout"] == 60
    assert np.array_equal(_pcm(payload), audio)
    assert await HttpASR("http://asr.local/v1", _Client({"nope": 1})).transcribe(audio) == ""
    assert await HttpASR("http://asr.local/v1", _Client("nope")).transcribe(audio) == ""


async def test_wrapper_transcribes_with_asr_and_streams_on_the_inner_client():
    audio = np.zeros(4, np.int16)
    inner = _Inner()
    client = _Client({"text": "from asr"})
    wrapped = AsrLLM(inner, HttpASR("http://asr.local/v1", client))
    assert await wrapped.transcribe(audio) == "from asr"
    assert inner.transcripts == 0
    assert [x async for x in wrapped.stream([], audio, [], [])] == ["hello"]
    assert inner.streams == 1
    assert await wrapped.eot_score([], audio) == 0.4


def _call(store, cfg):
    tid = store.create_tenant("t")
    return store.create_call(tid, None, "webSocket", cfg, {})


async def test_runtime_prefers_asr_only_when_url_and_client_are_set(monkeypatch):
    monkeypatch.setenv("VOX_ASR_URL", "http://asr.local/v1/")
    monkeypatch.setenv("VOX_ASR_MODEL", "whisper-test")
    monkeypatch.setenv("VOX_EOT", "model")
    inner, client = _Inner(), _Client({"text": "from asr"})
    store = Store()
    rt = Runtime(inner, lambda v: FakeTTS(), FakeVAD, store, http=client)
    sess = rt.build_session(_call(store, {"systemPrompt": "Be brief."}), _T())
    audio = np.array([1, -2, 3], np.int16)
    assert sess.llm is not inner
    assert await sess.llm.transcribe(audio) == "from asr"
    assert inner.transcripts == 0
    assert client.posts[0]["url"] == "http://asr.local/v1/audio/transcriptions"
    assert client.posts[0]["data"] == {"model": "whisper-test"}
    assert client.posts[0]["timeout"] == 60
    assert [x async for x in sess.llm.stream([], audio, [], [])] == ["hello"] and inner.streams == 1
    assert rt.llm is inner and await rt._eot_async()(audio, []) == 0.4

    monkeypatch.delenv("VOX_ASR_URL")
    rt_off = Runtime(inner, lambda v: FakeTTS(), FakeVAD, store, http=client)
    sess_off = rt_off.build_session(_call(store, {"systemPrompt": "Be brief."}), _T())
    assert sess_off.llm is inner
    assert await sess_off.llm.transcribe(audio) == "from-ultravox" and inner.transcripts == 1

    monkeypatch.setenv("VOX_ASR_URL", "http://asr.local/v1")
    rt_no_http = Runtime(inner, lambda v: FakeTTS(), FakeVAD, store)
    assert rt_no_http.build_session(_call(store, {"systemPrompt": "Be brief."}), _T()).llm is inner


def test_restate_sentence_is_in_the_prompt_only_when_the_flag_is_on(monkeypatch):
    monkeypatch.delenv("VOX_ASR_URL", raising=False)
    store = Store()
    rt = Runtime(_Inner(), lambda v: FakeTTS(), FakeVAD, store)
    on = rt.build_session(_call(store, {"systemPrompt": "Be brief.", "restateSlots": True, "language": "es"}), _T())
    text = on.history[0]["content"]
    assert text.startswith("Be brief.") and "Spanish" in text and RESTATE_SLOTS in text
    off = rt.build_session(_call(store, {"systemPrompt": "Be brief.", "language": "es"}), _T())
    assert RESTATE_SLOTS not in off.history[0]["content"] and "Spanish" in off.history[0]["content"]
    only = rt.build_session(_call(store, {"systemPrompt": "Be brief.", "restateSlots": True}), _T())
    assert only.history[0]["content"] == "Be brief.\n\n" + RESTATE_SLOTS


def test_agent_stores_restate_slots_only_when_enabled():
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    plain = c.post("/v1/agents", json={"name": "p", "systemPrompt": "hi"}, headers=h).json()
    assert "restateSlots" not in store.get_agent_any(plain["id"])["config"]
    flagged = c.post("/v1/agents", json={"name": "f", "systemPrompt": "hi", "restateSlots": True}, headers=h).json()
    assert store.get_agent_any(flagged["id"])["config"]["restateSlots"] is True
    assert c.patch(f"/v1/agents/{flagged['id']}", json={"restateSlots": False}, headers=h).status_code == 200
    assert "restateSlots" not in store.get_agent_any(flagged["id"])["config"]
    assert c.patch(f"/v1/agents/{plain['id']}", json={"systemPrompt": "still"}, headers=h).status_code == 200
    assert "restateSlots" not in store.get_agent_any(plain["id"])["config"]
    assert c.patch(f"/v1/agents/{plain['id']}", json={"restateSlots": True}, headers=h).status_code == 200
    assert store.get_agent_any(plain["id"])["config"]["restateSlots"] is True
