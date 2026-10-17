import pytest

from core import tts
from tests.unit.test_api import make, tenant


def test_voice_catalogue_and_language_mapping():
    assert tts.lang_code_of("ef_dora") == "e" and tts.language_of("ef_dora") == "es"
    assert tts.language_of("zm_yunxi") == "zh" and tts.language_of("bf_emma") == "en"
    assert tts.lang_code_of("nonsense") == "a"                       # unknown names fall back to English
    assert len(tts.VOICES) == len(set(tts.VOICES)) > 50
    assert set(tts.DEFAULT_VOICE.values()) <= set(tts.VOICES)         # every default voice exists
    assert {tts.language_of(v) for v in tts.DEFAULT_VOICE.values()} == set(tts.DEFAULT_VOICE)


def test_espeak_languages_are_off_when_the_gpl_fallback_is_off(monkeypatch):
    monkeypatch.setenv("VOX_NO_ESPEAK", "1")
    for code in "efhip":
        assert "espeak" in tts.why_unavailable(code)
    assert tts.why_unavailable("a") == "" and tts.why_unavailable("b") == ""
    with pytest.raises(tts.LanguageUnavailable):
        tts.make_pipeline("e")
    monkeypatch.delenv("VOX_NO_ESPEAK")
    assert tts.why_unavailable("e") == ""


def test_optional_package_languages_say_what_is_missing(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert "pyopenjtalk" in tts.why_unavailable("j") and "jieba" in tts.why_unavailable("z")


def test_pipeline_set_loads_each_language_once_and_only_when_needed():
    loaded = []
    ps = tts.PipelineSet(factory=lambda c: loaded.append(c) or f"pipe-{c}", preload="a")
    assert loaded == ["a"]
    assert ps.for_voice("af_heart") == "pipe-a" and ps.for_voice("ef_dora") == "pipe-e"
    assert ps.for_voice("em_alex") == "pipe-e"
    assert loaded == ["a", "e"]


def agent(c, key, **body):
    return c.post("/v1/agents", headers={"X-API-Key": key}, json={"name": "a", **body})


def test_language_selects_that_languages_default_voice(monkeypatch):
    monkeypatch.delenv("VOX_NO_ESPEAK", raising=False)
    c, *_ = make()
    key, _ = tenant(c)
    r = agent(c, key, language="es")
    assert r.status_code == 200 and r.json()["config"]["voice"] == "ef_dora" and r.json()["config"]["language"] == "es"
    assert agent(c, key, language="es", voice="em_alex").json()["config"]["voice"] == "em_alex"
    assert agent(c, key).json()["config"]["voice"] == "af_heart"                   # no language: unchanged behaviour


def test_language_validation(monkeypatch):
    monkeypatch.delenv("VOX_NO_ESPEAK", raising=False)
    c, *_ = make()
    key, _ = tenant(c)
    r = agent(c, key, language="es", voice="af_heart")
    assert r.status_code == 422 and "speaks en" in r.json()["detail"] and "ef_dora" in r.json()["detail"]
    assert agent(c, key, language="xx").status_code == 422
    monkeypatch.setenv("VOX_NO_ESPEAK", "1")
    r = agent(c, key, language="fr")
    assert r.status_code == 422 and "espeak" in r.json()["detail"]
    assert agent(c, key, language="en").status_code == 200                         # English stays available


def test_patch_language_switches_voice_and_calls_can_override(monkeypatch):
    monkeypatch.delenv("VOX_NO_ESPEAK", raising=False)
    c, *_ = make()
    key, _ = tenant(c)
    h = {"X-API-Key": key}
    aid = agent(c, key).json()["id"]
    r = c.patch(f"/v1/agents/{aid}", headers=h, json={"language": "fr"})
    assert r.status_code == 200 and r.json()["config"]["voice"] == "ff_siwis"
    r = c.patch(f"/v1/agents/{aid}", headers=h, json={"voice": "af_heart"})         # English voice, French agent
    assert r.status_code == 422
    call = c.post("/v1/calls", headers=h, json={"agentId": aid, "language": "en"})  # per-call override back to English
    assert call.status_code == 200
    stored = c.get(f"/v1/calls/{call.json()['id']}", headers=h)
    assert stored.status_code == 200


def test_voices_endpoint_lists_languages_with_availability(monkeypatch):
    monkeypatch.setenv("VOX_NO_ESPEAK", "1")
    c, *_ = make()
    key, _ = tenant(c)
    v = c.get("/v1/voices", headers={"X-API-Key": key}).json()
    by = {x["code"]: x for x in v["languages"]}
    assert by["en"]["available"] and by["es"]["available"] is False and "espeak" in by["es"]["reason"]
    assert "ef_dora" in v["voices"] and by["es"]["defaultVoice"] == "ef_dora"


def test_agent_language_is_added_to_the_system_prompt():
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD
    from transports.browser import BrowserTransport  # noqa: F401
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "webSocket", {"systemPrompt": "Be brief.", "language": "es"}, {})

    class T:
        out_rate = 24000
        async def send_audio(self, b): ...
        async def clear(self): ...
        async def end(self): ...
    sess = rt.build_session(call, T())
    sp = sess.history[0]["content"]
    assert sp.startswith("Be brief.") and "Spanish" in sp
    from api.runtime import ERROR_PHRASE
    assert sess.cfg.error_phrase == ERROR_PHRASE["es"]
    call2 = store.create_call(tid, None, "webSocket", {"systemPrompt": "Be brief."}, {})
    sess2 = rt.build_session(call2, T())
    assert "Spanish" not in sess2.history[0]["content"]
    assert sess2.cfg.error_phrase == ERROR_PHRASE["en"]
