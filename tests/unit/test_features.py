import asyncio
import json
import os
import time
import wave

import httpx
import numpy as np
import pytest

from api import tools as toolmod
from api.app import Settings
from api.tools import CallContext
from core import dtmf
from core.bm25 import BM25
from core.metrics import Metrics
from core.recorder import CallRecorder
from core.session import ToolCall
from test_api import SETTINGS, make, tenant
from test_session import FakeLLM, FakeTTS, Rig
from transports.browser import BrowserTransport
from transports.twilio import TwilioTransport


def H(k):
    return {"X-API-Key": k}


# ------------------------------------------------------------------ keys / audit
def test_key_create_revoke_rotate_and_last_key_protection():
    c, store, _ = make()
    k1, tid = tenant(c)
    keys = c.get("/v1/keys", headers=H(k1)).json()
    assert len(keys) == 1 and keys[0]["current"] and "apiKey" not in keys[0]
    r = c.delete(f"/v1/keys/{keys[0]['id']}", headers=H(k1))
    assert r.status_code == 409                                           # cannot lock yourself out
    k2 = c.post("/v1/keys", json={"label": "ci"}, headers=H(k1)).json()
    assert k2["apiKey"].startswith("vox_") and c.get("/v1/agents", headers=H(k2["apiKey"])).status_code == 200
    assert c.delete(f"/v1/keys/{keys[0]['id']}", headers=H(k2["apiKey"])).status_code == 200
    assert c.get("/v1/agents", headers=H(k1)).status_code == 401           # revoked key stops working
    rot = c.post(f"/v1/keys/{k2['id']}/rotate", headers=H(k2["apiKey"])).json()
    assert c.get("/v1/agents", headers=H(k2["apiKey"])).status_code == 401
    assert c.get("/v1/agents", headers=H(rot["apiKey"])).status_code == 200
    assert c.post(f"/v1/keys/{k2['id']}/rotate", headers=H(rot["apiKey"])).status_code == 404
    assert k2["apiKey"] not in json.dumps(store._all("SELECT * FROM api_keys"))   # only hashes stored


def test_keys_are_tenant_scoped():
    c, *_ = make()
    ka, _ = tenant(c, "a")
    kb, _ = tenant(c, "b")
    kid = c.get("/v1/keys", headers=H(ka)).json()[0]["id"]
    c.post("/v1/keys", json={}, headers=H(kb))
    assert c.delete(f"/v1/keys/{kid}", headers=H(kb)).status_code == 404
    assert c.post(f"/v1/keys/{kid}/rotate", headers=H(kb)).status_code == 404


def test_audit_log_records_changes_and_is_tenant_scoped():
    c, *_ = make()
    ka, _ = tenant(c, "a")
    kb, _ = tenant(c, "b")
    a = c.post("/v1/agents", json={"name": "n"}, headers=H(ka)).json()
    c.patch(f"/v1/agents/{a['id']}", json={"greeting": "hi"}, headers=H(ka))
    c.put("/v1/tools/t1", json={"spec": "hangUp"}, headers=H(ka))
    log = c.get("/v1/audit", headers=H(ka)).json()
    assert [e["action"] for e in log][:3] == ["tool.put", "agent.update", "agent.create"]
    assert log[1]["detail"] == {"fields": ["greeting"]} and log[0]["actor"].startswith("key_")
    assert c.get("/v1/audit", headers=H(kb)).json() == []
    older = c.get(f"/v1/audit?before={log[0]['id']}&limit=1", headers=H(ka)).json()
    assert len(older) == 1 and older[0]["id"] < log[0]["id"]


# ------------------------------------------------------------------ tool library
def test_tool_library_validation_redaction_and_agent_references():
    c, *_ = make()
    k, _ = tenant(c)
    h = H(k)
    spec = {"name": "lookup", "description": "d", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            "http": {"url": "https://example.com/x", "headers": {"Authorization": "Bearer SECRET"}}}
    assert c.put("/v1/tools/other", json={"spec": spec}, headers=h).status_code == 422       # name mismatch
    assert c.put("/v1/tools/lookup", json={"spec": {**spec, "http": {"url": "ftp://x"}}}, headers=h).status_code == 422
    assert c.put("/v1/tools/lookup", json={"spec": {"library": "x"}}, headers=h).status_code == 422
    r = c.put("/v1/tools/lookup", json={"spec": spec}, headers=h)
    assert r.status_code == 200 and "SECRET" not in r.text
    assert "SECRET" not in c.get("/v1/tools", headers=h).text and "SECRET" not in c.get("/v1/tools/lookup", headers=h).text
    # agents reference it by name
    assert c.post("/v1/agents", json={"name": "a", "tools": [{"library": "nope"}]}, headers=h).status_code == 422
    a = c.post("/v1/agents", json={"name": "a", "tools": [{"library": "lookup"}, "hangUp"]}, headers=h)
    assert a.status_code == 200
    assert c.delete("/v1/tools/lookup", headers=h).status_code == 409                       # in use
    c.patch(f"/v1/agents/{a.json()['id']}", json={"tools": []}, headers=h)
    assert c.delete("/v1/tools/lookup", headers=h).status_code == 200
    assert c.get("/v1/tools/lookup", headers=h).status_code == 404


def test_library_tool_is_resolved_at_call_time_so_edits_propagate():
    c, store, _ = make()
    k, tid = tenant(c)
    h = H(k)
    c.put("/v1/tools/bye", json={"spec": {"builtin": "hangUp", "strict": True}}, headers=h)
    a = c.post("/v1/agents", json={"name": "a", "tools": [{"library": "bye"}]}, headers=h).json()
    call = c.post("/v1/calls", json={"agentId": a["id"]}, headers=h).json()
    from api.runtime import Runtime
    rt = Runtime(object(), lambda v: None, lambda: None, store)

    class T:
        out_rate = 8000
        call_sid = ""

        async def end(self):
            pass
    specs = toolmod.resolve_library(store.get_call_any(call["id"])["config"]["tools"],
                                    lambda n: (store.get_tool(tid, n) or {}).get("spec"))
    assert specs == [{"builtin": "hangUp", "strict": True}]
    c.put("/v1/tools/bye", json={"spec": {"builtin": "hangUp", "strict": False}}, headers=h)
    specs = toolmod.resolve_library(store.get_call_any(call["id"])["config"]["tools"],
                                    lambda n: (store.get_tool(tid, n) or {}).get("spec"))
    assert specs == [{"builtin": "hangUp", "strict": False}]


# ------------------------------------------------------------------ knowledge base
DOCS = [{"id": "d1", "title": "Opening hours", "text": "We are open Monday to Friday from 9 am to 5 pm. Closed on weekends and public holidays."},
        {"id": "d2", "title": "Insurance", "text": "We accept Delta Dental and Cigna. Processing a benefits check usually takes five to ten minutes."},
        {"id": "d3", "title": "Parking", "text": "Free parking is available behind the building. " + "Filler words. " * 200 + "The garage code is 4721."}]


def test_bm25_ranks_relevant_passage_and_splits_long_documents():
    idx = BM25(DOCS)
    assert idx.search("when are you open", 1)[0]["title"] == "Opening hours"
    assert idx.search("do you take Cigna insurance", 1)[0]["title"] == "Insurance"
    hit = idx.search("garage code", 1)[0]
    assert "4721" in hit["text"] and len(hit["text"].split()) <= 95          # the passage, not the whole document
    assert idx.search("xyzzy quantum", 3) == [] and BM25([]).search("anything") == []


def test_knowledge_base_api_and_isolation():
    c, *_ = make()
    ka, _ = tenant(c, "a")
    kb, _ = tenant(c, "b")
    ha, hb = H(ka), H(kb)
    assert c.post("/v1/corpora", json={"name": "faq"}, headers=ha).status_code == 200
    assert c.post("/v1/corpora", json={"name": "faq"}, headers=ha).status_code == 409
    assert c.post("/v1/corpora", json={"name": "bad name"}, headers=ha).status_code == 422
    assert c.post("/v1/corpora", json={"name": "faq"}, headers=hb).status_code == 200      # same name, other tenant
    d = c.post("/v1/corpora/faq/documents", json={"title": "Hours", "text": "Open nine to five on weekdays."}, headers=ha).json()
    assert c.get("/v1/corpora", headers=ha).json()[0]["docs"] == 1 and c.get("/v1/corpora", headers=hb).json()[0]["docs"] == 0
    assert c.post("/v1/corpora/faq/search", json={"query": "weekday hours"}, headers=ha).json()["results"][0]["title"] == "Hours"
    assert c.post("/v1/corpora/faq/search", json={"query": "weekday hours"}, headers=hb).json()["results"] == []
    assert c.delete(f"/v1/corpora/faq/documents/{d['id']}", headers=hb).status_code == 404
    assert c.post("/v1/corpora/nope/documents", json={"title": "t", "text": "x"}, headers=ha).status_code == 404
    assert c.delete("/v1/corpora/faq", headers=ha).status_code == 200
    assert c.get("/v1/corpora/faq/documents", headers=ha).status_code == 404


async def test_query_corpus_tool_searches_and_picks_up_new_documents():
    store_docs = {"v": "1", "docs": list(DOCS[:2])}
    ctx = CallContext(docs=lambda name: (store_docs["v"], store_docs["docs"]) if name == "faq" else (None, []))
    schemas, fns = toolmod.build([{"builtin": "queryCorpus", "corpus": "faq", "topK": 1}], ctx)
    assert schemas[0]["function"]["name"] == "queryCorpus"
    out = await fns["queryCorpus"]({"query": "weekend hours"})
    assert out["results"][0]["title"] == "Opening hours"
    assert (await fns["queryCorpus"]({"query": "garage code"}))["results"] == []
    store_docs.update(v="2", docs=DOCS)                                   # a document was added: version changes
    assert "4721" in (await fns["queryCorpus"]({"query": "garage code"}))["results"][0]["text"]
    _, fns2 = toolmod.build([{"builtin": "queryCorpus", "corpus": "missing"}], ctx)
    assert "not found" in (await fns2["queryCorpus"]({"query": "x"}))["error"]


def test_query_corpus_validation():
    with pytest.raises(toolmod.ToolConfigError):
        toolmod.validate_specs([{"builtin": "queryCorpus"}])
    with pytest.raises(toolmod.ToolConfigError):
        toolmod.validate_specs([{"builtin": "queryCorpus", "corpus": "faq", "topK": 50}])


# ------------------------------------------------------------------ usage / voices
def test_usage_breakdown_and_voices_and_speed_validation():
    c, store, _ = make()
    k, tid = tenant(c)
    h = H(k)
    a = c.post("/v1/agents", json={"name": "a", "speed": 1.2}, headers=h).json()
    assert a["config"]["speed"] == 1.2
    assert c.post("/v1/agents", json={"name": "a", "speed": 3}, headers=h).status_code == 422
    assert c.post("/v1/agents", json={"name": "a", "voice": "nonsense"}, headers=h).status_code == 422
    assert c.patch(f"/v1/agents/{a['id']}", json={"voice": "nonsense"}, headers=h).status_code == 422
    assert "af_heart" in c.get("/v1/voices", headers=h).json()["voices"]
    now = time.time()
    for secs in (60, 120):
        call = store.create_call(tid, a["id"], "webSocket", {})
        store.update_call(call["id"], status="ended", joined=now - secs, ended=now)
    u = c.get("/v1/usage?days=7", headers=h).json()
    assert u["calls"] == 2 and u["minutes"] == 3.0
    assert u["byAgent"] == [{"key": a["id"], "calls": 2, "minutes": 3.0}] and len(u["byDay"]) >= 1


# ------------------------------------------------------------------ DTMF
def test_dtmf_tones_decode_back_to_their_digits():
    for d in "0123456789*#ABCD":
        assert dtmf.detect(dtmf.tone(d)) == d
    seq = dtmf.sequence("12w3")
    assert len(seq) == 3 * (960 + 640) + 4000                                 # 3 tones + gaps + 500 ms pause at 8 kHz


async def test_play_dtmf_tool_validates_and_sends():
    sent = []

    async def send(d):
        sent.append(d)
    schemas, fns = toolmod.build(["playDtmf"], CallContext(send_dtmf=send))
    assert schemas[0]["function"]["name"] == "playDtmf"
    assert await fns["playDtmf"]({"digits": "12#"}) == {"pressed": "12#"} and sent == ["12#"]
    assert "error" in await fns["playDtmf"]({"digits": "12;rm"})
    assert "error" in await fns["playDtmf"]({"digits": ""})
    _, fns2 = toolmod.build(["playDtmf"], CallContext())
    assert "only available on phone calls" in (await fns2["playDtmf"]({"digits": "1"}))["error"]


class DtmfSession:
    def __init__(self):
        self.texts, self.closed = [], False

    async def start(self):
        pass

    async def feed(self, f):
        pass

    async def inject_text(self, t):
        self.texts.append(t)

    async def close(self):
        self.closed = True


class ScriptWS:
    """Delivers scripted Twilio messages, optionally with delays, then EOF."""
    def __init__(self, script):
        self.script, self.sent = list(script), []

    async def receive_text(self):
        if not self.script:
            raise EOFError
        item = self.script.pop(0)
        if isinstance(item, float):
            await asyncio.sleep(item)
            return await self.receive_text()
        return json.dumps(item)

    async def send_text(self, s):
        self.sent.append(json.loads(s))


def dt(d):
    return {"event": "dtmf", "streamSid": "MZ", "dtmf": {"track": "inbound_track", "digit": d}}


async def run_twilio(script):
    sess = DtmfSession()

    async def factory(tr, custom):
        return sess
    ws = ScriptWS([{"event": "start", "streamSid": "MZ", "start": {"streamSid": "MZ", "callSid": "CA"}}, *script])
    t = TwilioTransport(ws, factory)
    await t.run()
    return sess, ws, t


async def test_keypad_digits_are_collected_into_one_text_turn_on_hash():
    sess, *_ = await run_twilio([dt("1"), dt("2"), dt("3"), dt("#"), 0.05])
    assert sess.texts == ["[The caller pressed these keypad digits: 123]"]


async def test_keypad_digits_flush_after_idle_pause(monkeypatch):
    import transports.twilio as tw
    monkeypatch.setattr(tw, "DTMF_IDLE_S", 0.05)
    sess, *_ = await run_twilio([dt("9"), dt("8"), 0.2])
    assert sess.texts == ["[The caller pressed these keypad digits: 98]"]


async def test_transport_sends_dtmf_tones_as_audio():
    ws = ScriptWS([])
    t = TwilioTransport(ws, None)
    t.stream_sid = "MZ"
    await t.send_dtmf("5")
    media = [m for m in ws.sent if m["event"] == "media"]
    assert media and any(m["event"] == "mark" for m in ws.sent)
    from core.audio import mulaw_to_pcm16
    import base64
    pcm = mulaw_to_pcm16(base64.b64decode(media[0]["media"]["payload"]))
    assert dtmf.detect(pcm[:960]) == "5"


async def test_session_text_turn_gets_a_reply_and_enters_history():
    llm = FakeLLM([["Got it."]])
    r = Rig(llm)
    await r.s.inject_text("[The caller pressed these keypad digits: 42]")
    await r.settle()
    assert llm.calls[0]["history"][-1] == {"role": "user", "content": "[The caller pressed these keypad digits: 42]"}
    assert llm.calls[0]["n_audio"] == 0
    assert r.tts.texts == ["Got it."] and r.s.history[-1] == {"role": "assistant", "content": "Got it."}
    assert r.s.stats["turns"] == 1


def test_vllm_client_sends_no_audio_part_for_a_text_turn():
    from core.llm import VllmClient
    c = VllmClient(client=object())
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "[digits 1]"}]
    assert c.build_messages(hist, np.zeros(0, np.int16), []) == hist
    assert c.build_messages(hist, np.ones(100, np.int16), [])[-1]["content"][0]["type"] == "input_audio"


# ------------------------------------------------------------------ voicemail
async def test_leave_voicemail_tool_defers_message_and_hangup_until_reply_done():
    said = []

    async def voicemail(msg):
        said.append(msg)
    schemas, fns = toolmod.build([{"builtin": "leaveVoicemail", "message": "Hi, call me back."}], CallContext(voicemail=voicemail))
    r = await fns["leaveVoicemail"]({})
    assert said == []                                                     # nothing happens until the Session runs `after`
    assert r.strict and r.after is not None
    await r.after()
    assert said == ["Hi, call me back."]
    for bad in ({"builtin": "leaveVoicemail"}, {"builtin": "leaveVoicemail", "message": "x" * 601}):
        with pytest.raises(toolmod.ToolConfigError):
            toolmod.validate_specs([bad])


async def test_session_say_and_wait_returns_after_playback():
    r = Rig(FakeLLM([["x."]]), tts=FakeTTS(word_ms=100))
    task = asyncio.create_task(r.s.say_and_wait("Please call us back."))
    await r.settle()
    assert not task.done()                      # audio still "playing" on the fake clock
    r.clk.t += 5000
    await asyncio.wait_for(task, 2)
    assert r.tts.texts == ["Please call us back."] and r.sent


# ------------------------------------------------------------------ client tools
class FakeBrowserWS:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.sent = []

    async def iter_messages(self):
        while True:
            m = await self.q.get()
            if m is None:
                return
            yield m

    async def send_bytes(self, b):
        self.sent.append(b)

    async def send_text(self, s):
        self.sent.append(json.loads(s))


async def test_client_tool_roundtrip_error_and_timeout():
    ws = FakeBrowserWS()

    async def factory(tr):
        return DtmfSession()
    tr = BrowserTransport(ws, factory)
    runner = asyncio.create_task(tr.run())
    await asyncio.sleep(0.01)
    spec = {"builtin": "client", "name": "open_page", "description": "Open a page", "parameters": {
        "type": "object", "properties": {"url": {"type": "string"}}}, "timeoutS": 0.5}
    schemas, fns = toolmod.build([spec], CallContext(client_call=tr.client_call))
    assert schemas[0]["function"]["name"] == "open_page"
    call = asyncio.create_task(fns["open_page"]({"url": "/pricing"}))
    await asyncio.sleep(0.01)
    inv = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "client_tool_invocation"][0]
    assert inv["name"] == "open_page" and inv["args"] == {"url": "/pricing"}
    await ws.q.put(("text", json.dumps({"type": "client_tool_result", "id": inv["id"], "result": {"ok": True}})))
    assert await call == {"ok": True}
    c2 = asyncio.create_task(fns["open_page"]({}))
    await asyncio.sleep(0.01)
    inv2 = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "client_tool_invocation"][1]
    await ws.q.put(("text", json.dumps({"type": "client_tool_result", "id": inv2["id"], "error": "blocked"})))
    assert await c2 == {"error": "blocked"}
    assert "did not answer" in (await fns["open_page"]({}))["error"]       # nobody answers: times out
    await ws.q.put(("text", "not json"))                                   # junk text frames are ignored
    await ws.q.put(None)
    await asyncio.wait_for(runner, 2)
    _, f3 = toolmod.build([spec], CallContext())
    assert "browser" in (await f3["open_page"]({}))["error"]


def test_client_tool_roundtrip_over_a_real_websocket():
    """Starlette adapter: text frames reach the transport (previously only binary frames were read)."""
    c, store, _ = make()
    k, _ = tenant(c)
    spec = {"builtin": "client", "name": "ping", "description": "ping the page", "timeoutS": 3}
    call = c.post("/v1/calls", json={"systemPrompt": "p", "tools": [spec]}, headers=H(k)).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    rt = c.app.state.runtime if hasattr(c.app.state, "runtime") else None
    with c.websocket_connect(path) as ws:
        ws.send_text(json.dumps({"type": "client_tool_result", "id": "zzz", "result": 1}))   # unknown id: ignored
        ws.send_bytes(np.zeros(512, dtype="<i2").tobytes())
    assert store.get_call_any(call["id"])["status"] == "ended"


def test_client_tool_validation():
    for bad in ({"builtin": "client"}, {"builtin": "client", "name": "x"},
                {"builtin": "client", "name": "x y", "description": "d"},
                {"builtin": "client", "name": "x", "description": "d", "timeoutS": 999}):
        with pytest.raises(toolmod.ToolConfigError):
            toolmod.validate_specs([bad])


# ------------------------------------------------------------------ recording
def test_recorder_aligns_channels_and_truncates_on_clear(tmp_path):
    t = [0.0]
    rec = CallRecorder(clock=lambda: t[0])
    for i in range(10):                                          # 10 frames of caller audio over 0.32 s
        rec.mic(np.full(512, 1000, np.int16))
        t[0] += 0.032
    rec.agent(np.full(8000, 2000, np.int16), 8000)               # 1 s of agent audio starting now (0.32 s)
    t[0] += 0.5                                                  # 0.5 s of it has played
    rec.clear()                                                  # caller interrupts
    st = rec.stereo()
    start = int(0.32 * 16000)
    assert st[0, 0] == 1000 and st[start - 1, 1] == 0 and st[start + 100, 1] != 0
    assert len(st) >= start + int(0.5 * 16000) - 5 and st[:, 1].nonzero()[0].max() < start + int(0.5 * 16000) + 5
    secs = rec.save(str(tmp_path / "a" / "c.wav"))
    with wave.open(str(tmp_path / "a" / "c.wav")) as w:
        assert w.getnchannels() == 2 and w.getframerate() == 16000 and w.getnframes() == len(st)
    assert secs == pytest.approx(len(st) / 16000)


def test_recorder_queues_back_to_back_agent_audio():
    t = [0.0]
    rec = CallRecorder(clock=lambda: t[0])
    rec.agent(np.full(1600, 7, np.int16), 16000)
    rec.agent(np.full(1600, 9, np.int16), 16000)                 # sent immediately after: plays after the first
    st = rec.stereo()
    assert st[0, 1] == 7 and st[1599, 1] == 7 and st[1600, 1] == 9 and len(st) == 3200


def test_recording_end_to_end_and_download(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX_RECORD_DIR", str(tmp_path))
    from test_api import speech, silence
    c, store, _ = make()
    ka, _ = tenant(c, "a")
    kb, _ = tenant(c, "b")
    a = c.post("/v1/agents", json={"name": "rec", "record": True}, headers=H(ka)).json()
    call = c.post("/v1/calls", json={"agentId": a["id"]}, headers=H(ka)).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    with c.websocket_connect(path) as ws:
        got = b""
        for f in silence(5) + speech(15) + silence(40):
            ws.send_bytes(f)
        t0 = time.time()
        while not got and time.time() - t0 < 5:
            m = ws.receive()
            got += m.get("bytes") or b""
    r = c.get(f"/v1/calls/{call['id']}/recording", headers=H(ka))
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert c.get(f"/v1/calls/{call['id']}/recording", headers=H(kb)).status_code == 404       # tenant isolation
    plain = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=H(ka)).json()            # record not set
    with c.websocket_connect(plain["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        ws.send_bytes(silence(1)[0])
    assert c.get(f"/v1/calls/{plain['id']}/recording", headers=H(ka)).status_code == 404


# ------------------------------------------------------------------ metrics
def test_metrics_text_format():
    m = Metrics()
    m.inc("c_total", 2, kind='a"b')
    m.observe("lat_ms", 120, buckets=(100, 200))
    m.observe("lat_ms", 50, buckets=(100, 200))
    m.gauge("g", lambda: 3)
    txt = m.render()
    assert 'c_total{kind="a\\"b"} 2.0' in txt
    assert 'lat_ms_bucket{le="100"} 1' in txt and 'lat_ms_bucket{le="200"} 2' in txt and 'lat_ms_bucket{le="+Inf"} 2' in txt
    assert "lat_ms_sum 170" in txt and "lat_ms_count 2" in txt and "g 3.0" in txt


def test_metrics_endpoint_is_protected_and_counts_calls():
    from test_api import speech, silence
    c, *_ = make(Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok", metrics_token="mt"))
    assert c.get("/metrics").status_code == 403
    assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 403
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=H(k)).json()
    with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        for f in silence(5) + speech(15) + silence(40):
            ws.send_bytes(f)
        t0, got = time.time(), b""
        while not got and time.time() - t0 < 5:
            got += ws.receive().get("bytes") or b""
    txt = c.get("/metrics", headers={"Authorization": "Bearer mt"}).text
    assert 'vox_calls_total{medium="webSocket",reason="hangup"}' in txt and "vox_first_audio_ms_count" in txt
    assert "vox_active_calls" in txt
    assert c.get("/metrics", headers={"X-Admin-Key": "adm"}).status_code == 200
    c2, *_ = make(Settings(admin_key="", metrics_token=""))
    assert c2.get("/metrics").status_code == 403                          # unconfigured = closed


# ------------------------------------------------------------------ outbound calls
def make_outbound(handler, **kw):
    from api import security
    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from starlette.testclient import TestClient
    from test_api import FakeLLM as L, FakeTTS as T, FakeVAD
    store = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(L(), lambda v: T(), FakeVAD, store, http=http)
    s = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                 twilio_account_sid="ACtest", twilio_number="+15550001111", **kw)
    return TestClient(create_app(store, rt, security.ConcurrencyLimiter(2), None, s)), store, rt


def test_outbound_call_places_twilio_request_with_stream_twiml():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], req.content.decode()
        return httpx.Response(201, json={"sid": "CA123"})
    c, store, _ = make_outbound(handler)
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "voicemailMessage": "Call back please",
                                           "record": True}, headers=H(k))
    assert r.status_code == 200 and r.json()["callSid"] == "CA123"
    assert seen["url"] == "https://api.twilio.com/2010-04-01/Accounts/ACtest/Calls.json" and seen["auth"].startswith("Basic ")
    from urllib.parse import parse_qs
    form = {k_: v[0] for k_, v in parse_qs(seen["body"]).items()}
    assert form["To"] == "+15552223333" and form["From"] == "+15550001111" and "wss://vox.example.com/ws/twilio/" in form["Twiml"]
    assert 'name="token"' in form["Twiml"] and form["MachineDetection"] == "DetectMessageEnd"
    assert form["StatusCallback"].endswith(f"/twilio/status/{r.json()['id']}") and "/twilio/amd/" in form["AsyncAmdStatusCallback"]
    call = store.get_call_any(r.json()["id"])
    assert call["medium"] == "twilio" and call["config"]["record"] is True
    assert call["metadata"]["direction"] == "outbound" and call["metadata"]["callSid"] == "CA123"


def test_outbound_validation_limits_and_twilio_errors():
    def ok(req):
        return httpx.Response(201, json={"sid": "CA1"})
    c, store, _ = make_outbound(ok, outbound_per_day=2, outbound_prefixes=("+1",))
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()

    def go(to, **kw):
        return c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": to, **kw}, headers=H(k))
    assert go("5551234").status_code == 422 and go("+15551234567;rm").status_code == 422
    assert go("+447700900123").status_code == 403                       # prefix allow-list
    assert c.post("/v1/calls/outbound", json={"agentId": "agt_x", "to": "+15551234567"}, headers=H(k)).status_code == 404
    assert go("+15551234567").status_code == 200 and go("+15551234568").status_code == 200
    assert go("+15551234569").status_code == 429                        # daily cap
    c2, *_ = make_outbound(lambda r: httpx.Response(400, json={"message": "unverified number"}))
    k2, _ = tenant(c2)
    a2 = c2.post("/v1/agents", json={"name": "a"}, headers=H(k2)).json()
    r = c2.post("/v1/calls/outbound", json={"agentId": a2["id"], "to": "+15551234567"}, headers=H(k2))
    assert r.status_code == 502 and "unverified number" in r.text
    assert c2.get("/v1/calls", headers=H(k2)).json()[0]["end_reason"] == "twilio_rejected"
    c3, *_ = make(SETTINGS)
    k3, _ = tenant(c3)
    assert c3.post("/v1/calls/outbound", json={"agentId": "a", "to": "+15551234567"}, headers=H(k3)).status_code == 503


def test_outbound_status_and_amd_callbacks_need_a_valid_signature():
    from api import security
    c, store, rt = make_outbound(lambda r: httpx.Response(201, json={"sid": "CA1"}))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    o = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567", "voicemailMessage": "hi"}, headers=H(k)).json()
    url = f"https://vox.example.com/twilio/status/{o['id']}"
    form = {"CallStatus": "no-answer"}
    assert c.post(f"/twilio/status/{o['id']}", data=form).status_code == 403
    sig = security.twilio_signature("tok", url, form)
    assert c.post(f"/twilio/status/{o['id']}", data=form, headers={"X-Twilio-Signature": sig}).status_code == 200
    cur = store.get_call_any(o["id"])
    assert cur["status"] == "ended" and cur["end_reason"] == "no-answer"
    # AMD: a machine triggers the voicemail message on the live session; a human does nothing
    spoke = []

    class S:
        async def say_and_wait(self, m):
            spoke.append(m)

    class Tr:
        async def end(self):
            spoke.append("END")
    rt.live[o["id"]] = (S(), Tr())
    o2 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567", "voicemailMessage": "hi"}, headers=H(k)).json()
    rt.live[o2["id"]] = (S(), Tr())
    url2 = f"https://vox.example.com/twilio/amd/{o2['id']}"
    human = {"AnsweredBy": "human"}
    r = c.post(f"/twilio/amd/{o2['id']}", data=human, headers={"X-Twilio-Signature": security.twilio_signature("tok", url2, human)})
    assert r.json()["action"] == "none" and spoke == []
    machine = {"AnsweredBy": "machine_end_beep"}
    r = c.post(f"/twilio/amd/{o2['id']}", data=machine, headers={"X-Twilio-Signature": security.twilio_signature("tok", url2, machine)})
    assert r.json()["action"] == "voicemail"
    time.sleep(0.3)
    assert spoke == ["hi", "END"]


def test_sip_outbound_originates_on_asterisk_and_not_twilio(monkeypatch):
    from urllib.parse import parse_qs, urlparse
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        if "twilio" in str(req.url):
            return httpx.Response(201, json={"sid": "CA123"})
        return httpx.Response(200, json={"id": "chan-1"})
    c, store, rt = make_outbound(handler, sip_token="s3cret")
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "channel": "sip",
                                           "voicemailMessage": "leave this"}, headers=H(k))
    assert r.status_code == 200, r.text
    url = str(seen[-1].url)
    assert "api.twilio.com" not in url and url.startswith("http://asterisk:8088/ari/channels")
    q = {k_: v[0] for k_, v in parse_qs(urlparse(url).query).items()}
    assert q["endpoint"] == "PJSIP/+15552223333@voxsun" and q["app"] == "vox"
    assert q["appArgs"] == f"amd,{r.json()['id']}" and q["callerId"] == "+15550001111"
    call = store.get_call_any(r.json()["id"])
    assert call["medium"] == "sip" and call["metadata"]["callSid"] == "chan-1"
    # default channel stays on Twilio even when ARI is configured
    r2 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552224444"}, headers=H(k))
    assert r2.status_code == 200 and "api.twilio.com" in str(seen[-1].url)
    assert store.get_call_any(r2.json()["id"])["medium"] == "twilio"
    # no voicemail: no amd appArgs
    r3 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552225555", "channel": "SIP"}, headers=H(k))
    assert r3.status_code == 200
    q3 = parse_qs(urlparse(str(seen[-1].url)).query)
    assert "appArgs" not in q3 and store.get_call_any(r3.json()["id"])["medium"] == "sip"
    assert c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552226666", "channel": "sms"},
                  headers=H(k)).status_code == 422

    # AMD: token required; machine_end on a live call speaks the message
    spoke = []

    class S:
        async def say_and_wait(self, m):
            spoke.append(m)

    class Tr:
        async def end(self):
            spoke.append("END")
    cid = r.json()["id"]
    assert c.post(f"/sip/amd/{cid}", data={"AnsweredBy": "machine_end_beep"}).status_code == 403
    assert c.post(f"/sip/amd/missing", json={"AnsweredBy": "human"},
                  headers={"X-Vox-Sip-Token": "s3cret"}).status_code == 404
    human = c.post(f"/sip/amd/{cid}", json={"AnsweredBy": "human"}, headers={"X-Vox-Sip-Token": "s3cret"})
    assert human.json()["action"] == "none" and store.get_call_any(cid)["metadata"]["answeredBy"] == "human"
    rt.live[cid] = (S(), Tr())
    machine = c.post(f"/sip/amd/{cid}", data={"AnsweredBy": "machine_end_beep"},
                     headers={"X-Vox-Sip-Token": "s3cret"})
    assert machine.json()["action"] == "voicemail"
    time.sleep(0.3)
    assert spoke == ["leave this", "END"]
    assert store.get_call_any(cid)["metadata"]["answeredBy"] == "machine_end_beep"
    # endpoint off when the token is unset, even if a header is sent
    c2, store2, _ = make_outbound(handler)
    k2, tid2 = tenant(c2)
    row = store2.create_call(tid2, None, "sip", {}, {"voicemail": "x"})
    assert c2.post(f"/sip/amd/{row['id']}", data={"AnsweredBy": "machine_end_beep"},
                   headers={"X-Vox-Sip-Token": "s3cret"}).status_code == 404


def test_sip_amd_hangs_up_when_the_voicemail_cannot_be_spoken(monkeypatch):
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    c, store, rt = make_outbound(lambda req: httpx.Response(200, json={"id": "chan-9"}), sip_token="s3cret")
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "channel": "sip",
                                           "voicemailMessage": "leave this"}, headers=H(k))
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    spoke = []

    class S:
        async def say_and_wait(self, m):
            spoke.append(m)
            raise RuntimeError("tts down")

    class Tr:
        async def end(self):
            spoke.append("END")
    rt.live[cid] = (S(), Tr())
    res = c.post(f"/sip/amd/{cid}", json={"AnsweredBy": "machine_end_beep"}, headers={"X-Vox-Sip-Token": "s3cret"})
    assert res.json()["action"] == "voicemail"
    deadline = time.time() + 2
    while spoke != ["leave this", "END"] and time.time() < deadline:
        time.sleep(0.05)
    assert spoke == ["leave this", "END"]


def test_sip_outbound_without_ari_is_503_and_stores_nothing(monkeypatch):
    monkeypatch.delenv("ASTERISK_ARI_URL", raising=False)
    c, store, _ = make_outbound(lambda req: httpx.Response(500, json={"message": "no"}))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "channel": "sip"}, headers=H(k))
    assert r.status_code == 503 and "ASTERISK_ARI_URL" in r.text
    assert store.list_calls(tid) == []
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")

    def down(req):
        raise RuntimeError("down")
    c3, store3, _ = make_outbound(down)
    k3, tid3 = tenant(c3)
    a3 = c3.post("/v1/agents", json={"name": "a"}, headers=H(k3)).json()
    r = c3.post("/v1/calls/outbound", json={"agentId": a3["id"], "to": "+15552223333", "channel": "sip"}, headers=H(k3))
    assert r.status_code == 502 and "could not reach Asterisk" in r.text and "Twilio" not in r.text
    assert store3.list_calls(tid3)[0]["end_reason"] == "asterisk_unreachable"
    c4, store4, _ = make_outbound(lambda req: httpx.Response(500, json={"message": "no endpoint"}))
    k4, tid4 = tenant(c4)
    a4 = c4.post("/v1/agents", json={"name": "a"}, headers=H(k4)).json()
    r = c4.post("/v1/calls/outbound", json={"agentId": a4["id"], "to": "+15552223333", "channel": "sip",
                                            "voicemailMessage": "hi"}, headers=H(k4))
    assert r.status_code == 502 and "Asterisk refused" in r.text and "no endpoint" in r.text
    assert store4.list_calls(tid4)[0]["end_reason"] == "asterisk_rejected"


def test_call_metadata_merge_keeps_both_keys():
    """Two writers must not replace the document. callSid and answeredBy both stay, with the keys already stored."""
    import threading
    from api.store import Store
    store = Store()
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "sip", {}, {"deal": "42", "voicemail": "hi", "direction": "outbound"})

    def write_sid():
        for _ in range(40):
            store.merge_call_metadata(call["id"], {"callSid": "chan-9"})

    def write_amd():
        for _ in range(40):
            store.merge_call_metadata(call["id"], {"answeredBy": "machine_end_beep"})

    threads = [threading.Thread(target=write_sid), threading.Thread(target=write_amd)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    meta = store.get_call_any(call["id"])["metadata"]
    assert meta["callSid"] == "chan-9" and meta["answeredBy"] == "machine_end_beep"
    assert meta["deal"] == "42" and meta["voicemail"] == "hi" and meta["direction"] == "outbound"
    assert store.find_sip_call(call_sid="chan-9")["id"] == call["id"]


def test_sip_answered_by_survives_call_sid_and_hangs_up_when_not_live(monkeypatch):
    """AnsweredBy posted during originate must still be there after callSid is saved, and a machine with no
    session must DELETE that ARI channel. A live session is spoken to and the channel is not deleted."""
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    app_box = {}
    deleted = []
    race = {"during": True}

    async def handler(req: httpx.Request):
        if req.method == "DELETE":
            deleted.append(str(req.url))
            return httpx.Response(204)
        args = req.url.params.get("appArgs", "")
        cid = args.split(",", 1)[1] if str(args).startswith("amd,") else ""
        if cid and race["during"]:
            transport = httpx.ASGITransport(app=app_box["app"])
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.post(f"/sip/amd/{cid}", data={"AnsweredBy": "machine_end_beep"},
                                        headers={"X-Vox-Sip-Token": "s3cret"})
                assert res.status_code == 200, res.text
                race["amd"] = res.json()
                during = app_box["store"].get_call_any(cid)["metadata"]
                assert during["answeredBy"] == "machine_end_beep" and "callSid" not in during
        return httpx.Response(200, json={"id": race["sid"]})

    race["sid"] = "chan-9"
    c, store, rt = make_outbound(handler, sip_token="s3cret")
    app_box["app"] = c.app
    app_box["store"] = store
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "channel": "sip",
                                           "voicemailMessage": "leave this", "metadata": {"deal": "42"}}, headers=H(k))
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    meta = store.get_call_any(cid)["metadata"]
    assert meta["answeredBy"] == "machine_end_beep" and meta["callSid"] == "chan-9"
    assert meta["deal"] == "42" and meta["voicemail"] == "leave this"
    assert meta["from"] == "+15550001111" and meta["to"] == "+15552223333" and meta["direction"] == "outbound"
    assert race["amd"]["action"] == "none"
    assert deleted == ["http://asterisk:8088/ari/channels/chan-9"]
    assert store.get_call_any(cid)["status"] == "ended"
    assert cid not in rt.live

    # The other order: callSid is already stored, the call is not live, machine_end still deletes that channel.
    race["during"] = False
    race["sid"] = "chan-2"
    r2 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552224444", "channel": "sip",
                                            "voicemailMessage": "leave this", "metadata": {"deal": "7"}}, headers=H(k))
    assert r2.status_code == 200, r2.text
    cid2 = r2.json()["id"]
    assert store.get_call_any(cid2)["metadata"]["callSid"] == "chan-2"
    assert "answeredBy" not in store.get_call_any(cid2)["metadata"]
    n = len(deleted)
    human = c.post(f"/sip/amd/{cid2}", json={"AnsweredBy": "human"}, headers={"X-Vox-Sip-Token": "s3cret"})
    assert human.json()["action"] == "none"
    assert store.get_call_any(cid2)["metadata"]["callSid"] == "chan-2"
    assert store.get_call_any(cid2)["metadata"]["answeredBy"] == "human"
    assert store.get_call_any(cid2)["status"] == "created" and len(deleted) == n
    machine = c.post(f"/sip/amd/{cid2}", data={"AnsweredBy": "machine_end_beep"}, headers={"X-Vox-Sip-Token": "s3cret"})
    assert machine.json()["action"] == "hangup"
    assert deleted[n:] == ["http://asterisk:8088/ari/channels/chan-2"]
    meta2 = store.get_call_any(cid2)["metadata"]
    assert meta2["callSid"] == "chan-2" and meta2["answeredBy"] == "machine_end_beep" and meta2["deal"] == "7"
    assert store.get_call_any(cid2)["status"] == "ended"

    # A live session is spoken to and hung up on the transport. The ARI channel id is not deleted.
    race["sid"] = "chan-3"
    r3 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552225555", "channel": "sip",
                                            "voicemailMessage": "leave this"}, headers=H(k))
    cid3 = r3.json()["id"]
    spoke = []

    class S:
        async def say_and_wait(self, m):
            spoke.append(m)

    class Tr:
        async def end(self):
            spoke.append("END")
    rt.live[cid3] = (S(), Tr())
    n = len(deleted)
    res = c.post(f"/sip/amd/{cid3}", json={"AnsweredBy": "machine_end_beep"}, headers={"X-Vox-Sip-Token": "s3cret"})
    assert res.json()["action"] == "voicemail"
    deadline = time.time() + 2
    while spoke != ["leave this", "END"] and time.time() < deadline:
        time.sleep(0.05)
    assert spoke == ["leave this", "END"] and deleted[n:] == []
    assert store.get_call_any(cid3)["metadata"]["callSid"] == "chan-3"
    assert store.get_call_any(cid3)["status"] == "created"


def test_audiosocket_adopts_outbound_row_and_binds_the_ari_channel(monkeypatch):
    """The AudioSocket UUID is merged onto the outbound row. Transfer sees the ARI id, not the UUID.
    A bind before any row creates the call with that channel id. An unbound UUID is not stored as callSid."""
    import socket
    import uuid
    from api import security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store
    from starlette.testclient import TestClient
    from test_api import FakeLLM, FakeTTS, FakeVAD
    from transports import audiosocket as A
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url)))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"id": "chan-9"})

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store, http=http)
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", twilio_number="+15550001111", sip_token="s3cret",
                        audiosocket_port=port, audiosocket_agent="pending")
    app = create_app(store, rt, security.ConcurrencyLimiter(4), None, settings)
    audio = "33333333-3333-3333-3333-333333333333"
    fresh = "44444444-4444-4444-4444-444444444444"
    unbound = "55555555-5555-5555-5555-555555555555"
    token = {"X-Vox-Sip-Token": "s3cret"}

    def closed_context(fn):
        for cell in fn.__closure__ or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if type(val).__name__ == "CallContext":
                return val
        raise AssertionError("transfer did not close over the call")

    def wait_live(cid):
        deadline = time.time() + 3
        while time.time() < deadline:
            cur = store.get_call_any(cid)
            if cur and cur["status"] == "active" and cid in rt.live:
                return cur
            time.sleep(0.05)
        raise AssertionError(store.get_call_any(cid))

    def wait_row(pred):
        deadline = time.time() + 3
        while time.time() < deadline:
            for row in store.list_calls(tid):
                if pred(row) and row["status"] == "active" and row["id"] in rt.live:
                    return row
            time.sleep(0.05)
        raise AssertionError([row["metadata"] for row in store.list_calls(tid)])

    def open_uuid(u):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        sock.sendall(A.pack(A.KIND_UUID, uuid.UUID(u).bytes))
        return sock

    def hang(sock):
        try:
            sock.sendall(A.pack(A.KIND_HANGUP))
        except OSError:
            pass
        sock.close()

    with TestClient(app) as c:
        k, _ = tenant(c)
        a = c.post("/v1/agents", json={"name": "a", "tools": [
            {"builtin": "transfer", "targets": {"billing": "+15551230001"}}]}, headers=H(k)).json()
        settings.audiosocket_agent = a["id"]
        assert c.post("/sip/bind", data={"uuid": audio, "channelId": "chan-9"}).status_code == 403
        assert c.post("/sip/bind", data={"uuid": "not-a-uuid", "channelId": "chan-9"}, headers=token).status_code == 422
        r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15552223333", "channel": "sip",
                                               "voicemailMessage": "leave this", "metadata": {"deal": "42"}},
                   headers=H(k))
        assert r.status_code == 200, r.text
        cid = r.json()["id"]
        bound = c.post("/sip/bind", data={"uuid": audio, "channelId": "chan-9"}, headers=token)
        assert bound.status_code == 200 and bound.json()["channelId"] == "chan-9"
        meta = store.get_call_any(cid)["metadata"]
        assert meta["audioSocketUuid"] == audio and meta["callSid"] == "chan-9" and meta["voicemail"] == "leave this"
        assert meta["deal"] == "42" and meta["from"] == "+15550001111" and meta["to"] == "+15552223333"
        tid = store.get_call_any(cid)["tenant_id"]
        assert len(store.list_calls(tid)) == 1
        sock = open_uuid(audio)
        try:
            active = wait_live(cid)
            meta = active["metadata"]
            assert meta["callSid"] == "chan-9" and meta["audioSocketUuid"] == audio
            assert meta["voicemail"] == "leave this" and meta["deal"] == "42" and meta["to"] == "+15552223333"
            assert [row["id"] for row in store.list_calls(tid)] == [cid]
            sess, _tr = rt.live[cid]
            assert closed_context(sess.tools["transfer"]).call_sid == "chan-9"
        finally:
            hang(sock)
        ended = None
        deadline = time.time() + 3
        while time.time() < deadline:
            ended = store.get_call_any(cid)
            if ended and ended["status"] == "ended":
                break
            time.sleep(0.05)
        assert ended["status"] == "ended"

        pre = c.post("/sip/bind", data={"uuid": fresh, "channelId": "1730000000.42"}, headers=token)
        assert pre.status_code == 200
        sock2 = open_uuid(fresh)
        try:
            created = wait_row(lambda row: row["metadata"].get("audioSocketUuid") == fresh)
            assert created["id"] != cid
            assert created["metadata"]["callSid"] == "1730000000.42"
            assert created["metadata"].get("voicemail") is None
            sess2, _ = rt.live[created["id"]]
            assert closed_context(sess2.tools["transfer"]).call_sid == "1730000000.42"
        finally:
            hang(sock2)

        sock3 = open_uuid(unbound)
        try:
            extra = wait_row(lambda row: row["metadata"].get("audioSocketUuid") == unbound)
            assert extra["metadata"].get("callSid", "") == ""
            sess3, _ = rt.live[extra["id"]]
            assert closed_context(sess3.tools["transfer"]).call_sid == ""
        finally:
            hang(sock3)
        assert all(method == "POST" and "/redirect" not in url for method, url in seen)
        assert all(audio not in url and fresh not in url and unbound not in url for _method, url in seen)

    c2, *_ = make_outbound(lambda req: httpx.Response(500))
    assert c2.post("/sip/bind", data={"uuid": audio, "channelId": "chan-9"},
                   headers={"X-Vox-Sip-Token": "s3cret"}).status_code == 404


# ------------------------------------------------------------------ draining (scale-in without dropping calls)
def test_drain_stops_new_calls_but_reports_readiness():
    from test_api import speech, silence
    c, store, lim = make()
    k, _ = tenant(c)
    assert c.get("/healthz").status_code == 200
    assert c.post("/admin/drain").status_code == 403
    r = c.post("/admin/drain", headers={"X-Admin-Key": "adm"})
    assert r.json()["draining"] is True
    h = c.get("/healthz")
    assert h.status_code == 503 and h.json()["draining"] is True            # load balancer sees "not ready"
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=H(k)).json()
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")) as ws:
            ws.receive_text()
    assert e.value.code == 1013 and store.get_call_any(call["id"])["end_reason"] == "draining"
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    from api import security
    form = {"From": "+15551230000"}
    sig = security.twilio_signature("tok", f"https://vox.example.com/twilio/voice/{a['id']}", form)
    assert c.post(f"/twilio/voice/{a['id']}", data=form, headers={"X-Twilio-Signature": sig}).status_code == 503
    assert c.post("/admin/undrain", headers={"X-Admin-Key": "adm"}).json() == {"draining": False}
    assert c.get("/healthz").status_code == 200


# ------------------------------------------------------------------ billing rate, CSV, call deletion
def test_rate_cost_csv_and_admin_patch():
    c, store, _ = make()
    k, tid = tenant(c)
    assert c.get("/v1/usage", headers=H(k)).json().get("cost") is None                    # no rate set: no cost claimed
    assert c.patch(f"/admin/tenants/{tid}", json={"ratePerMinute": 0.02}).status_code == 403
    r = c.patch(f"/admin/tenants/{tid}", json={"ratePerMinute": 0.02, "concurrencyLimit": 7}, headers={"X-Admin-Key": "adm"})
    assert r.json() == {"tenantId": tid, "concurrencyLimit": 7, "ratePerMinute": 0.02, "retentionDays": None}
    assert c.patch("/admin/tenants/tnt_nope", json={"ratePerMinute": 1}, headers={"X-Admin-Key": "adm"}).status_code == 404
    assert c.patch(f"/admin/tenants/{tid}", json={"ratePerMinute": -1}, headers={"X-Admin-Key": "adm"}).status_code == 422
    now = time.time()
    call = store.create_call(tid, None, "webSocket", {})
    store.update_call(call["id"], status="ended", joined=now - 150, ended=now)
    u = c.get("/v1/usage", headers=H(k)).json()
    assert u["minutes"] == 2.5 and u["cost"] == 0.05 and u["byDay"][0]["cost"] == 0.05
    csv = c.get("/v1/usage.csv", headers=H(k)).text.strip().splitlines()
    assert csv[0] == "day,calls,minutes,cost" and csv[1].endswith(",1,2.5,0.05")
    c2, *_ = make()
    k2, t2 = tenant(c2)
    assert c2.get("/v1/usage.csv", headers=H(k2)).text.strip() == "day,calls,minutes,cost"


def test_delete_call_removes_record_transcript_and_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX_RECORD_DIR", str(tmp_path))
    c, store, _ = make()
    ka, ta = tenant(c, "a")
    kb, _ = tenant(c, "b")
    call = store.create_call(ta, None, "webSocket", {})
    store.add_message(call["id"], "user", "secret words", 1.0)
    (tmp_path / f"{call['id']}.wav").write_bytes(b"RIFF....")
    (tmp_path / "utterances").mkdir()
    (tmp_path / "utterances" / f"{call['id']}-001.wav").write_bytes(b"x")
    store.update_call(call["id"], status="active")
    assert c.delete(f"/v1/calls/{call['id']}", headers=H(ka)).status_code == 409           # not while live
    store.update_call(call["id"], status="ended")
    assert c.delete(f"/v1/calls/{call['id']}", headers=H(kb)).status_code == 404           # other tenant
    assert c.delete(f"/v1/calls/{call['id']}", headers=H(ka)).status_code == 200
    assert store.get_call_any(call["id"]) is None and store.messages(call["id"]) == []
    assert not (tmp_path / f"{call['id']}.wav").exists() and not list((tmp_path / "utterances").iterdir())
    assert c.get("/v1/audit", headers=H(ka)).json()[0]["action"] == "call.delete"


def test_old_database_is_migrated_in_place(tmp_path):
    """A DB created by the first release (no key ids, no revocation, no rate) keeps working after an upgrade."""
    import sqlite3
    from api import security
    from api.store import Store
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE tenants(id TEXT PRIMARY KEY, name TEXT, webhook_url TEXT, webhook_secret TEXT, concurrency_limit INTEGER, created REAL);
      CREATE TABLE api_keys(key_hash TEXT PRIMARY KEY, tenant_id TEXT, label TEXT, created REAL);
      CREATE TABLE agents(id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, config TEXT, created REAL);
      CREATE TABLE calls(id TEXT PRIMARY KEY, tenant_id TEXT, agent_id TEXT, medium TEXT, status TEXT, config TEXT, join_token TEXT,
                         metadata TEXT, created REAL, joined REAL, ended REAL, end_reason TEXT);
      CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT, role TEXT, text TEXT, t REAL);
    """)
    db.execute("INSERT INTO tenants VALUES('tnt_1','old',NULL,NULL,NULL,1.0)")
    db.execute("INSERT INTO api_keys VALUES(?, 'tnt_1', 'initial', 1.0)", (security.hash_key("vox_oldkey"),))
    db.commit()
    db.close()
    s = Store(path)
    t = s.tenant_by_key(security.hash_key("vox_oldkey"))
    assert t["id"] == "tnt_1" and t["key_id"].startswith("key_") and t["rate_per_minute"] is None
    kid = s.add_key("tnt_1", security.hash_key("vox_new"), "ci")
    assert s.revoke_key("tnt_1", kid) and s.tenant_by_key(security.hash_key("vox_new")) is None
    s.close()
    assert Store(path).tenant_by_key(security.hash_key("vox_oldkey"))["key_id"] == t["key_id"]      # stable across restarts


# ------------------------------------------------------------------ roles
def test_viewer_developer_admin_roles():
    c, store, _ = make()
    admin, tid = tenant(c)
    ha = H(admin)
    dev = c.post("/v1/keys", json={"label": "ci", "role": "developer"}, headers=ha).json()
    view = c.post("/v1/keys", json={"label": "ro", "role": "viewer"}, headers=ha).json()
    assert dev["role"] == "developer" and c.post("/v1/keys", json={"role": "root"}, headers=ha).status_code == 422
    hd, hv = H(dev["apiKey"]), H(view["apiKey"])
    a = c.post("/v1/agents", json={"name": "a"}, headers=ha).json()
    # viewer: can read, cannot change anything
    assert c.get("/v1/agents", headers=hv).status_code == 200 and c.get("/v1/usage", headers=hv).status_code == 200
    assert c.get("/v1/me", headers=hv).json()["role"] == "viewer"
    for method, path, body in (("post", "/v1/agents", {"name": "x"}), ("patch", f"/v1/agents/{a['id']}", {"greeting": "x"}),
                               ("delete", f"/v1/agents/{a['id']}", None), ("put", "/v1/tools/t", {"spec": "hangUp"}),
                               ("post", "/v1/calls", {}), ("post", "/v1/corpora", {"name": "k"})):
        r = getattr(c, method)(path, headers=hv, **({"json": body} if body is not None else {}))
        assert r.status_code == 403 and "read-only" in r.text, (method, path, r.status_code)
    # developer: changes agents and tools, but not keys, audit, webhook, or call deletion
    assert c.post("/v1/agents", json={"name": "x"}, headers=hd).status_code == 200
    assert c.put("/v1/tools/t", json={"spec": "hangUp"}, headers=hd).status_code == 200
    call = store.create_call(tid, None, "webSocket", {})
    store.update_call(call["id"], status="ended")
    assert c.post("/v1/calls", json={}, headers=hd).status_code == 200
    for method, path, body in (("get", "/v1/keys", None), ("post", "/v1/keys", {}), ("get", "/v1/audit", None),
                               ("put", "/v1/webhook", {"url": "https://x.example/h"}), ("delete", f"/v1/calls/{call['id']}", None)):
        r = getattr(c, method)(path, headers=hd, **({"json": body} if body is not None else {}))
        assert r.status_code == 403 and "admin key is required" in r.text, (method, path, r.status_code)
    assert c.get("/v1/keys", headers=ha).json()[1]["role"] == "developer"
    # a viewer can still search a knowledge base (read-only operation)
    c.post("/v1/corpora", json={"name": "k"}, headers=ha)
    assert c.post("/v1/corpora/k/search", json={"query": "x"}, headers=hv).status_code == 200
    # rotation keeps the role; the last ADMIN key cannot be revoked even if other keys exist
    rot = c.post(f"/v1/keys/{dev['id']}/rotate", headers=ha).json()
    assert c.get("/v1/agents", headers=H(rot["apiKey"])).status_code == 200
    assert c.post("/v1/agents", json={"name": "z"}, headers=H(rot["apiKey"])).status_code == 200
    assert c.post("/v1/keys", json={"role": "viewer"}, headers=H(rot["apiKey"])).status_code == 403
    admin_id = c.get("/v1/keys", headers=ha).json()[0]["id"]
    assert c.delete(f"/v1/keys/{admin_id}", headers=ha).status_code == 409
    assert c.delete(f"/v1/keys/{view['id']}", headers=ha).status_code == 200            # revoking a viewer is fine


def test_keys_from_before_roles_are_admins(tmp_path):
    import sqlite3
    from api import security
    from api.store import Store
    path = str(tmp_path / "o.db")
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE tenants(id TEXT PRIMARY KEY, name TEXT, webhook_url TEXT, webhook_secret TEXT, concurrency_limit INTEGER, created REAL);"
                     "CREATE TABLE api_keys(key_hash TEXT PRIMARY KEY, tenant_id TEXT, label TEXT, created REAL, id TEXT, revoked REAL);")
    db.execute("INSERT INTO tenants VALUES('t','n',NULL,NULL,NULL,1)")
    db.execute("INSERT INTO api_keys VALUES(?, 't', '', 1, 'key_1', NULL)", (security.hash_key("vox_k"),))
    db.commit()
    db.close()
    assert Store(path).tenant_by_key(security.hash_key("vox_k"))["role"] == "admin"


# ------------------------------------------------------------------ retention
def test_retention_purges_old_ended_calls_files_and_leaves_the_rest(tmp_path):
    from api import retention
    c, store, _ = make()
    ka, ta = tenant(c, "a")
    kb, tb = tenant(c, "b")
    now = time.time()
    day = 86400

    def mk(tid, age_days, status="ended"):
        call = store.create_call(tid, None, "webSocket", {})
        store.update_call(call["id"], status=status, joined=now - age_days * day - 60, ended=now - age_days * day)
        store.add_message(call["id"], "user", "hi", 1.0)
        (tmp_path / f"{call['id']}.wav").write_bytes(b"RIFF")
        (tmp_path / "utterances").mkdir(exist_ok=True)
        (tmp_path / "utterances" / f"{call['id']}-001.wav").write_bytes(b"x")
        return call["id"]
    old_a, new_a, live_old_a, old_b = mk(ta, 40), mk(ta, 5), mk(ta, 40, "active"), mk(tb, 40)
    store.update_tenant(ta, retention_days=30)                       # tenant a: 30 days; tenant b: no limit
    out = retention.purge_expired(store, str(tmp_path))
    assert out == {ta: 1}
    assert store.get_call_any(old_a) is None and store.messages(old_a) == []
    assert not (tmp_path / f"{old_a}.wav").exists() and not (tmp_path / "utterances" / f"{old_a}-001.wav").exists()
    for kept in (new_a, live_old_a, old_b):                          # recent, still-live and no-retention-tenant calls survive
        assert store.get_call_any(kept) is not None and (tmp_path / f"{kept}.wav").exists()
    assert store.list_audit(ta)[0]["action"] == "retention.purge"
    assert retention.purge_expired(store, str(tmp_path), default_days=10) == {tb: 1}      # b has no limit of its own: the default applies; a keeps its 30
    assert store.get_call_any(old_b) is None and store.get_call_any(new_a) is not None      # server default (10 d) now applies to b
    assert store.get_call_any(live_old_a) is not None


def test_retention_via_admin_api(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX_RECORD_DIR", str(tmp_path))
    c, store, _ = make()
    k, tid = tenant(c)
    r = c.patch(f"/admin/tenants/{tid}", json={"retentionDays": 7}, headers={"X-Admin-Key": "adm"})
    assert r.json()["retentionDays"] == 7 and c.get("/v1/me", headers=H(k)).json()["retentionDays"] == 7
    call = store.create_call(tid, None, "webSocket", {})
    store.update_call(call["id"], status="ended", ended=time.time() - 10 * 86400)
    assert c.post("/admin/purge").status_code == 403
    assert c.post("/admin/purge", headers={"X-Admin-Key": "adm"}).json() == {"erased": {tid: 1}}
    assert store.get_call_any(call["id"]) is None
    assert c.patch(f"/admin/tenants/{tid}", json={"retentionDays": 0}, headers={"X-Admin-Key": "adm"}).json()["retentionDays"] is None


# ------------------------------------------------------------------ embedding / hybrid knowledge-base search
class ConceptEmbedder:
    """Toy embedder: words of the same concept share a dimension, standing in for a real model's semantics."""
    CONCEPTS = [{"fee", "price", "priced", "cost", "dollars", "charge"}, {"doctor", "dentist", "physician", "appointment", "appointments"},
                {"open", "hours", "monday", "friday", "weekdays"}]

    def __init__(self):
        self.calls = 0
        self.min_sim = 0.0
        self.fail = False

    async def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding server down")
        return [[float(sum(w.strip(".,?") in c for w in t.lower().split())) for c in self.CONCEPTS] for t in texts]


KB = [{"id": "p", "title": "Pricing", "text": "A dentist appointment is priced at 80 dollars."},
      {"id": "h", "title": "Hours", "text": "We are open Monday to Friday."}]


async def test_hybrid_search_finds_meaning_that_keywords_miss_and_falls_back():
    from core.embed import hybrid_search
    bm = BM25(KB)
    q = "what is the fee for seeing the doctor"
    assert bm.search(q, 1) == []                                              # no shared words: keywords alone find nothing
    emb = ConceptEmbedder()
    hits = await hybrid_search(bm, emb, q, 1)
    assert hits[0]["title"] == "Pricing"
    hits = await hybrid_search(bm, emb, "when are you open on weekdays", 2)
    assert hits[0]["title"] == "Hours"
    emb.fail = True
    ctx = CallContext(docs=lambda n: ("1", KB), embedder=emb)
    _, fns = toolmod.build([{"builtin": "queryCorpus", "corpus": "kb", "topK": 1}], ctx)
    out = await fns["queryCorpus"]({"query": "what are your opening hours on Monday"})
    assert out["mode"] == "keyword" and out["results"][0]["title"] == "Hours"       # embedder down: keyword search still answers
    emb.fail = False
    out = await fns["queryCorpus"]({"query": q})
    assert out["mode"] == "hybrid" and out["results"][0]["title"] == "Pricing"


async def test_openai_embedder_batches_and_caches():
    from core.embed import OpenAIEmbedder
    seen = []

    class C:
        async def post(self, url, **kw):
            seen.append(kw["json"]["input"])
            inputs = kw["json"]["input"]
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in reversed(list(enumerate(inputs)))]},
                                  request=httpx.Request("POST", url))
    e = OpenAIEmbedder("http://x/v1/", "m", C(), batch=2)
    v = await e.embed(["a", "bb", "ccc", "a"])
    assert v == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.0, 1.0]]                  # order restored, duplicates share one request slot
    assert seen == [["a", "bb"], ["ccc"]]
    await e.embed(["bb", "ccc"])
    assert len(seen) == 2                                                          # served from the cache


def test_corpus_search_endpoint_uses_embeddings_when_configured():
    c, store, _ = make()
    k, tid = tenant(c)
    c.post("/v1/corpora", json={"name": "kb"}, headers=H(k))
    for d in KB:
        c.post("/v1/corpora/kb/documents", json={"title": d["title"], "text": d["text"]}, headers=H(k))
    q = {"query": "what is the fee for seeing the doctor"}
    r = c.post("/v1/corpora/kb/search", json=q, headers=H(k)).json()
    assert r["mode"] == "keyword" and r["results"] == []
    c.app.state.runtime.embedder = ConceptEmbedder()
    r = c.post("/v1/corpora/kb/search", json=q, headers=H(k)).json()
    assert r["mode"] == "hybrid" and r["results"][0]["title"] == "Pricing"
    c.app.state.runtime.embedder.fail = True                                        # embedding server down: still answers
    r = c.post("/v1/corpora/kb/search", json={"query": "opening hours monday"}, headers=H(k)).json()
    assert r["mode"] == "keyword" and r["results"][0]["title"] == "Hours"


# ------------------------------------------------------------------ warm transfer
from api import security as _sec


def _signed(c, path, form, base="https://vox.example.com", query=""):
    url = base + path + (f"?{query}" if query else "")
    return {"X-Twilio-Signature": _sec.twilio_signature("tok", url, form)}


async def test_warm_transfer_tool_builds_dial_with_whisper_and_return_callbacks():
    posted = {}

    class Client:
        async def post(self, url, **kw):
            posted.update(url=url, data=kw["data"])
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))
    ctx = CallContext(call_id="call_1", call_sid="CA9", twilio={"sid": "ACx", "token": "t"}, base_url="https://vox.example.com")
    spec = {"builtin": "transfer", "mode": "warm", "targets": {"billing": "+15551230001"}}
    schemas, fns = toolmod.build([spec], ctx, Client())
    r = await fns["transfer"]({"to": "billing"})
    assert posted == {}                                                       # nothing happens until the goodbye is spoken
    await r.after()
    tw = posted["data"]["Twiml"]
    assert posted["url"].endswith("/Calls/CA9.json")
    assert 'action="https://vox.example.com/twilio/transfer-result/call_1?to=billing"' in tw
    assert '<Number url="https://vox.example.com/twilio/whisper/call_1">+15551230001</Number>' in tw and 'timeout="25"' in tw
    ctx2 = CallContext(call_id="call_1", call_sid="CA9", twilio={"sid": "ACx", "token": "t"})   # no public URL known
    _, f2 = toolmod.build([spec], ctx2, Client())
    assert "public URL" in (await f2["transfer"]({"to": "billing"}))["error"]
    _, f3 = toolmod.build([{**spec, "mode": "cold"}], ctx, Client())
    r3 = await f3["transfer"]({"to": "billing"})
    posted.clear()
    await r3.after()
    assert posted["data"]["Twiml"] == "<Response><Dial>+15551230001</Dial></Response>"           # cold transfer unchanged
    with pytest.raises(toolmod.ToolConfigError):
        toolmod.validate_specs([{**spec, "mode": "tepid"}])


def test_warm_transfer_whisper_summarises_the_caller_safely_and_needs_a_signature():
    c, store, _ = make()
    k, tid = tenant(c)
    call = store.create_call(tid, None, "twilio", {})
    for text in ("first thing", "I want to cancel <Redirect>http://evil</Redirect> my plan", "[inaudible]"):
        store.add_message(call["id"], "user", text, 1.0)
    store.add_message(call["id"], "agent", "Sure, one moment.", 2.0)
    path = f"/twilio/whisper/{call['id']}"
    assert c.post(path, data={"CallSid": "CA1"}).status_code == 403
    r = c.post(path, data={"CallSid": "CA1"}, headers=_signed(c, path, {"CallSid": "CA1"}))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/xml")
    assert "<Say>" in r.text and "cancel" in r.text and "first thing" in r.text and "Sure, one moment" not in r.text
    assert "<Redirect>" not in r.text and "evil" in r.text                      # markup stripped; plain words only
    empty = store.create_call(tid, None, "twilio", {})
    p2 = f"/twilio/whisper/{empty['id']}"
    assert "did not leave details" in c.post(p2, data={"A": "b"}, headers=_signed(c, p2, {"A": "b"})).text


def test_warm_transfer_result_hangs_up_when_answered_and_returns_the_caller_to_the_agent_otherwise():
    c, store, _ = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!"}, headers=H(k)).json()
    call = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666"})
    path, q = f"/twilio/transfer-result/{call['id']}", "to=billing"
    done = {"DialCallStatus": "completed"}
    assert c.post(f"{path}?{q}", data=done).status_code == 403
    r = c.post(f"{path}?{q}", data=done, headers=_signed(c, path, done, query=q))
    assert "<Hangup/>" in r.text and "<Stream" not in r.text
    miss = {"DialCallStatus": "no-answer"}
    r = c.post(f"{path}?{q}", data=miss, headers=_signed(c, path, miss, query=q))
    assert "<Connect>" in r.text and "wss://vox.example.com/ws/twilio/" in r.text
    resumed = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == call["id"]]
    assert len(resumed) == 1 and resumed[0]["medium"] == "twilio"
    cfg = resumed[0]["config"]
    assert "billing is not available" in cfg["greeting"] and "Be brief." in cfg["systemPrompt"] and "billing" in cfg["systemPrompt"]
    evil = {"DialCallStatus": "busy"}
    q2 = "to=" + "x%3Cscript%3E"
    r = c.post(f"{path}?{q2}", data=evil, headers=_signed(c, path, evil, query=q2))
    assert "<script>" not in r.text and "<script>" not in store.list_calls(tid)[0]["config"]["greeting"]
