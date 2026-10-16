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
