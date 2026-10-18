import asyncio
import json
import os
import threading
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
    assert c2.app.state.limiter.local_active() == 0                       # a rejected dial must not keep the slot
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
    assert c.app.state.limiter.local_active() == 0                        # never connected: the reservation is gone
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
    assert r.json()["action"] == "voicemail" and r.json()["delivered"] is True
    time.sleep(0.3)
    assert spoke == ["hi", "END"]


def test_outbound_holds_a_slot_while_dialing_and_releases_if_twilio_never_connects():
    from api import security
    box = {}

    def handler(req):
        box["n"] = box.get("n", 0) + 1
        box["during"] = box["lim"].local_active()
        if box["n"] == 1:
            return httpx.Response(201, json={"sid": "CA1"})
        return httpx.Response(400, json={"message": "nope"})

    c, store, _ = make_outbound(handler, node_max_calls=1)
    box["lim"] = c.app.state.limiter
    k, tid = tenant(c, limit=10)
    h = H(k)
    a = c.post("/v1/agents", json={"name": "a"}, headers=h).json()
    first = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567"}, headers=h)
    assert first.status_code == 200 and box["during"] == 1 and box["n"] == 1
    assert c.get("/healthz").status_code == 503 and c.get("/healthz").json()["nodeFull"] is True
    before = len(store.list_calls(tid))
    denied = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234568"}, headers=h)
    assert denied.status_code == 503 and box["n"] == 1 and len(store.list_calls(tid)) == before
    url = f"https://vox.example.com/twilio/status/{first.json()['id']}"
    form = {"CallStatus": "no-answer"}
    assert c.post(f"/twilio/status/{first.json()['id']}", data=form,
                  headers={"X-Twilio-Signature": security.twilio_signature("tok", url, form)}).status_code == 200
    assert c.app.state.limiter.local_active() == 0 and c.get("/healthz").status_code == 200
    bad = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234569"}, headers=h)
    assert bad.status_code == 502 and "nope" in bad.text
    assert c.app.state.limiter.local_active() == 0 and c.get("/healthz").status_code == 200
    refused = [x for x in store.list_calls(tid) if x["end_reason"] == "twilio_rejected"]
    assert len(refused) == 1


def test_outbound_failure_does_not_drop_a_leg_the_socket_already_adopted():
    """A dial that fails after /ws/twilio has admitted must leave that leg active and keep the slot."""
    import threading

    def scenario(fail):
        inside, release = threading.Event(), threading.Event()

        async def handler(req):
            inside.set()
            for _ in range(500):
                if release.is_set():
                    break
                await asyncio.sleep(0.01)
            return fail()

        c, store, _ = make_outbound(handler, node_max_calls=1)
        box: dict = {}

        def post(agent_id, key):
            try:
                box["r"] = c.post("/v1/calls/outbound", json={"agentId": agent_id, "to": "+15551234567"}, headers=H(key))
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        with c:
            k, tid = tenant(c, limit=10)
            a = c.post("/v1/agents", json={"name": "a", "greeting": ""}, headers=H(k)).json()
            th = threading.Thread(target=post, args=(a["id"], k))
            th.start()
            try:
                assert inside.wait(5)
                call = store.list_calls(tid)[0]
                with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
                    ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
                        "streamSid": "MZ1", "callSid": "CA1",
                        "customParameters": {"token": call["join_token"]}}}))
                    adopted = False
                    for _ in range(50):
                        if store.get_call_any(call["id"])["status"] == "active":
                            adopted = True
                            break
                        time.sleep(0.02)
                    assert adopted
                    release.set()
                    th.join(5)
                    assert not th.is_alive() and "err" not in box
                    body = box["r"].json()
                    assert box["r"].status_code == 200 and body["id"] == call["id"] and body["status"] == "active"
                    row = store.get_call_any(call["id"])
                    assert row["status"] == "active" and not row["end_reason"]
                    assert c.app.state.limiter.local_active() == 1
                    denied = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234568"}, headers=H(k))
                    assert denied.status_code == 503 and "node full" in denied.text
                    assert c.app.state.limiter.local_active() == 1 and len(store.list_calls(tid)) == 1
            finally:
                release.set()
                th.join(5)

    def boom():
        raise RuntimeError("twilio dropped")

    def rejected():
        return httpx.Response(400, json={"message": "nope"})

    def html():
        return httpx.Response(201, content=b"<html>queued</html>", headers={"content-type": "text/html"})

    scenario(boom)
    scenario(rejected)
    scenario(html)


def test_outbound_non_json_success_releases_the_slot_and_ends_the_row():
    """A 201 whose body is not JSON must not keep the reservation or leave /healthz failed."""
    def html(req):
        return httpx.Response(201, content=b"<html>queued</html>", headers={"content-type": "text/html"})

    c, store, _ = make_outbound(html, node_max_calls=1)
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567"}, headers=H(k))
    assert r.status_code == 502
    assert c.app.state.limiter.local_active() == 0 and c.get("/healthz").status_code == 200
    row = store.list_calls(tid)[0]
    assert row["status"] == "ended" and row["end_reason"] == "twilio_rejected"

    def ok(req):
        return httpx.Response(201, json={"sid": "CA1"})

    c2, store2, _ = make_outbound(ok, node_max_calls=1)
    k2, tid2 = tenant(c2, limit=10)
    a2 = c2.post("/v1/agents", json={"name": "a"}, headers=H(k2)).json()
    orig = store2.update_call

    def boom(call_id, **fields):
        raise RuntimeError("disk full")

    store2.update_call = boom
    r2 = c2.post("/v1/calls/outbound", json={"agentId": a2["id"], "to": "+15551234567"}, headers=H(k2))
    assert r2.status_code == 502
    # The ended write throws too. The slot must already be gone; the row stays created.
    assert c2.app.state.limiter.local_active() == 0 and c2.get("/healthz").status_code == 200
    row2 = store2.list_calls(tid2)[0]
    assert row2["status"] == "created" and not row2["end_reason"]
    store2.update_call = orig


def test_count_outbound_is_not_limited_by_list_calls_page_size():
    from api.store import Store
    store = Store()
    tid = store.create_tenant("acme")
    since = time.time() - 60
    for i in range(5):
        store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "n": i})
    store.create_call(tid, None, "webSocket", {}, {"from": "+15550001"})  # not outbound
    old = store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "n": "old"})
    store._x("UPDATE calls SET created=? WHERE id=?", (since - 86400, old["id"]))
    assert len(store.list_calls(tid, limit=1)) == 1
    assert store.count_outbound(tid, since) == 5
    # the route must not go back to scanning a short page

    def ok(req):
        return httpx.Response(201, json={"sid": "CA9"})
    c, store2, _ = make_outbound(ok, outbound_per_day=5)
    k, tid2 = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    for i in range(5):
        store2.create_call(tid2, a["id"], "twilio", {}, {"direction": "outbound", "n": i})
    assert len(store2.list_calls(tid2, limit=1)) == 1
    assert store2.count_outbound(tid2, time.time() - 86400) == 5
    store2.list_calls = lambda tenant_id, limit=50: []  # a truncated scan would under-count and allow the call
    r = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551230000"}, headers=H(k))
    assert r.status_code == 429


class _OutboundRedis:
    """Lock commands, plus the old sorted-set script.

    That script still admits when the set is empty. The cap is the SQL count; a regression that trusts
    this set fails the tests below. ``eval`` follows the release script it is given: compare-and-delete
    only when the script reads the token, so an unconditional DEL is observable.
    """

    def __init__(self):
        self._mu = threading.Lock()
        self.z, self.expire, self.scripts = {}, {}, []
        self.kv = {}          # key -> (value, expiry or None)
        self.sets = []

    def register_script(self, script):
        self.scripts.append(script)

        def call(keys=None, args=None, client=None):
            key = keys[0]
            # Current node-admission script: now, expiry, member, cap.
            # The older daily-cap script passed cap as args[2], the member as args[3], and a cutoff in args[4].
            # Either form still admits into an empty set, so a regression that trusts this set is visible.
            if len(args) >= 5:
                now, window, cap, member = float(args[0]), float(args[1]), int(args[2]), args[3]
                cutoff = float(str(args[4])[1:])
                score = now
            else:
                cutoff, score, member, cap = float(args[0]), float(args[1]), args[2], int(args[3])
                window = score
            with self._mu:
                bucket = self.z.setdefault(key, {})
                for m, old in list(bucket.items()):
                    if old < cutoff:
                        del bucket[m]
                self.expire[key] = window
                if len(args) < 5 and member in bucket:
                    bucket[member] = score
                    return 1
                if len(bucket) >= cap:
                    return 0
                bucket[member] = score
                return 1

        return call

    def _purge_unlocked(self) -> None:
        now = time.time()
        for k, (_val, exp) in list(self.kv.items()):
            if exp is not None and exp <= now:
                del self.kv[k]

    def set(self, name, value, ex=None, nx=False, **kwargs):
        with self._mu:
            self._purge_unlocked()
            ex_n = None if ex is None else float(ex)
            self.sets.append({"name": name, "ex": ex_n, "nx": bool(nx)})
            if nx and name in self.kv:
                return None
            exp = None if ex_n is None else time.time() + ex_n
            self.kv[name] = (value, exp)
            return True

    def get(self, name):
        with self._mu:
            self._purge_unlocked()
            item = self.kv.get(name)
            return None if item is None else item[0]

    def eval(self, script, numkeys, *keys_and_args):
        key = keys_and_args[0]
        token = keys_and_args[1] if len(keys_and_args) > 1 else None
        with self._mu:
            self._purge_unlocked()
            if "get" in script and "ARGV[1]" in script:
                cur = self.kv.get(key)
                if cur is None or cur[0] != token:
                    return 0
                if "pexpire" in script or "expire" in script:
                    raw = float(keys_and_args[2])
                    ttl = raw / 1000.0 if "pexpire" in script else raw
                    self.kv[key] = (cur[0], None if ttl <= 0 else time.time() + ttl)
                    return 1
                if "del" in script:
                    del self.kv[key]
                    return 1
                return 0
            if "del" in script:
                self.kv.pop(key, None)
                return 1
            raise AssertionError(script)


def test_fresh_redis_does_not_ignore_stored_outbound_calls():
    """An empty Redis must not admit another full cap over rows SQL already has, or keep a member."""
    from api import security
    from api.store import Store

    store = Store()
    tid = store.create_tenant("acme")
    per_day = 2
    for i in range(per_day):
        store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "n": i})
    store.create_call(tid, None, "webSocket", {}, {"from": "+15550001"})  # not outbound, must not count
    fake = _OutboundRedis()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    assert cap.allow(tid, per_day) is False
    assert cap.allow(tid, per_day + 1) is True
    assert fake.z == {}                                            # no member added on the empty key
    before = store.count_outbound(tid, time.time() - 86400)
    with cap.hold(tid, per_day) as allowed:
        assert allowed is False
        if allowed:                                                # a fall-open would consume a slot
            store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert store.count_outbound(tid, time.time() - 86400) == before == per_day
    assert fake.z == {} and fake.kv == {}
    assert fake.sets and all(s["name"] == f"vox:outbound-lock:{tid}" for s in fake.sets)
    assert all(s["ex"] is not None and s["ex"] <= 30 for s in fake.sets)
    other = store.create_tenant("other")
    assert cap.allow(other, per_day) is True                       # full tenant does not block another


def test_redis_outbound_cap_does_not_reset_at_utc_midnight():
    """The window is the SQL rolling 24h. A new UTC date, or an empty Redis, must not open another cap."""
    from datetime import datetime, timezone
    from unittest.mock import patch
    from api import security
    from api.store import Store

    t0 = datetime(2026, 10, 17, 23, 50, tzinfo=timezone.utc).timestamp()
    store = Store()
    tid = store.create_tenant("acme")
    current = store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    edge = store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "edge": True})
    older = store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "old": True})
    store._x("UPDATE calls SET created=? WHERE id=?", (t0, current["id"]))
    store._x("UPDATE calls SET created=? WHERE id=?", (t0 - 86400, edge["id"]))             # exactly 24h before t0
    store._x("UPDATE calls SET created=? WHERE id=?", (t0 - 86400 - 1, older["id"]))
    fake = _OutboundRedis()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    now = {"t": t0}
    midnight = datetime(2026, 10, 18, 0, 10, tzinfo=timezone.utc).timestamp()
    with patch("api.security.time.time", lambda: now["t"]):
        assert cap.allow(tid, 2) is False                          # edge still counts; the older row does not
        assert cap.allow(tid, 3) is True
        now["t"] = midnight
        assert cap.allow(tid, 1) is False                          # past UTC midnight, the t0 row is still inside 24h
        now["t"] = t0 + 86400
        assert cap.allow(tid, 1) is False                          # a call placed exactly 24h ago still counts
        now["t"] = t0 + 86400 + 1
        assert cap.allow(tid, 1) is True
    assert fake.z == {}                                            # Redis never became a second counter


def test_two_nodes_share_sql_outbound_cap_when_redis_is_empty(tmp_path):
    """Two stores, one Redis. Preloaded rows plus an empty key must still stop at the cap."""
    from api import security
    from api.store import Store

    path = str(tmp_path / "calls.db")
    store_a, store_b = Store(path), Store(path)
    for s in (store_a, store_b):
        s.db.execute("PRAGMA busy_timeout=5000")
    tid = store_a.create_tenant("acme")
    for i in range(4):
        store_a.create_call(tid, None, "twilio", {}, {"direction": "outbound", "n": i})
    fake = _OutboundRedis()
    caps = (
        (security.OutboundCap(security.RedisLimiter(client=fake), store_a), store_a),
        (security.OutboundCap(security.RedisLimiter(client=fake), store_b), store_b),
    )
    n, limit = 20, 5
    bar = threading.Barrier(n)
    wins, errors = [], []

    def worker(i):
        cap, store = caps[i % 2]
        try:
            bar.wait(timeout=5)
            with cap.hold(tid, limit) as allowed:
                if not allowed:
                    return
                time.sleep(0.02)                                   # widen the gap a missing mutex would lose
                store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
                wins.append(i)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors and all(not t.is_alive() for t in threads)
    since = time.time() - 86400
    assert len(wins) == 1
    assert store_a.count_outbound(tid, since) == limit
    assert store_b.count_outbound(tid, since) == limit
    assert fake.z == {} and fake.kv == {}                          # no leftover member and no stuck lock
    store_a.close()
    store_b.close()


def test_redis_outage_fails_outbound_attempt_closed():
    from api import security
    from api.store import Store

    class Down(_OutboundRedis):
        def set(self, *a, **k):
            raise ConnectionError("redis down")

    store = Store()
    tid = store.create_tenant("acme")
    cap = security.OutboundCap(security.RedisLimiter(client=Down()), store)
    with cap.hold(tid, 5) as allowed:
        assert allowed is False
        if allowed:
            store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert store.count_outbound(tid, time.time() - 86400) == 0


def test_expired_outbound_lock_does_not_stick_the_cap_shut():
    from api import security
    from api.store import Store

    store = Store()
    tid = store.create_tenant("acme")
    fake = _OutboundRedis()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    assert cap.LOCK_TTL_S <= 30 and cap.LOCK_WAIT_S >= cap.LOCK_TTL_S
    cap.LOCK_TTL_S = 0.25
    token = cap._acquire_lock(tid)
    assert token and fake.get(cap._lock_key(tid)) == token
    time.sleep(0.5)                                                # dead holder: TTL, not an explicit release
    assert fake.get(cap._lock_key(tid)) is None
    started = time.monotonic()
    with cap.hold(tid, 1) as allowed:
        assert allowed                                   # lease object: truth is ownership, not `is True`
        store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert time.monotonic() - started < 1.0
    assert store.count_outbound(tid, time.time() - 86400) == 1
    fresh = cap._acquire_lock(tid)                                 # late release of the dead token must not drop this
    cap._release_lock(tid, token)
    assert fake.get(cap._lock_key(tid)) == fresh
    cap._release_lock(tid, fresh)
    assert fake.get(cap._lock_key(tid)) is None


def test_inprocess_outbound_hold_admits_exactly_the_cap():
    """Without Redis, store.lock has to cover the count and the insert. Cap 5 admits 5, not 6."""
    from api import security
    from api.store import Store

    store = Store()
    tid = store.create_tenant("acme")
    cap = security.OutboundCap(security.ConcurrencyLimiter(10), store)
    n, limit = 30, 5
    bar = threading.Barrier(n)
    wins, errors = [], []

    def worker():
        try:
            bar.wait(timeout=5)
            with cap.hold(tid, limit) as allowed:
                if not allowed:
                    return
                time.sleep(0.01)
                store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
                wins.append(1)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors and all(not t.is_alive() for t in threads)
    assert len(wins) == limit
    assert store.count_outbound(tid, time.time() - 86400) == limit


@pytest.mark.parametrize("how", ["delete", "expire"])
def test_outbound_lock_lost_before_insert_admits_one_row(how):
    """Cap 1 stays one row when the key is deleted or the TTL fires between allow() and create_call.

    The holder already inside hold must not insert. The next holder, who takes the new lease, may.
    """
    from api import security
    from api.store import Store

    store = Store()
    tid = store.create_tenant("acme")
    fake = _OutboundRedis()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    cap.LOCK_TTL_S = 30
    cap.LOCK_WAIT_S = 3
    cap._refresh_interval = lambda: 60.0          # parked refresher must not put the lost key back
    key = cap._lock_key(tid)
    inside, opened = threading.Event(), threading.Event()
    errors, inserted = [], []

    def holder():
        try:
            with cap.hold(tid, 1) as allowed:
                assert allowed
                inside.set()
                assert opened.wait(timeout=5)
                if allowed:                        # re-read after the lease is gone; a stale True inserts
                    store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "who": "holder"})
                    inserted.append("holder")
        except Exception as e:  # noqa: BLE001
            errors.append(("holder", repr(e)))

    def other():
        try:
            assert opened.wait(timeout=5)
            with cap.hold(tid, 1) as allowed:
                if allowed:
                    store.create_call(tid, None, "twilio", {}, {"direction": "outbound", "who": "other"})
                    inserted.append("other")
        except Exception as e:  # noqa: BLE001
            errors.append(("other", repr(e)))

    first, second = threading.Thread(target=holder), threading.Thread(target=other)
    first.start()
    assert inside.wait(timeout=5)
    with fake._mu:
        cur = fake.kv.get(key)
        assert cur is not None
        if how == "delete":
            fake.kv.pop(key, None)                 # Redis restart: --save "" and no volume
        else:
            fake.kv[key] = (cur[0], time.time() - 1)  # TTL fired under the live holder
    second.start()
    opened.set()
    first.join(timeout=8)
    second.join(timeout=8)
    assert not errors, errors
    assert not first.is_alive() and not second.is_alive()
    assert inserted == ["other"], (how, inserted)
    assert store.count_outbound(tid, time.time() - 86400) == 1


def test_outbound_hold_refreshes_its_lease():
    """A live holder extends the mutex well inside LOCK_TTL_S and does not renew someone else's token."""
    from api import security
    from api.store import Store

    store = Store()
    tid = store.create_tenant("acme")
    fake = _OutboundRedis()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    assert cap._refresh_interval() * 2 < cap.LOCK_TTL_S
    cap.LOCK_TTL_S = 8
    assert cap._refresh_interval() * 2 < cap.LOCK_TTL_S
    cap._refresh_interval = lambda: 0.05
    key = cap._lock_key(tid)
    with cap.hold(tid, 1) as allowed:
        assert allowed
        token = fake.get(key)
        assert token
        with fake._mu:
            fake.kv[key] = (token, time.time() + 1000)  # sentinel a refresh has to overwrite
        deadline = time.monotonic() + 2
        renewed = False
        while time.monotonic() < deadline:
            with fake._mu:
                item = fake.kv.get(key)
            if item and item[0] == token and item[1] is not None and item[1] < time.time() + 30:
                renewed = True
                break
            time.sleep(0.01)
        assert renewed
        assert allowed
        store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert store.count_outbound(tid, time.time() - 86400) == 1
    token = cap._acquire_lock(tid)
    assert token
    soon = time.time() + 30
    with fake._mu:
        fake.kv[key] = (token, soon)
    assert cap._refresh_lock(tid, "other-token") is False
    with fake._mu:
        assert fake.kv.get(key) == (token, soon)
    assert cap._refresh_lock(tid, token) is True
    with fake._mu:
        item = fake.kv.get(key)
    assert item is not None and item[0] == token and item[1] > time.time() + 4
    with fake._mu:
        fake.kv.pop(key, None)
    assert cap._refresh_lock(tid, token) is False   # a deleted key is not recreated
    assert fake.get(key) is None
    cap._release_lock(tid, token)
    assert fake.get(key) is None


def test_outbound_get_error_after_lock_does_not_insert():
    """SET took the mutex, then GET raises. The lease is falsy, so cap 1 does not insert; a later GET still may."""
    from api import security
    from api.store import Store

    class GetDown(_OutboundRedis):
        def __init__(self):
            super().__init__()
            self.fail_get = True

        def get(self, name):
            if self.fail_get:
                raise ConnectionError("get failed")
            return super().get(name)

    store = Store()
    tid = store.create_tenant("acme")
    fake = GetDown()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    cap.LOCK_WAIT_S = 2
    key = cap._lock_key(tid)
    with cap.hold(tid, 1) as allowed:
        with fake._mu:
            assert key in fake.kv                         # SET succeeded; only the ownership GET fails
        assert bool(allowed) is False                    # not `is False`: the yield is a lease, not a bool
        if allowed:
            store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert store.count_outbound(tid, time.time() - 86400) == 0
    fake.fail_get = False
    with cap.hold(tid, 1) as allowed:
        assert bool(allowed) is True
        if allowed:
            store.create_call(tid, None, "twilio", {}, {"direction": "outbound"})
    assert store.count_outbound(tid, time.time() - 86400) == 1


def test_outbound_refresh_retries_after_eval_error():
    """A refresh EVAL blip is retried. The next success replaces the sentinel. A blip does not recreate a missing key."""
    from api import security
    from api.store import Store

    class FlakyEval(_OutboundRedis):
        def __init__(self):
            super().__init__()
            self.refresh_raises_left = 1
            self.refresh_attempts = 0
            self.refresh_errors = 0

        def eval(self, script, numkeys, *keys_and_args):
            if "pexpire" in script:
                self.refresh_attempts += 1
                if self.refresh_raises_left > 0:
                    self.refresh_raises_left -= 1
                    self.refresh_errors += 1
                    raise ConnectionError("eval blip")
            return super().eval(script, numkeys, *keys_and_args)

    store = Store()
    tid = store.create_tenant("acme")
    fake = FlakyEval()
    cap = security.OutboundCap(security.RedisLimiter(client=fake), store)
    cap.LOCK_TTL_S = 8
    cap._refresh_interval = lambda: 0.05
    key = cap._lock_key(tid)
    with cap.hold(tid, 1) as allowed:
        assert bool(allowed) is True
        token = fake.get(key)
        assert token
        with fake._mu:
            fake.kv[key] = (token, time.time() + 1000)   # sentinel the retry has to overwrite
        deadline = time.monotonic() + 2
        renewed = False
        while time.monotonic() < deadline:
            with fake._mu:
                item = fake.kv.get(key)
            if item and item[0] == token and item[1] is not None:
                remaining = item[1] - time.time()
                if 1 < remaining <= cap.LOCK_TTL_S + 0.25:
                    renewed = True
                    break
            time.sleep(0.01)
        assert renewed
        assert fake.refresh_errors >= 1
        assert fake.refresh_attempts >= fake.refresh_errors + 1
    with fake._mu:
        fake.kv.pop(key, None)
    fake.refresh_raises_left = 1
    assert cap._refresh_lock(tid, token) is True         # blip retries; it must not SET the key back
    assert fake.get(key) is None


def test_outbound_route_does_not_dial_when_the_lock_disappears_before_insert():
    """The handler must skip create_call and Twilio when the lease vanishes after the count."""
    from api import security
    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from starlette.testclient import TestClient
    from test_api import FakeLLM as L, FakeTTS as T, FakeVAD

    posts = []

    def handler(req):
        posts.append(str(req.url))
        return httpx.Response(201, json={"sid": "CA1"})

    store = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(L(), lambda v: T(), FakeVAD, store, http=http)
    fake = _OutboundRedis()
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", twilio_number="+15550001111", outbound_per_day=1)
    c = TestClient(create_app(store, rt, security.RedisLimiter(client=fake), None, settings))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    entered, release = threading.Event(), threading.Event()
    orig = store.count_outbound
    state = {"block": True}

    def wrapped(tenant_id, since):
        if state["block"]:
            state["block"] = False
            entered.set()
            assert release.wait(timeout=5)
            with fake._mu:
                fake.kv.clear()                    # gone after allow()'s count, before create_call
        return orig(tenant_id, since)

    store.count_outbound = wrapped
    box = {}

    def go():
        box["r"] = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551230000"}, headers=H(k))

    t = threading.Thread(target=go)
    t.start()
    assert entered.wait(timeout=5)
    release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert box["r"].status_code == 429 and "outbound limit" in box["r"].text
    assert posts == [] and orig(tid, time.time() - 86400) == 0
    r2 = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551230001"}, headers=H(k))
    assert r2.status_code == 200 and r2.json()["callSid"] == "CA1"
    assert len(posts) == 1 and orig(tid, time.time() - 86400) == 1


def test_drain_during_outbound_admit_places_nothing():
    """Drain that lands after the handler's first check and before insert is 503, with no row and no dial."""
    posts = []

    def handler(req):
        posts.append(str(req.url))
        return httpx.Response(201, json={"sid": "CA1"})

    c, store, _ = make_outbound(handler, outbound_per_day=5)
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    entered, release = threading.Event(), threading.Event()
    orig = store.count_outbound

    def wrapped(tenant_id, since):
        entered.set()
        assert release.wait(timeout=5)
        return orig(tenant_id, since)

    store.count_outbound = wrapped
    box = {}

    def go():
        box["r"] = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551230000"}, headers=H(k))

    t = threading.Thread(target=go)
    t.start()
    assert entered.wait(timeout=5)
    assert c.app.state.draining is False           # the check above run_in_executor already passed
    c.app.state.draining = True
    release.set()
    t.join(timeout=10)
    assert not t.is_alive(), box
    r = box["r"]
    assert r.status_code == 503 and "node is draining" in r.text
    assert posts == []
    assert orig(tid, time.time() - 86400) == 0
    assert store.list_calls(tid) == []


async def test_outbound_lock_wait_does_not_block_the_event_loop():
    """/healthz must return while /v1/calls/outbound is still polling a held cap lock.

    The poll sleeps. Running that sleep on the event loop keeps healthz from starting until the wait ends.
    """
    from api import security
    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from test_api import FakeLLM as L, FakeTTS as T, FakeVAD

    class HeldLock(_OutboundRedis):
        def __init__(self):
            super().__init__()
            self.rejected_nx = 0

        def set(self, name, value, ex=None, nx=False, **kwargs):
            result = super().set(name, value, ex=ex, nx=nx, **kwargs)
            if nx and result is None:
                self.rejected_nx += 1
            return result

        def ping(self):
            return True

    def handler(req):
        return httpx.Response(201, json={"sid": "CA1"})

    store = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(L(), lambda v: T(), FakeVAD, store, http=http)
    fake = HeldLock()
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", twilio_number="+15550001111", outbound_per_day=5)
    app = create_app(store, rt, security.RedisLimiter(client=fake), None, settings)
    old_wait = security.OutboundCap.LOCK_WAIT_S
    security.OutboundCap.LOCK_WAIT_S = 3          # inline sleep would block the loop for this long
    outbound_task = None
    key = ""
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/admin/tenants", json={"name": "acme"}, headers={"X-Admin-Key": "adm"})
            assert created.status_code == 200
            key_api, tid = created.json()["apiKey"], created.json()["tenantId"]
            agent = await client.post("/v1/agents", json={"name": "a"}, headers={"X-API-Key": key_api})
            assert agent.status_code == 200
            key = f"vox:outbound-lock:{tid}"
            with fake._mu:
                fake.kv[key] = ("other-holder", time.time() + 60)
            outbound_task = asyncio.create_task(client.post(
                "/v1/calls/outbound", json={"agentId": agent.json()["id"], "to": "+15551230000"},
                headers={"X-API-Key": key_api}))

            async def until_polling():
                while fake.rejected_nx < 1:
                    await asyncio.sleep(0.01)

            try:
                await asyncio.wait_for(until_polling(), timeout=2)
            except TimeoutError:
                raise AssertionError("outbound lock poll never yielded; the wait is on the event loop") from None
            assert not outbound_task.done()               # still inside the lock poll
            started = time.monotonic()
            health = await client.get("/healthz")
            elapsed = time.monotonic() - started
            assert health.status_code == 200 and health.json()["ok"] is True
            assert elapsed < 0.5                           # not stuck behind LOCK_WAIT_S
            assert not outbound_task.done()
            assert fake.rejected_nx >= 1
            with fake._mu:
                assert key in fake.kv                     # the other holder still owns it
                fake.kv.pop(key, None)                    # let the poll acquire and finish
            resp = await outbound_task
            outbound_task = None
            assert resp.status_code == 200 and resp.json()["callSid"] == "CA1"
    finally:
        security.OutboundCap.LOCK_WAIT_S = old_wait
        if key:
            with fake._mu:
                fake.kv.pop(key, None)
        if outbound_task is not None and not outbound_task.done():
            await outbound_task


# ------------------------------------------------------------------ draining (scale-in without dropping calls)
def test_drain_stops_new_calls_but_reports_readiness():
    from test_api import speech, silence
    c, store, lim = make()
    k, tid = tenant(c)
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
    before = len(store.list_calls(tid))
    busy = c.post(f"/twilio/voice/{a['id']}", data=form, headers={"X-Twilio-Signature": sig})
    assert busy.status_code == 200 and "Sorry, all of our lines are busy. Please call back shortly." in busy.text
    assert "<Hangup/>" in busy.text and "<Stream" not in busy.text
    assert len(store.list_calls(tid)) == before                         # draining does not create a call row
    assert c.post("/admin/undrain", headers={"X-Admin-Key": "adm"}).json() == {"draining": False}
    assert c.get("/healthz").status_code == 200


def test_node_cap_refuses_the_second_call_and_frees_when_the_first_ends():
    from test_api import silence
    from starlette.websockets import WebSocketDisconnect
    from api import security
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=1, overflow_url="https://overflow.example/busy?x=1&y=2")
    c, store, lim = make(settings)
    k, tid = tenant(c, limit=10)                 # tenant cap is wider than the node cap
    h = {"X-API-Key": k}
    assert c.get("/healthz").status_code == 200 and c.get("/healthz").json()["nodeFull"] is False
    a = c.post("/v1/agents", json={"name": "a"}, headers=h).json()
    form = {"From": "+15551230000", "CallSid": "CA1"}
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    open_line = c.post(f"/twilio/voice/{a['id']}", data=form, headers={"X-Twilio-Signature": security.twilio_signature("tok", url, form)})
    assert open_line.status_code == 200 and "<Stream" in open_line.text and "<Redirect>" not in open_line.text
    tw = store.list_calls(tid)[0]
    assert lim.local_active() == 1                                  # promised calls hold a slot before the socket
    hz = c.get("/healthz")
    assert hz.status_code == 503 and hz.json()["nodeFull"] is True and hz.json()["ok"] is False
    c1 = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=h).json()
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(c1["joinUrl"].replace("wss://vox.example.com", "")) as ws1:
            ws1.receive_text()
    assert e.value.code == 1013
    assert store.get_call_any(c1["id"])["end_reason"] == "node_full"
    assert lim.active(tid) == 1                                     # refused before the tenant slot was taken
    before = len(store.list_calls(tid))
    form2 = {"From": "+15551230001", "CallSid": "CA2"}
    busy = c.post(f"/twilio/voice/{a['id']}", data=form2,
                  headers={"X-Twilio-Signature": security.twilio_signature("tok", url, form2)})
    assert busy.status_code == 200
    assert "<Redirect>https://overflow.example/busy?x=1&amp;y=2</Redirect>" in busy.text
    assert "lines are busy" not in busy.text and len(store.list_calls(tid)) == before
    with c.websocket_connect(f"/ws/twilio/{tw['id']}") as ws:
        ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": tw["join_token"]}}}))
        # The follow-up yields so admit() has marked this reserved id active before the assert.
        ws.send_text(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "m0"}}))
        # admit() reads the tenant off the loop, so the mark returns before that reserved id is active.
        deadline = time.time() + 2
        while time.time() < deadline and store.get_call_any(tw["id"])["status"] != "active":
            time.sleep(0.02)
        assert store.get_call_any(tw["id"])["status"] == "active"   # already reserved: not a second slot
        assert lim.local_active() == 1
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    assert lim.local_active() == 0 and c.get("/healthz").status_code == 200
    c3 = c.post("/v1/calls", json={}, headers=h).json()
    with c.websocket_connect(c3["joinUrl"].replace("wss://vox.example.com", "")) as ws3:
        ws3.send_bytes(silence(1)[0])
        deadline = time.time() + 2
        while time.time() < deadline and store.get_call_any(c3["id"])["status"] != "active":
            time.sleep(0.02)
        assert store.get_call_any(c3["id"])["status"] == "active" and lim.local_active() == 1
    assert lim.local_active() == 0 and store.get_call_any(c3["id"])["end_reason"] == "hangup"


def test_overflow_url_rejects_a_non_http_scheme():
    with pytest.raises(ValueError, match="http"):
        make(Settings(admin_key="adm", overflow_url="ftp://files.example/x"))


def test_full_node_does_not_start_another_twilio_or_outbound_call():
    """A node that is already at VOX_NODE_MAX_CALLS must not mint another call the way drain refuses one."""
    import json
    from starlette.websockets import WebSocketDisconnect
    from test_api import sign, silence, twilio_form
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=1)
    c, store, lim = make(settings)
    k, tid = tenant(c, limit=10)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "a", "greeting": "hi"}, headers=h).json()
    held = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=h).json()
    with c.websocket_connect(held["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        ws.send_bytes(silence(1)[0])
        deadline = time.time() + 2
        while time.time() < deadline and lim.local_active() != 1:
            time.sleep(0.02)
        assert lim.local_active() == 1
        before = len(store.list_calls(tid))
        form = twilio_form()
        voice = f"https://vox.example.com/twilio/voice/{a['id']}"
        busy = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(voice, form))
        assert busy.status_code == 200 and "<Stream" not in busy.text and "all of our lines are busy" in busy.text
        assert len(store.list_calls(tid)) == before                     # full node: no row, no slot
        tw = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+15550001", "to": "+15550002"})
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(f"/ws/twilio/{tw['id']}") as tws:
                tws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
                    "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": tw["join_token"]}}}))
                tws.receive_text()
        assert store.get_call_any(tw["id"])["end_reason"] == "node_full"
        rurl = f"https://vox.example.com/twilio/resume/{tw['id']}"
        resumed = c.post(f"/twilio/resume/{tw['id']}", data={"CallSid": "CA1"}, headers=sign(rurl, {"CallSid": "CA1"}))
        assert resumed.status_code == 200 and "<Stream" not in resumed.text
        assert "all of our lines are busy" in resumed.text and len(store.list_calls(tid)) == before + 1
    own = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+15550001", "to": "+15550002"})
    store.update_call(own["id"], status="active", joined=time.time())
    assert lim.acquire(tid, own["id"], 10) and lim.local_active() == 1
    ourl = f"https://vox.example.com/twilio/resume/{own['id']}"
    again = c.post(f"/twilio/resume/{own['id']}", data={"CallSid": "CA1"}, headers=sign(ourl, {"CallSid": "CA1"}))
    assert "<Stream" in again.text and lim.local_active() == 1          # replacement id is already held
    new = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == own["id"]][0]
    assert lim.try_admit(tid, new["id"], 10, 1) == "ok" and lim.local_active() == 1
    hz = c.get("/healthz")
    assert hz.status_code == 503 and hz.json()["nodeFull"] is True
    other = c.post("/v1/calls", json={}, headers=h).json()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(other["joinUrl"].replace("wss://vox.example.com", "")) as ows:
            ows.receive_text()
    assert store.get_call_any(other["id"])["end_reason"] == "node_full" and lim.local_active() == 1
    with c.websocket_connect(f"/ws/twilio/{new['id']}") as tws:
        tws.send_text(json.dumps({"event": "start", "streamSid": "MZ2", "start": {
            "streamSid": "MZ2", "callSid": "CA2", "customParameters": {"token": new["join_token"]}}}))
        tws.send_text(json.dumps({"event": "mark", "streamSid": "MZ2", "mark": {"name": "m0"}}))
        deadline = time.time() + 2
        while time.time() < deadline and store.get_call_any(new["id"])["status"] != "active":
            time.sleep(0.02)
        assert store.get_call_any(new["id"])["status"] == "active" and lim.local_active() == 1
        tws.send_text(json.dumps({"event": "stop", "streamSid": "MZ2"}))
    assert lim.local_active() == 0
    # Outbound must not dial Twilio or insert a row while the node is full.
    seen = []

    def handler(req):
        seen.append(1)
        return httpx.Response(201, json={"sid": "CAOUT"})
    c2, store2, _ = make_outbound(handler, node_max_calls=1)
    k2, tid2 = tenant(c2, limit=10)
    h2 = {"X-API-Key": k2}
    a2 = c2.post("/v1/agents", json={"name": "a"}, headers=h2).json()
    b2 = c2.post("/v1/calls", json={}, headers=h2).json()
    with c2.websocket_connect(b2["joinUrl"].replace("wss://vox.example.com", "")) as ws2:
        ws2.send_bytes(silence(1)[0])
        deadline = time.time() + 2
        while time.time() < deadline and c2.app.state.limiter.local_active() != 1:
            time.sleep(0.02)
        assert c2.app.state.limiter.local_active() == 1
        before2 = len(store2.list_calls(tid2))
        denied = c2.post("/v1/calls/outbound", json={"agentId": a2["id"], "to": "+15552223333"}, headers=h2)
        assert denied.status_code == 503 and len(store2.list_calls(tid2)) == before2 and seen == []


def test_failed_transfer_does_not_create_a_call_when_the_node_is_full():
    from test_api import silence
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=1)
    c, store, lim = make(settings)
    k, tid = tenant(c, limit=10)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!"}, headers=h).json()
    ended = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666"})
    store.update_call(ended["id"], status="ended", ended=time.time(), end_reason="stream_closed")
    held = c.post("/v1/calls", json={}, headers=h).json()
    with c.websocket_connect(held["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        ws.send_bytes(silence(1)[0])
        deadline = time.time() + 2
        while time.time() < deadline and lim.local_active() != 1:
            time.sleep(0.02)
        assert lim.local_active() == 1
        path, q = f"/twilio/transfer-result/{ended['id']}", "to=billing"
        miss = {"DialCallStatus": "no-answer"}
        before = len(store.list_calls(tid))
        r = c.post(f"{path}?{q}", data=miss, headers=_signed(c, path, miss, query=q))
        assert r.status_code == 200 and "<Stream" not in r.text and "busy" in r.text
        assert len(store.list_calls(tid)) == before
    # The caller's own slot must not block the return leg: release it, then the new stream can be admitted.
    own = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666"})
    store.update_call(own["id"], status="active", joined=time.time())
    assert lim.acquire(tid, own["id"], 10)
    path, q = f"/twilio/transfer-result/{own['id']}", "to=billing"
    r = c.post(f"{path}?{q}", data=miss, headers=_signed(c, path, miss, query=q))
    assert "<Stream" in r.text and lim.local_active() == 1
    new = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == own["id"]][0]
    assert c.get("/healthz").status_code == 503 and c.get("/healthz").json()["nodeFull"] is True
    from starlette.websockets import WebSocketDisconnect
    extra = c.post("/v1/calls", json={}, headers=h).json()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(extra["joinUrl"].replace("wss://vox.example.com", "")) as ews:
            ews.receive_text()
    assert store.get_call_any(extra["id"])["end_reason"] == "node_full" and lim.local_active() == 1
    import json
    with c.websocket_connect(f"/ws/twilio/{new['id']}") as tws:
        tws.send_text(json.dumps({"event": "start", "streamSid": "MZ9", "start": {
            "streamSid": "MZ9", "callSid": "CA9", "customParameters": {"token": new["join_token"]}}}))
        tws.send_text(json.dumps({"event": "mark", "streamSid": "MZ9", "mark": {"name": "m0"}}))
        deadline = time.time() + 2
        while time.time() < deadline and store.get_call_any(new["id"])["status"] != "active":
            time.sleep(0.02)
        assert store.get_call_any(new["id"])["status"] == "active" and lim.local_active() == 1
        tws.send_text(json.dumps({"event": "stop", "streamSid": "MZ9"}))
    assert store.get_call_any(new["id"])["end_reason"] != "node_full"
    assert lim.local_active() == 0


def test_sip_call_is_refused_without_a_row_when_the_node_is_full():
    import socket
    import uuid
    from api import security
    from api.runtime import Runtime
    from api.store import Store
    from starlette.testclient import TestClient
    from test_api import FakeLLM, FakeTTS, FakeVAD
    from transports import audiosocket as A
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a", 10)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    lim = security.ConcurrencyLimiter(4)
    app = __import__("api.app", fromlist=["create_app"]).create_app(
        store, rt, lim, None,
        Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"], node_max_calls=1))
    assert lim.acquire(tid, "call_held", 10) and lim.local_active() == 1
    with TestClient(app):
        before = len(store.list_calls(tid))
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(1)
        sock.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
        try:
            sock.recv(64)
        except OSError:
            pass
        time.sleep(0.2)
        sock.close()
    assert len(store.list_calls(tid)) == before
    assert lim.local_active() == 1


def test_sip_call_holds_a_slot_until_the_uuid_arrives_and_releases_if_it_never_does():
    import socket
    from api import security
    from api.runtime import Runtime
    from api.store import Store
    from starlette.testclient import TestClient
    from test_api import FakeLLM, FakeTTS, FakeVAD
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a", 10)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    lim = security.ConcurrencyLimiter(4)
    app = __import__("api.app", fromlist=["create_app"]).create_app(
        store, rt, lim, None,
        Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"], node_max_calls=1))
    with TestClient(app) as client:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(1)
        held = False
        for _ in range(50):
            if lim.local_active() == 1:
                held = True
                break
            time.sleep(0.05)
        assert held
        hz = client.get("/healthz")
        assert hz.status_code == 503 and hz.json()["nodeFull"] is True
        assert lim.try_admit(tid, "other", 10, 1) == "node" and lim.local_active() == 1
        assert store.list_calls(tid) == []                              # no row until the UUID is in
        sock.close()
        freed = False
        for _ in range(50):
            if lim.local_active() == 0:
                freed = True
                break
            time.sleep(0.05)
        assert freed and store.list_calls(tid) == []


def _sip_app(port, store, lim, ag, node_max_calls=1):
    from api.app import create_app
    from api.runtime import Runtime
    from test_api import FakeLLM, FakeTTS, FakeVAD
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    return create_app(store, rt, lim, None,
                      Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"],
                               node_max_calls=node_max_calls))


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_stalled_sip_uuid_read_releases_the_slot(monkeypatch):
    """One byte and a peer that stays open must not hold the only slot past the UUID deadline."""
    import socket
    from api import security
    from api.store import Store
    from starlette.testclient import TestClient
    monkeypatch.setattr("api.app.AUDIOSOCKET_UUID_TIMEOUT_S", 0.4)
    port = _free_port()
    store = Store()
    tid = store.create_tenant("a", 10)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    lim = security.ConcurrencyLimiter(4)
    app = _sip_app(port, store, lim, ag)
    sock = None
    try:
        with TestClient(app) as client:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            sock.sendall(b"\x01")
            held = False
            for _ in range(40):
                if lim.local_active() == 1:
                    held = True
                    break
                time.sleep(0.02)
            assert held and client.get("/healthz").status_code == 503
            freed = False
            for _ in range(40):
                if lim.local_active() == 0:
                    freed = True
                    break
                time.sleep(0.05)
            assert freed and store.list_calls(tid) == []
            assert client.get("/healthz").status_code == 200
            # Abort drops the socket. A drain/_close path would have written the hangup frame first.
            from transports.audiosocket import KIND_HANGUP, pack
            sock.settimeout(1)
            try:
                peer = sock.recv(64)
            except OSError:
                peer = b""
            assert not peer.startswith(pack(KIND_HANGUP))
    finally:
        if sock is not None:
            sock.close()


def test_sip_shutdown_releases_a_stalled_uuid_read(monkeypatch):
    import socket
    from api import security
    from api.store import Store
    from starlette.testclient import TestClient
    monkeypatch.setattr("api.app.AUDIOSOCKET_UUID_TIMEOUT_S", 30)
    port = _free_port()
    store = Store()
    tid = store.create_tenant("a", 10)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    lim = security.ConcurrencyLimiter(4)
    app = _sip_app(port, store, lim, ag)
    sock = None
    try:
        with TestClient(app) as client:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            sock.sendall(b"\x01")
            held = False
            for _ in range(50):
                if lim.local_active() == 1:
                    held = True
                    break
                time.sleep(0.02)
            assert held and client.get("/healthz").json()["nodeFull"] is True
            exit_at = time.perf_counter()
        # Shutdown must abort the accepted writer. Waiting out the 30s UUID deadline also frees the slot.
        assert time.perf_counter() - exit_at < 2.0
        assert lim.local_active() == 0 and store.list_calls(tid) == []
    finally:
        if sock is not None:
            sock.close()


def test_sip_admit_failure_ends_the_row_and_does_not_rewrite_a_reject():
    import socket
    import uuid
    from api import security
    from api.store import Store
    from starlette.testclient import TestClient
    from transports import audiosocket as A
    port = _free_port()
    store = Store()
    tid = store.create_tenant("a", 10)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    lim = security.ConcurrencyLimiter(4)
    app = _sip_app(port, store, lim, ag)
    orig = store.update_call

    def boom(call_id, **fields):
        if fields.get("status") == "active":
            raise RuntimeError("active write failed")
        return orig(call_id, **fields)

    store.update_call = boom
    sock = None
    try:
        with TestClient(app):
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            sock.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
            row = None
            for _ in range(50):
                calls = store.list_calls(tid)
                if calls and calls[0]["status"] == "ended":
                    row = calls[0]
                    break
                time.sleep(0.05)
            assert row is not None and row["end_reason"] == "admit_failed"
            assert lim.local_active() == 0
    finally:
        if sock is not None:
            sock.close()
        store.update_call = orig

    # Drain wins inside admit and ends the row before the exception. The finally must leave that reason.
    port2 = _free_port()
    app2 = _sip_app(port2, store, lim, ag)
    sock2 = None
    try:
        with TestClient(app2) as client:
            sock2 = socket.create_connection(("127.0.0.1", port2), timeout=5)
            held = False
            for _ in range(50):
                if lim.local_active() == 1:
                    held = True
                    break
                time.sleep(0.02)
            assert held
            client.app.state.draining = True
            sock2.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
            row = None
            for _ in range(50):
                calls = [c for c in store.list_calls(tid) if c["end_reason"] == "draining"]
                if calls:
                    row = calls[0]
                    break
                time.sleep(0.05)
            assert row is not None and row["status"] == "ended" and row["end_reason"] == "draining"
            assert lim.local_active() == 0
    finally:
        if sock2 is not None:
            sock2.close()


def test_node_cap_zero_admits_a_second_call_and_healthz_stays_ready():
    from test_api import silence
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=0)
    c, store, lim = make(settings)
    k, _tid = tenant(c, limit=10)
    h = {"X-API-Key": k}
    first = c.post("/v1/calls", json={}, headers=h).json()
    second = c.post("/v1/calls", json={}, headers=h).json()
    path = lambda call: call["joinUrl"].replace("wss://vox.example.com", "")
    with c.websocket_connect(path(first)) as ws1:
        ws1.send_bytes(silence(1)[0])
        with c.websocket_connect(path(second)) as ws2:
            ws2.send_bytes(silence(1)[0])
            deadline = time.time() + 2
            while time.time() < deadline and not (
                    store.get_call_any(first["id"])["status"] == "active"
                    and store.get_call_any(second["id"])["status"] == "active"):
                time.sleep(0.02)
            assert store.get_call_any(first["id"])["status"] == "active"
            assert store.get_call_any(second["id"])["status"] == "active"
            assert lim.local_active() == 2
            hz = c.get("/healthz")
            assert hz.status_code == 200 and hz.json()["nodeFull"] is False and hz.json()["ok"] is True


def test_concurrent_voice_webhooks_cannot_exceed_the_node_cap():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from api import security
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=1)
    c, store, lim = make(settings)
    k, tid = tenant(c, limit=10)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "a"}, headers=h).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    barrier = threading.Barrier(8)

    def once(i):
        barrier.wait()
        form = {"From": f"+1555123000{i}", "CallSid": f"CA{i}"}
        return c.post(f"/twilio/voice/{a['id']}", data=form,
                      headers={"X-Twilio-Signature": security.twilio_signature("tok", url, form)}).text

    with ThreadPoolExecutor(8) as ex:
        texts = list(ex.map(once, range(8)))
    assert sum("<Stream" in t for t in texts) == 1
    assert sum("all of our lines are busy" in t for t in texts) == 7
    assert len(store.list_calls(tid)) == 1 and lim.local_active() == 1
    assert c.get("/healthz").status_code == 503 and c.get("/healthz").json()["nodeFull"] is True


def test_concurrent_joins_cannot_exceed_the_node_cap():
    import threading
    from starlette.websockets import WebSocketDisconnect
    from test_api import silence
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        node_max_calls=1)
    c, store, lim = make(settings)
    k, tid = tenant(c, limit=10)
    h = {"X-API-Key": k}
    calls = [c.post("/v1/calls", json={}, headers=h).json() for _ in range(8)]
    barrier = threading.Barrier(len(calls))
    errors = []

    def join(call):
        barrier.wait()
        path = call["joinUrl"].replace("wss://vox.example.com", "")
        try:
            with c.websocket_connect(path) as ws:
                ws.send_bytes(silence(1)[0])
                threading.Event().wait(0.3)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=join, args=(call,)) for call in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    reasons = [store.get_call_any(call["id"])["end_reason"] for call in calls]
    assert reasons.count("hangup") == 1
    assert reasons.count("node_full") == len(calls) - 1
    assert lim.local_active() == 0 and lim.active(tid) == 0


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

async def test_amd_voicemail_is_spoken_by_the_runtime_that_owns_the_call():
    import asyncio
    from api import security
    from api.callbus import LocalCallBus
    from api.runtime import LimitedTTS, Runtime
    from test_session import EnergyVAD, FakeLLM, FakeTTS

    bus = LocalCallBus()
    c, store, rt = make_outbound(lambda r: httpx.Response(201, json={"sid": "CA1"}))
    rt.bus = bus
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    o = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567",
                                           "voicemailMessage": "please call back"}, headers=H(k)).json()
    url = f"https://vox.example.com/twilio/amd/{o['id']}"
    machine = {"AnsweredBy": "machine_end_beep"}
    sig = {"X-Twilio-Signature": security.twilio_signature("tok", url, machine)}
    assert c.post(f"/twilio/amd/{o['id']}", data=machine).status_code == 403
    assert rt.live.get(o["id"]) is None
    missed = c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig)
    assert missed.json() == {"action": "voicemail", "delivered": False}
    human = {"AnsweredBy": "human"}
    seen = []
    bus.subscribe(o["id"], seen.append)
    hum = c.post(f"/twilio/amd/{o['id']}", data=human,
                 headers={"X-Twilio-Signature": security.twilio_signature("tok", url, human)})
    assert hum.json()["action"] == "none" and seen == []
    bus.unsubscribe(o["id"])

    tts = FakeTTS(word_ms=20)
    ended = []

    class T:
        out_rate = 8000
        call_sid = "CA1"

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            ended.append("END")

    LimitedTTS._sem = None
    try:
        other = Runtime(FakeLLM([["x"]]), lambda v: tts, EnergyVAD, store, bus=bus)
        other.build_session(store.get_call_any(o["id"]), T())
        assert o["id"] not in rt.live and o["id"] in other.live
        hit = c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig)
        assert hit.json() == {"action": "voicemail", "delivered": True}
        for _ in range(40):
            if tts.texts and ended:
                break
            await asyncio.sleep(0.05)
        assert tts.texts == ["please call back"] and ended == ["END"]
        # This node has the call in `live`, so it must speak directly and not also publish
        # (the other runtime is still subscribed and would speak a second time).
        before = list(tts.texts)
        spoke = []

        class S:
            async def say_and_wait(self, m):
                spoke.append(m)

        class Tr:
            async def end(self):
                spoke.append("END")

        rt.live[o["id"]] = (S(), Tr())
        local = c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig)
        assert local.json()["delivered"] is True
        await asyncio.sleep(0.3)
        assert spoke == ["please call back", "END"] and tts.texts == before
        rt.live.pop(o["id"], None)
        other.finalize(o["id"])
        again = c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig)
        assert again.json() == {"action": "voicemail", "delivered": False}
    finally:
        LimitedTTS._sem = None

async def test_voicemail_hangs_up_when_speech_fails_and_ignores_a_second_amd():
    """A TTS failure must still hang up, and a second AMD must not start another message over the first."""
    import asyncio
    from api import security
    from api.callbus import LocalCallBus
    from api.runtime import LimitedTTS, Runtime
    from core.metrics import METRICS
    from test_session import EnergyVAD, FakeLLM, FakeTTS

    def vm_errors() -> float:
        return METRICS.counters.get(("vox_errors_total", (("where", "voicemail"),)), 0.0)

    async def settle(target: float) -> None:
        for _ in range(40):
            if vm_errors() >= target:
                return
            await asyncio.sleep(0.05)
        assert vm_errors() >= target

    bus = LocalCallBus()
    c, store, _rt = make_outbound(lambda r: httpx.Response(201, json={"sid": "CA1"}))
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    o = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567",
                                           "voicemailMessage": "please call back"}, headers=H(k)).json()
    started = asyncio.Event()
    release = asyncio.Event()
    says: list[str] = []
    ended: list[str] = []

    class T:
        out_rate = 8000
        call_sid = "CA1"

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            ended.append("END")

    LimitedTTS._sem = None
    try:
        other = Runtime(FakeLLM([["x"]]), lambda v: FakeTTS(), EnergyVAD, store, bus=bus)
        sess = other.build_session(store.get_call_any(o["id"]), T())

        async def say(msg):
            says.append(msg)
            started.set()
            await release.wait()
            raise RuntimeError("tts down")

        sess.say_and_wait = say
        bus_errors = vm_errors()
        assert bus.publish(o["id"], {"op": "voicemail", "text": "one"})
        assert bus.publish(o["id"], {"op": "voicemail", "text": "two"})
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0.05)
        assert says == ["one"]
        release.set()
        for _ in range(40):
            if ended:
                break
            await asyncio.sleep(0.05)
        assert ended == ["END"]
        # The ignored retry must not count a second failure; the speech error must count once.
        await settle(bus_errors + 1)
        assert vm_errors() == bus_errors + 1
    finally:
        LimitedTTS._sem = None
        release.set()

    # The webhook on the node that owns the socket has the same duty.
    spoke: list[str] = []

    class S:
        async def say_and_wait(self, m):
            spoke.append(m)
            raise RuntimeError("tts down")

    class Tr:
        async def end(self):
            spoke.append("END")

    _rt.live[o["id"]] = (S(), Tr())
    url = f"https://vox.example.com/twilio/amd/{o['id']}"
    machine = {"AnsweredBy": "machine_end_beep"}
    sig = {"X-Twilio-Signature": security.twilio_signature("tok", url, machine)}
    direct_errors = vm_errors()
    assert c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig).json()["delivered"] is True
    again = c.post(f"/twilio/amd/{o['id']}", data=machine, headers=sig)
    assert again.json()["delivered"] is True
    for _ in range(40):
        if spoke.count("END") >= 1:
            break
        time.sleep(0.05)
    assert spoke == ["please call back", "END"]
    # Owning-node speech failure has to show up on the same counter as the bus path.
    await settle(direct_errors + 1)
    assert vm_errors() == direct_errors + 1

async def test_owner_amd_and_a_bus_retry_do_not_both_speak():
    """The owner speaks from the webhook. A retry that lands elsewhere publishes; that must not start a second message."""
    from api import security
    from api.callbus import LocalCallBus
    from api.runtime import LimitedTTS

    bus = LocalCallBus()
    c, store, rt = make_outbound(lambda r: httpx.Response(201, json={"sid": "CA1"}))
    rt.bus = bus
    k, _ = tenant(c)
    a = c.post("/v1/agents", json={"name": "a"}, headers=H(k)).json()
    machine = {"AnsweredBy": "machine_end_beep"}
    release = asyncio.Event()
    says: list[str] = []

    class T:
        out_rate = 8000
        call_sid = "CA1"

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            says.append("END")

    async def say(msg):
        says.append(msg)
        await release.wait()

    def open_call():
        o = c.post("/v1/calls/outbound", json={"agentId": a["id"], "to": "+15551234567",
                                               "voicemailMessage": "please call back"}, headers=H(k)).json()
        sess = rt.build_session(store.get_call_any(o["id"]), T())
        sess.say_and_wait = say
        url = f"https://vox.example.com/twilio/amd/{o['id']}"
        sig = {"X-Twilio-Signature": security.twilio_signature("tok", url, machine)}
        return o["id"], sig

    LimitedTTS._sem = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app),
                                      base_url="https://vox.example.com") as client:
            cid, sig = open_call()
            hit = await client.post(f"/twilio/amd/{cid}", data=machine, headers=sig)
            assert hit.json()["delivered"] is True
            assert bus.publish(cid, {"op": "voicemail", "text": "please call back"}) is True
            await asyncio.sleep(0.05)
            assert says == ["please call back"]
            release.set()
            for _ in range(40):
                if says.count("END") >= 1:
                    break
                await asyncio.sleep(0.05)
            assert says.count("END") == 1

            says.clear()
            release.clear()
            cid, sig = open_call()
            assert bus.publish(cid, {"op": "voicemail", "text": "from-bus"}) is True
            await asyncio.sleep(0.05)
            hit = await client.post(f"/twilio/amd/{cid}", data=machine, headers=sig)
            assert hit.json()["delivered"] is True
            await asyncio.sleep(0.05)
            assert says == ["from-bus"]
    finally:
        release.set()
        LimitedTTS._sem = None
