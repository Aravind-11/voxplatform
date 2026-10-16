"""Call recovery: when a phone call's media stream is lost but the caller is still on the line, Twilio follows the
<Redirect> after <Connect>; the server then hangs up (agent ended it) or reconnects the caller with the conversation."""
import base64
import json
import time

import numpy as np

from api.app import Settings
from core.audio import pcm16_to_mulaw
from tests.unit.test_api import make, sign, tenant, twilio_form

BASE = "https://vox.example.com"


def start_call(c, store, key, tid, **agent):
    a = c.post("/v1/agents", json={"name": "n", **agent}, headers={"X-API-Key": key}).json()
    form = twilio_form()
    xml = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(f"{BASE}/twilio/voice/{a['id']}", form)).text
    return a, xml, store.list_calls(tid)[0]


def resume(c, call_id):
    url = f"{BASE}/twilio/resume/{call_id}"
    form = {"CallSid": "CA1"}
    return c.post(f"/twilio/resume/{call_id}", data=form, headers=sign(url, form))


def talk(ws, token):
    media = lambda p: json.dumps({"event": "media", "streamSid": "MZ1", "media": {  # noqa: E731
        "track": "inbound", "payload": base64.b64encode(pcm16_to_mulaw(p)).decode()}})
    ws.send_text(json.dumps({"event": "connected"}))
    ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
        "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": token}}}))
    loud, quiet = np.full(160, 5000, dtype=np.int16), np.zeros(160, dtype=np.int16)
    for p in [quiet] * 10 + [loud] * 60 + [quiet] * 200:
        ws.send_text(media(p))
    end = time.time() + 5
    while time.time() < end:
        if json.loads(ws.receive_text()).get("event") == "media":
            return


def test_twiml_connects_then_redirects_to_resume():
    c, store, _ = make()
    k, tid = tenant(c)
    _, xml, call = start_call(c, store, k, tid)
    assert xml.index("</Connect>") < xml.index("<Redirect") and f"{BASE}/twilio/resume/{call['id']}" in xml


def test_resume_requires_a_valid_twilio_signature():
    c, store, _ = make()
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    assert c.post(f"/twilio/resume/{call['id']}", data={"CallSid": "CA1"}).status_code == 403


def test_agent_hangup_is_not_resumed():
    c, store, lim = make()
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    store.update_call(call["id"], status="active", end_reason="agent_hangup")     # what the transport records before closing
    r = resume(c, call["id"])
    assert r.status_code == 200 and "<Hangup/>" in r.text and "<Stream" not in r.text
    assert len(store.list_calls(tid)) == 1                                       # no new call created


def test_lost_stream_reconnects_with_history_and_frees_the_slot():
    c, store, lim = make()
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    # the node serving the call died: status stuck at active, a concurrency slot still held, a conversation recorded
    store.update_call(call["id"], status="active", joined=time.time())
    assert lim.acquire(tid, call["id"], 5)
    store.add_message(call["id"], "agent", "Hello, how can I help?", time.time())
    store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    r = resume(c, call["id"])
    assert r.status_code == 200 and "<Stream" in r.text and "<Redirect" in r.text
    old = store.get_call_any(call["id"])
    assert old["status"] == "ended" and old["end_reason"] == "stream_lost" and lim.total_active() == 0
    new = [x for x in store.list_calls(tid) if x["id"] != call["id"]][0]
    assert new["metadata"]["resumedFrom"] == call["id"] and new["metadata"]["resumes"] == 1
    assert new["metadata"]["from"] == "+15550001" and new["tenant_id"] == tid        # same caller, same account
    # the new session starts with the earlier conversation and a short apology instead of the greeting
    sess = c.app.state.runtime.build_session(new, type("T", (), {"out_rate": 8000, "call_sid": "CA1",
                                                                  "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                  "end": lambda s: None})())
    assert [m["content"] for m in sess.history[1:]] == ["Hello, how can I help?", "I want Tuesday at eleven."]
    assert [m["role"] for m in sess.history[1:]] == ["assistant", "user"]
    assert "cut off" in sess.cfg.greeting and "Hello" not in sess.cfg.greeting


def test_resumes_are_capped_so_a_broken_call_cannot_loop():
    c, store, _ = make()
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    cur = call["id"]
    for _ in range(2):                                                           # two resumes allowed
        store.update_call(cur, status="active")
        r = resume(c, cur)
        assert "<Stream" in r.text
        cur = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == cur][0]["id"]
    store.update_call(cur, status="active")
    r = resume(c, cur)
    assert "<Stream" not in r.text and "technical difficulties" in r.text and "<Hangup/>" in r.text


def test_real_stream_close_by_twilio_is_marked_stream_closed_and_agent_end_marked_agent_hangup():
    c, store, lim = make()
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        talk(ws, call["join_token"])
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    cur = store.get_call_any(call["id"])
    assert cur["status"] == "ended" and cur["end_reason"] == "stream_closed" and lim.total_active() == 0
    # Twilio then asks what to do: the caller is still there, so the call is resumed
    assert "<Stream" in resume(c, call["id"]).text


def test_unknown_or_browser_calls_just_hang_up_and_draining_node_refuses():
    c, store, _ = make()
    k, tid = tenant(c)
    assert "<Hangup/>" in resume(c, "call_nope").text
    browser = c.post("/v1/calls", json={}, headers={"X-API-Key": k}).json()
    assert "<Hangup/>" in resume(c, browser["id"]).text
    c.post("/admin/drain", headers={"X-Admin-Key": "adm"})
    assert resume(c, "call_x").status_code == 503


async def test_transport_end_records_the_reason_before_closing_the_stream():
    from transports.twilio import TwilioTransport
    order = []

    class WS:
        async def close(self):
            order.append("close")
    tr = TwilioTransport(WS(), None)
    tr.before_end = lambda: order.append("recorded")
    assert tr.ended_by_us is False
    await tr.end()
    assert order == ["recorded", "close"] and tr.ended_by_us is True
