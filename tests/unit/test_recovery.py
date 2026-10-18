"""Call recovery: when a phone call's media stream is lost but the caller is still on the line, Twilio follows the
<Redirect> after <Connect>; the server then hangs up (agent ended it) or reconnects the caller with the conversation."""
import base64
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import time
import uuid

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
    new = [x for x in store.list_calls(tid) if x["id"] != call["id"]][0]
    assert old["status"] == "ended" and old["end_reason"] == "stream_lost"
    assert lim.total_active() == 1                                          # dead leg freed, replacement already held
    assert lim.try_admit(tid, new["id"], 5, 32) == "ok" and lim.total_active() == 1
    assert new["metadata"]["resumedFrom"] == call["id"] and new["metadata"]["resumes"] == 1
    assert new["metadata"]["from"] == "+15550001" and new["tenant_id"] == tid        # same caller, same account
    # the new session starts with the earlier conversation and a short apology instead of the greeting
    sess = c.app.state.runtime.build_session(new, type("T", (), {"out_rate": 8000, "call_sid": "CA1",
                                                                  "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                  "end": lambda s: None})())
    assert [m["content"] for m in sess.history[1:]] == ["Hello, how can I help?", "I want Tuesday at eleven."]
    assert [m["role"] for m in sess.history[1:]] == ["assistant", "user"]
    assert "cut off" in sess.cfg.greeting and "Hello" not in sess.cfg.greeting


def test_failed_warm_transfer_keeps_history_and_the_apology():
    c, store, _ = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!"},
               headers={"X-API-Key": k}).json()
    call = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666"})
    store.add_message(call["id"], "agent", "Hello, how can I help?", time.time())
    store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    path, q = f"/twilio/transfer-result/{call['id']}", "to=billing"
    form = {"DialCallStatus": "no-answer"}
    r = c.post(f"{path}?{q}", data=form, headers=sign(f"{BASE}{path}?{q}", form))
    assert r.status_code == 200 and "<Connect>" in r.text
    new = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == call["id"]][0]
    assert new["metadata"].get("resumeReason") == "transfer"
    sess = c.app.state.runtime.build_session(new, type("T", (), {"out_rate": 8000, "call_sid": "CA1",
                                                                  "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                  "end": lambda s: None})())
    assert [m["content"] for m in sess.history[1:]] == ["Hello, how can I help?", "I want Tuesday at eleven."]
    assert [m["role"] for m in sess.history[1:]] == ["assistant", "user"]
    assert "not available" in sess.cfg.greeting and "cut off" not in sess.cfg.greeting


def test_tools_doc_does_not_say_a_failed_warm_transfer_starts_fresh():
    """docs/TOOLS.md and the STATUS.md warm-transfer row are what an integrator follows.
    They must describe the returned session, not a blank one."""
    root = pathlib.Path(__file__).resolve().parents[2]
    text = (root / "docs" / "TOOLS.md").read_text()
    line = next(ln for ln in text.splitlines() if '"mode":"warm"' in ln)
    assert "fresh conversation" not in line
    assert "earlier transcript" in line and "not-available apology" in line
    row = next(ln for ln in (root / "docs" / "STATUS.md").read_text().splitlines()
               if ln.startswith("| Warm transfer"))
    assert "fresh conversation" not in row
    assert "keeps the earlier transcript" in row and "not-available apology" in row


def test_stream_loss_after_a_failed_transfer_is_not_labeled_transfer():
    """The transfer apology is only for the call created by transfer-result. A later dropped stream is a reconnect."""
    c, store, _ = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!"},
               headers={"X-API-Key": k}).json()
    call = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666"})
    store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    path, q = f"/twilio/transfer-result/{call['id']}", "to=billing"
    form = {"DialCallStatus": "no-answer"}
    assert c.post(f"{path}?{q}", data=form, headers=sign(f"{BASE}{path}?{q}", form)).status_code == 200
    transferred = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == call["id"]][0]
    assert transferred["metadata"].get("resumeReason") == "transfer"
    store.update_call(transferred["id"], status="active")
    assert "<Stream" in resume(c, transferred["id"]).text
    lost = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == transferred["id"]][0]
    assert lost["metadata"].get("resumeReason") != "transfer"
    assert "resumeReason" not in lost["metadata"]
    sess = c.app.state.runtime.build_session(lost, type("T", (), {"out_rate": 8000, "call_sid": "CA1",
                                                                   "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                   "end": lambda s: None})())
    assert "cut off" in sess.cfg.greeting and "not available" not in sess.cfg.greeting
    assert "I want Tuesday at eleven." in [m["content"] for m in sess.history]


def test_resumes_are_capped_so_a_broken_call_cannot_loop():
    c, store, lim = make()
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
    assert store.get_call_any(cur)["status"] == "ended" and lim.local_active() == 0


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


def test_sip_recover_seeds_history_from_the_lost_audiosocket_call():
    """POST /sip/recover with the AudioSocket uuid (not the call id) continues the transcript on a new sip call."""
    c, store, lim = make(Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret"))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n", "greeting": "Hello!"}, headers={"X-API-Key": k}).json()
    call = store.create_call(tid, a["id"], "sip", a["config"],
                             {"callSid": "sip-uuid-1", "from": "+15550001", "to": "+15550002", "channel": "sip"})
    store.update_call(call["id"], status="active", joined=time.time())
    assert lim.acquire(tid, call["id"], 5)
    store.add_message(call["id"], "agent", "Hello, how can I help?", time.time())
    store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    # A wrong token must not resume first. Checking only 403 would still pass after the row was ended.
    assert len(store.list_calls(tid)) == 1 and lim.total_active() == 1
    for bad in ({}, {"X-Vox-Sip-Token": "wrong"}):
        denied = c.post("/sip/recover", json={"uuid": "sip-uuid-1"}, headers=bad)
        assert denied.status_code == 403
        assert len(store.list_calls(tid)) == 1
        held = store.get_call_any(call["id"])
        assert held["status"] == "active" and held.get("end_reason") is None
        assert lim.total_active() == 1
    r = c.post("/sip/recover", json={"uuid": "sip-uuid-1"}, headers={"X-Vox-Sip-Token": "sip-secret"})
    assert r.status_code == 200 and r.json()["agentId"] == a["id"] and r.json()["callId"] != call["id"]
    old = store.get_call_any(call["id"])
    assert old["status"] == "ended" and old["end_reason"] == "stream_lost" and lim.total_active() == 0
    new = store.get_call_any(r.json()["callId"])
    assert new["medium"] == "sip" and new["metadata"]["resumedFrom"] == call["id"] and new["metadata"]["resumes"] == 1
    sess = c.app.state.runtime.build_session(new, type("T", (), {"out_rate": 8000, "call_sid": "sip-uuid-1",
                                                                  "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                  "end": lambda s: None})())
    assert [m["content"] for m in sess.history[1:]] == ["Hello, how can I help?", "I want Tuesday at eleven."]
    assert [m["role"] for m in sess.history[1:]] == ["assistant", "user"]
    hung = store.create_call(tid, a["id"], "sip", a["config"], {"callSid": "sip-uuid-2", "channel": "sip"})
    store.update_call(hung["id"], status="ended", ended=time.time(), end_reason="agent_hangup")
    refused = c.post("/sip/recover", json={"uuid": "sip-uuid-2"}, headers={"X-Vox-Sip-Token": "sip-secret"})
    assert refused.status_code == 200 and refused.json()["action"] == "hangup"
    assert not [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == hung["id"]]
    before_missing = len(store.list_calls(tid))
    missing = c.post("/sip/recover", json={"uuid": "missing"}, headers={"X-Vox-Sip-Token": "sip-secret"})
    assert missing.status_code == 404
    assert len(store.list_calls(tid)) == before_missing
    assert not any(row["metadata"].get("callSid") == "missing" for row in store.list_calls(tid))
    assert store.find_call_by_sid("missing", medium="sip") is None
    off, *_ = make()
    assert off.post("/sip/recover", json={"uuid": "sip-uuid-1"}, headers={"X-Vox-Sip-Token": "sip-secret"}).status_code == 404


def test_find_call_by_sid_escapes_like_wildcards():
    """% and _ in a callSid are literal. An unescaped LIKE would return a newer wider row."""
    from api.store import Store

    store = Store()
    tid = store.create_tenant("a")
    ag = store.create_agent(tid, "n", {})
    pct = store.create_call(tid, ag["id"], "sip", {}, {"callSid": "100%", "channel": "sip"})
    store.create_call(tid, ag["id"], "sip", {}, {"callSid": "100xyz", "channel": "sip"})
    under = store.create_call(tid, ag["id"], "sip", {}, {"callSid": "ab_cd", "channel": "sip"})
    store.create_call(tid, ag["id"], "sip", {}, {"callSid": "abXcd", "channel": "sip"})
    only = store.create_call(tid, ag["id"], "sip", {}, {"callSid": "%", "channel": "sip"})
    store.create_call(tid, ag["id"], "sip", {}, {"callSid": "plain", "channel": "sip"})
    assert store.find_call_by_sid("100%", medium="sip")["id"] == pct["id"]
    assert store.find_call_by_sid("ab_cd", medium="sip")["id"] == under["id"]
    assert store.find_call_by_sid("%", medium="sip")["id"] == only["id"]
    assert store.find_call_by_sid("nope", medium="sip") is None


def test_sip_recover_continues_the_sip_row_not_a_newer_websocket_call_sid():
    """A tenant can store the same callSid on a newer web call. Recover still follows the sip row."""
    c, store, lim = make(Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret"))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n", "greeting": "Hello!"}, headers={"X-API-Key": k}).json()
    sip = store.create_call(tid, a["id"], "sip", a["config"], {"callSid": "shared-sid", "channel": "sip"})
    store.update_call(sip["id"], status="active", joined=time.time())
    assert lim.acquire(tid, sip["id"], 5)
    store.add_message(sip["id"], "agent", "Hello, how can I help?", time.time())
    store.add_message(sip["id"], "user", "I want Tuesday at eleven.", time.time())
    web = store.create_call(tid, a["id"], "webSocket", a["config"], {"callSid": "shared-sid"})
    assert store.find_call_by_sid("shared-sid")["id"] == web["id"]
    assert store.find_call_by_sid("shared-sid", medium="sip")["id"] == sip["id"]
    before = len(store.list_calls(tid))
    r = c.post("/sip/recover", json={"uuid": "shared-sid"}, headers={"X-Vox-Sip-Token": "sip-secret"})
    assert r.status_code == 200 and r.json()["callId"] not in (sip["id"], web["id"])
    stayed = store.get_call_any(web["id"])
    assert stayed["status"] == "created" and stayed.get("end_reason") is None and stayed.get("ended") is None
    old = store.get_call_any(sip["id"])
    assert old["status"] == "ended" and old["end_reason"] == "stream_lost" and lim.total_active() == 0
    new = store.get_call_any(r.json()["callId"])
    assert new["medium"] == "sip" and new["metadata"]["resumedFrom"] == sip["id"]
    assert len(store.list_calls(tid)) == before + 1
    sess = c.app.state.runtime.build_session(new, type("T", (), {"out_rate": 8000, "call_sid": "shared-sid",
                                                                  "send_audio": lambda s, *a: None, "clear": lambda s: None,
                                                                  "end": lambda s: None})())
    assert [m["content"] for m in sess.history[1:]] == ["Hello, how can I help?", "I want Tuesday at eleven."]


class _SortedSetRedis:
    """The sorted-set commands RedisLimiter uses, shared by two nodes when redis-server is absent."""

    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}

    def register_script(self, _lua: str):
        def call(keys=None, args=None):
            key = keys[0]
            now, exp, member, cap = float(args[0]), float(args[1]), str(args[2]), int(float(args[3]))
            z = self.sets.setdefault(key, {})
            for item, score in list(z.items()):
                if score <= now:
                    del z[item]
            if member in z:
                z[member] = exp
                return 1
            if len(z) >= cap:
                return 0
            z[member] = exp
            return 1
        return call

    def zrem(self, key, member):
        self.sets.get(key, {}).pop(str(member), None)

    def zscore(self, key, member):
        return self.sets.get(key, {}).get(str(member))

    def zcount(self, key, low, _high):
        floor = float("-inf") if low in ("-inf", b"-inf") else float(low)
        return sum(score >= floor for score in self.sets.get(key, {}).values())

    def zadd(self, key, mapping, xx=False):
        z = self.sets.setdefault(key, {})
        for member, score in mapping.items():
            if xx and str(member) not in z:
                continue
            z[str(member)] = float(score)

    def scan_iter(self, match="*"):
        prefix = str(match).rstrip("*")
        return [key for key in self.sets if key.startswith(prefix)]


def _two_node_limiters():
    """Owner and non-owner RedisLimiters over one set. redis-server when installed, else the in-memory set."""
    from api.security import RedisLimiter

    if shutil.which("redis-server"):
        try:
            import redis
        except ImportError:
            redis = None
        if redis is not None:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            client = redis.Redis(port=port, decode_responses=True)
            for _ in range(50):
                try:
                    client.ping()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.05)
            else:
                proc.terminate()
                raise RuntimeError("redis-server did not start")
            kw = dict(default_limit=4, prefix="sip45:")
            return RedisLimiter(f"redis://127.0.0.1:{port}/0", **kw), RedisLimiter(f"redis://127.0.0.1:{port}/0", **kw), proc.terminate
    client = _SortedSetRedis()
    kw = dict(default_limit=4, prefix="sip45:", client=client)
    return RedisLimiter(**kw), RedisLimiter(**kw), lambda: None


def _redis_lab(limiter):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD

    store = Store()
    tid = store.create_tenant("a")
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hello!", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    settings = Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret")
    app = create_app(store, rt, limiter, None, settings)
    return SimpleNamespace(app=app, store=store, rt=rt, tid=tid, ag=ag, client=TestClient)


def _member_live(limiter, tenant_id, call_id) -> bool:
    score = limiter.r.zscore(limiter._key(tenant_id), call_id)
    return score is not None and float(score) >= time.time()


def test_other_node_recover_does_not_free_a_live_redis_slot():
    """POST /sip/recover is not sticky. A slot another node still holds is not a dead leg."""
    owner, other, stop = _two_node_limiters()
    try:
        lab = _redis_lab(other)
        uid = "sip-owned-elsewhere"
        call = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"], {"callSid": uid, "channel": "sip"})
        lab.store.update_call(call["id"], status="active", joined=time.time())
        assert owner.acquire(lab.tid, call["id"], 4)
        assert call["id"] not in lab.rt.live and other.local_active() == 0
        headers = {"X-Vox-Sip-Token": "sip-secret"}
        with lab.client(lab.app) as c:
            refused = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
            assert refused.status_code == 409
            row = lab.store.get_call_any(call["id"])
            assert row["status"] == "active" and row.get("end_reason") is None
            assert _member_live(owner, lab.tid, call["id"]) and owner.active(lab.tid) == 1
            assert len(lab.store.list_calls(lab.tid)) == 1
            owner.release(lab.tid, call["id"])
            assert not _member_live(owner, lab.tid, call["id"])
            resumed = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
            assert resumed.status_code == 200 and resumed.json()["callId"] != call["id"]
            old = lab.store.get_call_any(call["id"])
            assert old["status"] == "ended" and old["end_reason"] == "stream_lost"
            assert len(lab.store.list_calls(lab.tid)) == 2
            nxt = lab.store.get_call_any(resumed.json()["callId"])
            assert nxt["medium"] == "sip" and nxt["metadata"]["resumedFrom"] == call["id"]
    finally:
        stop()


def test_same_node_redis_dead_leg_still_ends_and_remote_hangup_stays_held():
    """This node's _held is the dead leg we release. Another node's agent_hangup is still its call."""
    owner, other, stop = _two_node_limiters()
    try:
        lab = _redis_lab(other)
        headers = {"X-Vox-Sip-Token": "sip-secret"}
        dead_uid = "sip-held-here"
        dead = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"], {"callSid": dead_uid, "channel": "sip"})
        lab.store.update_call(dead["id"], status="active", joined=time.time())
        assert other.acquire(lab.tid, dead["id"], 4)
        assert dead["id"] not in lab.rt.live
        hang_uid = "sip-hang-elsewhere"
        hang = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"], {"callSid": hang_uid, "channel": "sip"})
        lab.store.update_call(hang["id"], status="active", joined=time.time(), end_reason="agent_hangup")
        assert owner.acquire(lab.tid, hang["id"], 4)
        with lab.client(lab.app) as c:
            same = c.post("/sip/recover", json={"uuid": dead_uid}, headers=headers)
            assert same.status_code == 200 and same.json()["callId"] != dead["id"]
            row = lab.store.get_call_any(dead["id"])
            assert row["status"] == "ended" and row["end_reason"] == "stream_lost"
            assert not _member_live(other, lab.tid, dead["id"]) and other.local_active() == 0
            before = len(lab.store.list_calls(lab.tid))
            refused = c.post("/sip/recover", json={"uuid": hang_uid}, headers=headers)
            assert refused.status_code == 409
            held = lab.store.get_call_any(hang["id"])
            assert held["status"] == "active" and held["end_reason"] == "agent_hangup"
            assert _member_live(owner, lab.tid, hang["id"])
            assert len(lab.store.list_calls(lab.tid)) == before
            owner.release(lab.tid, hang["id"])
            done = c.post("/sip/recover", json={"uuid": hang_uid}, headers=headers)
            assert done.status_code == 200 and done.json() == {"action": "hangup"}
            closed = lab.store.get_call_any(hang["id"])
            assert closed["status"] == "ended" and closed["end_reason"] == "agent_hangup"
            assert len(lab.store.list_calls(lab.tid)) == before
            assert not _member_live(owner, lab.tid, hang["id"])
    finally:
        stop()


def test_next_audiosocket_keeps_the_recovered_transcript():
    """The AudioSocket that follows /sip/recover must run that call. A new blank row drops the transcript."""
    import socket
    import uuid
    from fastapi.testclient import TestClient
    from api import security
    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD
    from transports import audiosocket as A

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a")
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hello!", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    settings = Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret",
                        audiosocket_port=port, audiosocket_agent=ag["id"])
    lim = security.ConcurrencyLimiter(default_limit=2)
    app = create_app(store, rt, lim, None, settings)
    uid = str(uuid.uuid4())
    call = store.create_call(tid, ag["id"], "sip", ag["config"],
                             {"callSid": uid, "from": "+15550001", "to": "+15550002", "channel": "sip"})
    store.update_call(call["id"], status="active", joined=time.time())
    assert lim.acquire(tid, call["id"], 5)
    store.add_message(call["id"], "agent", "Hello, how can I help?", time.time())
    store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    with TestClient(app) as c:
        r = c.post("/sip/recover", json={"uuid": uid}, headers={"X-Vox-Sip-Token": "sip-secret"})
        assert r.status_code == 200
        new_id = r.json()["callId"]
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(A.pack(A.KIND_UUID, uuid.UUID(uid).bytes))
        live = None
        for _ in range(100):
            live = app.state.runtime.live.get(new_id)
            if live:
                break
            time.sleep(0.01)
        assert live, "AudioSocket opened a different call, so the recovered transcript was not attached"
        body = [(m["role"], m["content"]) for m in live[0].history]
        assert ("assistant", "Hello, how can I help?") in body
        assert ("user", "I want Tuesday at eleven.") in body
        assert len(store.list_calls(tid)) == 2
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        for _ in range(100):
            if store.get_call_any(new_id)["status"] == "ended":
                break
            time.sleep(0.05)
        assert store.get_call_any(new_id)["status"] == "ended" and lim.total_active() == 0


def test_sip_recover_leaves_a_live_call_alone_and_a_retry_returns_the_same_row():
    """Recover while this node still holds the socket must not free the slot or open a second call.
    A second POST of a recover that already created the continuation returns that row."""
    c, store, lim = make(Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret"))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n", "greeting": "Hello!"}, headers={"X-API-Key": k}).json()
    call = store.create_call(tid, a["id"], "sip", a["config"], {"callSid": "sip-live", "channel": "sip"})
    store.update_call(call["id"], status="active", joined=time.time())
    assert lim.acquire(tid, call["id"], 5)
    c.app.state.runtime.live[call["id"]] = (object(), object())
    headers = {"X-Vox-Sip-Token": "sip-secret"}
    refused = c.post("/sip/recover", json={"uuid": "sip-live"}, headers=headers)
    assert refused.status_code == 409
    assert store.get_call_any(call["id"])["status"] == "active"
    assert lim.total_active() == 1
    assert len(store.list_calls(tid)) == 1
    del c.app.state.runtime.live[call["id"]]
    first = c.post("/sip/recover", json={"uuid": "sip-live"}, headers=headers)
    assert first.status_code == 200
    again = c.post("/sip/recover", json={"uuid": "sip-live"}, headers=headers)
    assert again.status_code == 200 and again.json()["callId"] == first.json()["callId"]
    assert len(store.list_calls(tid)) == 2
    assert lim.total_active() == 0


def test_unknown_or_browser_calls_just_hang_up_and_draining_node_refuses():
    c, store, lim = make()
    k, tid = tenant(c)
    assert "<Hangup/>" in resume(c, "call_nope").text
    browser = c.post("/v1/calls", json={}, headers={"X-API-Key": k}).json()
    assert "<Hangup/>" in resume(c, browser["id"]).text
    a = c.post("/v1/agents", json={"name": "n", "systemPrompt": "Be brief.", "greeting": "Hello"},
               headers={"X-API-Key": k}).json()
    cfg = store.get_agent_any(a["id"])["config"]
    hung = store.create_call(tid, a["id"], "twilio", cfg, {"from": "+15550001", "to": "+15550002"})
    store.update_call(hung["id"], status="active", end_reason="agent_hangup")
    live = store.create_call(tid, a["id"], "twilio", cfg, {"from": "+15550001", "to": "+15550002"})
    assert lim.acquire(tid, live["id"], 10)
    xfer = store.create_call(tid, a["id"], "twilio", cfg, {"from": "+15550001", "to": "+15550002"})
    store.update_call(xfer["id"], status="active", joined=time.time())
    assert lim.acquire(tid, xfer["id"], 10)
    assert lim.local_active() == 2
    c.post("/admin/drain", headers={"X-Admin-Key": "adm"})
    # Hangup reasons win over drain, and an unknown id is still TwiML rather than a 503 JSON error.
    hung_up = resume(c, hung["id"])
    assert hung_up.status_code == 200 and "<Hangup/>" in hung_up.text and "busy" not in hung_up.text
    unknown = resume(c, "call_x")
    assert unknown.status_code == 200 and "<Hangup/>" in unknown.text
    before = len(store.list_calls(tid))
    refused = resume(c, live["id"])
    assert refused.status_code == 200 and "all of our lines are busy" in refused.text and "<Stream" not in refused.text
    assert store.get_call_any(live["id"])["end_reason"] == "stream_lost"
    assert len(store.list_calls(tid)) == before and lim.local_active() == 1
    path = f"/twilio/transfer-result/{xfer['id']}?to=billing"
    miss = {"DialCallStatus": "no-answer"}
    tr = c.post(path, data=miss, headers=sign(f"{BASE}{path}", miss))
    assert tr.status_code == 200 and "all of our lines are busy" in tr.text and "<Stream" not in tr.text
    assert "cannot take your call" not in tr.text
    assert len(store.list_calls(tid)) == before and lim.local_active() == 0


def test_draining_resume_and_transfer_redirect_when_overflow_is_set():
    settings = Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok",
                        overflow_url="https://overflow.example/busy")
    c, store, lim = make(settings)
    k, tid = tenant(c)
    _, _, call = start_call(c, store, k, tid)
    assert lim.local_active() == 1
    own = store.create_call(tid, call["agent_id"], "twilio", call["config"], {"from": "+15550001", "to": "+15550002"})
    store.update_call(own["id"], status="active", joined=time.time())
    assert lim.acquire(tid, own["id"], 10) and lim.local_active() == 2
    c.post("/admin/drain", headers={"X-Admin-Key": "adm"})
    r = resume(c, call["id"])
    assert r.status_code == 200 and "<Redirect>https://overflow.example/busy</Redirect>" in r.text
    assert "<Stream" not in r.text and lim.local_active() == 1
    assert store.get_call_any(call["id"])["end_reason"] == "stream_lost"
    before = len(store.list_calls(tid))
    path = f"/twilio/transfer-result/{own['id']}?to=billing"
    miss = {"DialCallStatus": "no-answer"}
    tr = c.post(path, data=miss, headers=sign(f"{BASE}{path}", miss))
    assert tr.status_code == 200 and "<Redirect>https://overflow.example/busy</Redirect>" in tr.text
    assert "<Stream" not in tr.text and len(store.list_calls(tid)) == before and lim.local_active() == 0


def _poll(pred, seconds=2.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _sip_lab(concurrency=None):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import security
    from api.app import create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD
    from transports import audiosocket as audio

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a", concurrency_limit=concurrency)
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hello!", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    settings = Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret",
                        audiosocket_port=port, audiosocket_agent=ag["id"])
    lim = security.ConcurrencyLimiter(default_limit=4)
    app = create_app(store, rt, lim, None, settings)
    return SimpleNamespace(app=app, store=store, rt=rt, lim=lim, tid=tid, ag=ag, port=port,
                           client=TestClient, audio=audio)


def _lost_sip(lab, uid, *, status="active", hold_slot=False):
    call = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"],
                                 {"callSid": uid, "from": "+15550001", "to": "+15550002", "channel": "sip"})
    if status == "active":
        lab.store.update_call(call["id"], status="active", joined=time.time())
        if hold_slot:
            assert lab.lim.acquire(lab.tid, call["id"], 5)
    else:
        lab.store.update_call(call["id"], status="ended", ended=time.time(), joined=time.time(), end_reason=status)
    lab.store.add_message(call["id"], "agent", "Hello, how can I help?", time.time())
    lab.store.add_message(call["id"], "user", "I want Tuesday at eleven.", time.time())
    return call


def _open_uuid(lab, uid):
    sock = socket.create_connection(("127.0.0.1", lab.port), timeout=5)
    sock.settimeout(5)
    sock.sendall(lab.audio.pack(lab.audio.KIND_UUID, uuid.UUID(uid).bytes))
    return sock


def _push_uuid(lab, uid):
    """Send a uuid and wait until the server closes. Used when the socket must not join."""
    sock = _open_uuid(lab, uid)
    sock.settimeout(5)
    seen_eof = False
    try:
        while True:
            try:
                chunk = sock.recv(4096)
            except (ConnectionResetError, BrokenPipeError):
                seen_eof = True
                break
            except socket.timeout:
                break
            if chunk == b"":
                seen_eof = True
                break
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return seen_eof


def _history(sess):
    return [(m["role"], m["content"]) for m in sess.history]


def _wait_live(lab, call_id, seconds=2.0):
    found = {}

    def ready():
        live = lab.rt.live.get(call_id)
        if live:
            found["live"] = live
            return True
        return False

    assert _poll(ready, seconds), f"call {call_id} was not admitted"
    return found["live"]


def test_sip_recover_stops_at_two_resumes():
    """Two successful SIP resumes then a third POST is the same cap as Twilio. A missing check would keep creating rows."""
    c, store, _ = make(Settings(admin_key="adm", public_base_url=BASE, twilio_auth_token="tok", sip_token="sip-secret"))
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n", "greeting": "Hello!"}, headers={"X-API-Key": k}).json()
    call = store.create_call(tid, a["id"], "sip", a["config"], {"callSid": "sip-cap", "channel": "sip"})
    store.update_call(call["id"], status="active", joined=time.time())
    headers = {"X-Vox-Sip-Token": "sip-secret"}
    cur = call["id"]
    for _ in range(2):
        r = c.post("/sip/recover", json={"uuid": "sip-cap"}, headers=headers)
        assert r.status_code == 200 and r.json().get("callId") and r.json()["callId"] != cur
        cur = r.json()["callId"]
        store.update_call(cur, status="ended", ended=time.time(), joined=time.time(), end_reason="stream_closed")
    assert len(store.list_calls(tid)) == 3
    r = c.post("/sip/recover", json={"uuid": "sip-cap"}, headers=headers)
    assert r.status_code == 200 and r.json() == {"action": "hangup", "reason": "resume cap"}
    assert len(store.list_calls(tid)) == 3
    last = store.get_call_any(cur)
    assert last["status"] == "ended" and last["end_reason"] == "stream_closed"


def test_dead_sip_hangup_ends_the_row_and_frees_its_slot():
    """A recover hangup for a leg this process is not running must end it. Leaving it active keeps the slot."""
    lab = _sip_lab()
    headers = {"X-Vox-Sip-Token": "sip-secret"}

    def open_row(status="active", end_reason=None, resumes=0, hold=False):
        uid = str(uuid.uuid4())
        meta = {"callSid": uid, "channel": "sip"}
        if resumes:
            meta["resumes"] = resumes
        call = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"], meta)
        fields = {}
        if status != "created":
            fields["status"] = status
            fields["joined"] = time.time()
        if end_reason is not None:
            fields["end_reason"] = end_reason
        if fields:
            lab.store.update_call(call["id"], **fields)
        if hold:
            assert lab.lim.acquire(lab.tid, call["id"], 1)
        return uid, call["id"]

    def recover(c, uid):
        return c.post("/sip/recover", json={"uuid": uid}, headers=headers)

    def assert_dead_hangup(c, uid, call_id, body, end_reason):
        before = len(lab.store.list_calls(lab.tid))
        assert call_id not in lab.rt.live
        r = recover(c, uid)
        assert r.status_code == 200 and r.json() == body
        row = lab.store.get_call_any(call_id)
        assert row["status"] == "ended" and row["end_reason"] == end_reason and row["ended"] is not None
        assert lab.lim.total_active() == 0
        assert len(lab.store.list_calls(lab.tid)) == before
        assert _push_uuid(lab, uid)
        assert len(lab.store.list_calls(lab.tid)) == before
        again = lab.store.get_call_any(call_id)
        assert again["status"] == "ended" and again["end_reason"] == end_reason
        assert lab.lim.total_active() == 0 and call_id not in lab.rt.live

    with lab.client(lab.app) as c:
        uid, cid = open_row(resumes=2, hold=True)
        lab.rt.live[cid] = (object(), object())
        before = len(lab.store.list_calls(lab.tid))
        live = recover(c, uid)
        assert live.status_code == 409
        row = lab.store.get_call_any(cid)
        assert row["status"] == "active" and row.get("end_reason") is None and row.get("ended") is None
        assert lab.lim.total_active() == 1 and len(lab.store.list_calls(lab.tid)) == before
        del lab.rt.live[cid]
        assert lab.lim.acquire(lab.tid, "other", 1) is False
        assert_dead_hangup(c, uid, cid, {"action": "hangup", "reason": "resume cap"}, "stream_lost")
        assert lab.lim.acquire(lab.tid, "other", 1) is True
        lab.lim.release(lab.tid, "other")

        uid, cid = open_row(status="created", resumes=2)
        assert lab.store.get_call_any(cid)["status"] == "created"
        assert_dead_hangup(c, uid, cid, {"action": "hangup", "reason": "resume cap"}, "stream_lost")

        uid, cid = open_row(end_reason="agent_hangup", hold=True)
        lab.rt.live[cid] = (object(), object())
        before = len(lab.store.list_calls(lab.tid))
        live = recover(c, uid)
        assert live.status_code == 200 and live.json() == {"action": "hangup"}
        row = lab.store.get_call_any(cid)
        assert row["status"] == "active" and row["end_reason"] == "agent_hangup" and row.get("ended") is None
        assert lab.lim.total_active() == 1 and len(lab.store.list_calls(lab.tid)) == before
        del lab.rt.live[cid]
        assert_dead_hangup(c, uid, cid, {"action": "hangup"}, "agent_hangup")

        for reason in ("transfer", "no_credit"):
            uid, cid = open_row(end_reason=reason, hold=True)
            assert lab.lim.total_active() == 1
            assert_dead_hangup(c, uid, cid, {"action": "hangup"}, reason)


def test_rejected_resume_keeps_the_same_continuation_and_transcript():
    """A full tenant cap must not spend the resume. Two rejected sockets and a retry POST stay on one call id."""
    lab = _sip_lab(concurrency=1)
    assert lab.store.get_tenant(lab.tid)["concurrency_limit"] == 1
    uid = str(uuid.uuid4())
    _lost_sip(lab, uid, status="stream_closed")
    assert lab.lim.acquire(lab.tid, "blocker", 1)
    headers = {"X-Vox-Sip-Token": "sip-secret"}
    with lab.client(lab.app) as c:
        first = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
        assert first.status_code == 200
        cid = first.json()["callId"]
        assert lab.store.get_call_any(cid)["metadata"]["resumes"] == 1
        assert len(lab.store.list_calls(lab.tid)) == 2
        assert _push_uuid(lab, uid)
        row = lab.store.get_call_any(cid)
        assert row["status"] == "created" and row.get("end_reason") is None and row.get("joined") is None
        assert row["metadata"]["resumes"] == 1 and len(lab.store.list_calls(lab.tid)) == 2
        assert lab.lim.total_active() == 1
        # Admit's write stuck (handler died before restore): the retry POST must not mint resumes=2.
        lab.store.update_call(cid, status="ended", ended=time.time(), end_reason="concurrency_limit")
        again = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
        assert again.status_code == 200 and again.json()["callId"] == cid
        row = lab.store.get_call_any(cid)
        assert row["status"] == "created" and row.get("end_reason") is None and row["metadata"]["resumes"] == 1
        assert len(lab.store.list_calls(lab.tid)) == 2
        assert _push_uuid(lab, uid)
        row = lab.store.get_call_any(cid)
        assert row["status"] == "created" and row["metadata"]["resumes"] == 1
        lab.lim.release(lab.tid, "blocker")
        assert lab.lim.total_active() == 0
        sock = _open_uuid(lab, uid)
        try:
            live = _wait_live(lab, cid)
            body = _history(live[0])
            assert ("assistant", "Hello, how can I help?") in body
            assert ("user", "I want Tuesday at eleven.") in body
            assert "cut off" in live[0].cfg.greeting
            assert lab.store.get_call_any(cid)["metadata"]["resumes"] == 1
            assert len(lab.store.list_calls(lab.tid)) == 2
            assert lab.lim.total_active() == 1
        finally:
            try:
                sock.sendall(lab.audio.pack(lab.audio.KIND_HANGUP))
            except OSError:
                pass
            sock.close()
        assert _poll(lambda: lab.store.get_call_any(cid)["status"] == "ended")
        assert lab.lim.total_active() == 0


def test_last_allowed_resume_stays_when_that_continuation_never_joined():
    """resumes=2 is still allowed. A cap check before the never-joined return would hang this row up."""
    lab = _sip_lab()
    uid = str(uuid.uuid4())
    parent = _lost_sip(lab, uid, status="stream_lost")
    cont = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"],
                                 {"callSid": uid, "channel": "sip", "from": "+15550001", "to": "+15550002",
                                  "resumedFrom": parent["id"], "resumes": 2})
    assert lab.store.find_call_by_sid(uid, medium="sip")["id"] == cont["id"]
    assert cont["status"] == "created" and cont.get("joined") is None and cont.get("end_reason") is None
    headers = {"X-Vox-Sip-Token": "sip-secret"}
    with lab.client(lab.app) as c:
        before = len(lab.store.list_calls(lab.tid))
        assert before == 2 and lab.lim.total_active() == 0
        kept = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
        assert kept.status_code == 200 and kept.json()["callId"] == cont["id"]
        row = lab.store.get_call_any(cont["id"])
        assert row["status"] == "created" and row.get("end_reason") is None and row.get("joined") is None
        assert row["metadata"]["resumes"] == 2
        assert len(lab.store.list_calls(lab.tid)) == before and lab.lim.total_active() == 0
        # Admit wrote the refusal and died before restore. The cap must not spend this resume.
        lab.store.update_call(cont["id"], status="ended", ended=time.time(), end_reason="concurrency_limit")
        again = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
        assert again.status_code == 200 and again.json()["callId"] == cont["id"]
        row = lab.store.get_call_any(cont["id"])
        assert row["status"] == "created" and row.get("end_reason") is None and row.get("joined") is None
        assert row.get("ended") is None and row["metadata"]["resumes"] == 2
        assert len(lab.store.list_calls(lab.tid)) == before and lab.lim.total_active() == 0
        sock = _open_uuid(lab, uid)
        try:
            live = _wait_live(lab, cont["id"])
            body = _history(live[0])
            assert ("assistant", "Hello, how can I help?") in body
            assert ("user", "I want Tuesday at eleven.") in body
            assert lab.store.get_call_any(cont["id"])["metadata"]["resumes"] == 2
            assert len(lab.store.list_calls(lab.tid)) == before
            assert lab.lim.total_active() == 1
        finally:
            try:
                sock.sendall(lab.audio.pack(lab.audio.KIND_HANGUP))
            except OSError:
                pass
            sock.close()
        assert _poll(lambda: lab.store.get_call_any(cont["id"])["status"] == "ended")
        assert len(lab.store.list_calls(lab.tid)) == before
        assert lab.lim.total_active() == 0


def test_agent_hangup_then_same_audiosocket_uuid_does_not_create_a_call():
    lab = _sip_lab()
    uid = str(uuid.uuid4())
    with lab.client(lab.app) as c:
        sock = _open_uuid(lab, uid)
        try:
            assert _poll(lambda: bool(lab.rt.live))
            cid = next(iter(lab.rt.live))
            _sess, transport = lab.rt.live[cid]
            fut = c.portal.start_task_soon(transport.end)
            sock.settimeout(0.2)
            deadline = time.time() + 3
            while time.time() < deadline and not (
                    fut.done() and lab.store.get_call_any(cid)["status"] == "ended"):
                try:
                    sock.recv(4096)
                except socket.timeout:
                    pass
            fut.result(timeout=1)
            assert lab.store.get_call_any(cid)["end_reason"] == "agent_hangup"
            assert cid not in lab.rt.live
        finally:
            sock.close()
        before = len(lab.store.list_calls(lab.tid))
        assert before == 1
        for reason_uid, reason in ((str(uuid.uuid4()), "transfer"), (str(uuid.uuid4()), "no_credit")):
            seeded = lab.store.create_call(lab.tid, lab.ag["id"], "sip", lab.ag["config"],
                                           {"callSid": reason_uid, "channel": "sip"})
            lab.store.update_call(seeded["id"], status="ended", ended=time.time(), end_reason=reason)
        assert _push_uuid(lab, uid)
        assert len(lab.store.list_calls(lab.tid)) == before + 2
        assert lab.store.get_call_any(cid)["status"] == "ended"
        assert lab.store.get_call_any(cid)["end_reason"] == "agent_hangup"
        assert not lab.rt.live
        for reason in ("transfer", "no_credit"):
            rows = [row for row in lab.store.list_calls(lab.tid) if row.get("end_reason") == reason]
            assert len(rows) == 1
            assert _push_uuid(lab, rows[0]["metadata"]["callSid"])
            assert len([row for row in lab.store.list_calls(lab.tid) if row.get("end_reason") == reason]) == 1
        assert len(lab.store.list_calls(lab.tid)) == before + 2


def test_second_audiosocket_while_recovered_call_is_live_does_not_create_a_call():
    lab = _sip_lab()
    uid = str(uuid.uuid4())
    call = _lost_sip(lab, uid, hold_slot=True)
    headers = {"X-Vox-Sip-Token": "sip-secret"}
    with lab.client(lab.app) as c:
        recovered = c.post("/sip/recover", json={"uuid": uid}, headers=headers)
        assert recovered.status_code == 200
        cid = recovered.json()["callId"]
        assert lab.lim.total_active() == 0
        sock = _open_uuid(lab, uid)
        try:
            live = _wait_live(lab, cid)
            assert ("user", "I want Tuesday at eleven.") in _history(live[0])
            assert lab.lim.total_active() == 1
            other = _open_uuid(lab, uid)
            try:
                other.settimeout(2)
                closed = False
                deadline = time.time() + 2
                while time.time() < deadline:
                    if len(lab.store.list_calls(lab.tid)) > 2 or lab.lim.total_active() > 1:
                        break
                    try:
                        chunk = other.recv(64)
                    except socket.timeout:
                        continue
                    if chunk == b"" or (chunk and chunk[0] == lab.audio.KIND_HANGUP):
                        closed = True
                        break
                assert closed, "the duplicate socket joined the call"
                assert len(lab.store.list_calls(lab.tid)) == 2
                assert lab.lim.total_active() == 1
                assert list(lab.rt.live) == [cid]
                assert ("user", "I want Tuesday at eleven.") in _history(lab.rt.live[cid][0])
                assert c.post("/sip/recover", json={"uuid": uid}, headers=headers).status_code == 409
                assert lab.store.get_call_any(call["id"])["end_reason"] == "stream_lost"
            finally:
                other.close()
        finally:
            try:
                sock.sendall(lab.audio.pack(lab.audio.KIND_HANGUP))
            except OSError:
                pass
            sock.close()
        assert _poll(lambda: lab.store.get_call_any(cid)["status"] == "ended")
        assert lab.lim.total_active() == 0


def test_audiosocket_without_recover_post_keeps_the_lost_transcript():
    """A socket for an active uuid this process is not running resumes it. stream_closed does too."""
    lab = _sip_lab()
    uid = str(uuid.uuid4())
    call = _lost_sip(lab, uid, hold_slot=True)
    with lab.client(lab.app):
        sock = _open_uuid(lab, uid)
        cid = None
        try:
            assert _poll(lambda: any(row["id"] != call["id"] and lab.rt.live.get(row["id"])
                                     for row in lab.store.list_calls(lab.tid)))
            cid = next(row["id"] for row in lab.store.list_calls(lab.tid) if row["id"] != call["id"])
            live = lab.rt.live[cid]
            assert ("user", "I want Tuesday at eleven.") in _history(live[0])
            assert "cut off" in live[0].cfg.greeting
            assert len(lab.store.list_calls(lab.tid)) == 2
            nxt = lab.store.get_call_any(cid)
            assert nxt["metadata"]["resumedFrom"] == call["id"] and nxt["metadata"]["resumes"] == 1
            assert lab.store.get_call_any(call["id"])["status"] == "ended"
            assert lab.store.get_call_any(call["id"])["end_reason"] == "stream_lost"
            assert lab.lim.total_active() == 1
        finally:
            try:
                sock.sendall(lab.audio.pack(lab.audio.KIND_HANGUP))
            except OSError:
                pass
            sock.close()
        assert cid and _poll(lambda: lab.store.get_call_any(cid)["status"] == "ended")
        assert lab.store.get_call_any(cid)["end_reason"] == "stream_closed"
        assert lab.lim.total_active() == 0
        sock = _open_uuid(lab, uid)
        try:
            assert _poll(lambda: len(lab.store.list_calls(lab.tid)) == 3 and any(
                lab.rt.live.get(row["id"]) for row in lab.store.list_calls(lab.tid) if row["id"] not in (call["id"], cid)))
            third = next(row for row in lab.store.list_calls(lab.tid) if row["id"] not in (call["id"], cid))
            live = lab.rt.live[third["id"]]
            assert third["metadata"]["resumedFrom"] == cid and third["metadata"]["resumes"] == 2
            assert ("user", "I want Tuesday at eleven.") in _history(live[0])
            assert "cut off" in live[0].cfg.greeting
            assert lab.store.get_call_any(cid)["end_reason"] == "stream_closed"
            assert lab.lim.total_active() == 1
        finally:
            try:
                sock.sendall(lab.audio.pack(lab.audio.KIND_HANGUP))
            except OSError:
                pass
            sock.close()
        assert _poll(lambda: lab.lim.total_active() == 0)


def test_kamailio_pins_call_id_local_part_and_asterisk2_is_a_second_node():
    """The last character of a Call-ID is the host. '.com' must not send every INVITE to one Asterisk."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "deploy")
    kam = open(os.path.join(root, "sip", "kamailio.cfg"), encoding="utf-8").read()
    assert 'if ($ci =~ "[02468aceACE]@")' in kam
    assert 'if ($ci =~ "[02468aceACE]$")' not in kam
    pin = re.compile(r"[02468aceACE]@")
    host_tail = re.compile(r"[02468aceACE]$")
    even, odd = "abc124@sip.example.com", "abc125@sip.example.com"
    assert pin.search(even) and pin.search(odd) is None
    assert host_tail.search(even) is None and host_tail.search(odd) is None
    ext = open(os.path.join(root, "sip", "extensions.conf"), encoding="utf-8").read()
    ext2 = open(os.path.join(root, "sip", "extensions.api2.conf"), encoding="utf-8").read()
    assert "api1:9092" in ext and "api2:9092" not in ext
    assert "api2:9092" in ext2 and "api1:9092" not in ext2
    reg = open(os.path.join(root, "sip", "pjsip.register.conf"), encoding="utf-8").read()
    noreg = open(os.path.join(root, "sip", "pjsip.no-register.conf"), encoding="utf-8").read()
    pjsip = open(os.path.join(root, "sip", "pjsip.conf"), encoding="utf-8").read()
    assert "type=registration" in reg
    assert "type=registration" not in noreg and "type=registration" not in pjsip
    assert '#include "pjsip.register.conf"' in pjsip
    compose = open(os.path.join(root, "docker-compose.yml"), encoding="utf-8").read()
    primary, _, secondary = compose.partition("\n  asterisk2:")
    assert "pjsip.no-register.conf" not in primary and "extensions.api2.conf" not in primary
    assert "pjsip.no-register.conf" in secondary and "extensions.api2.conf" in secondary


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
