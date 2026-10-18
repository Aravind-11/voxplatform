import base64
import json
import time

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from api import security
from api.app import Settings, create_app
from api.runtime import Runtime
from api.store import Store
from core.audio import pcm16_to_mulaw


class FakeVAD:
    def prob(self, frame):
        return 1.0 if np.abs(frame).max() > 1000 else 0.0


class FakeLLM:
    async def stream(self, history, audio, followups, tools):
        yield "Hello there. "

    async def transcribe(self, audio):
        return "hi"


class FakeTTS:
    sample_rate = 24000

    async def synthesize(self, text):
        yield np.full(2400, 500, dtype=np.int16)


SETTINGS = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok")


def make(settings=SETTINGS, dispatcher=None):
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    lim = security.ConcurrencyLimiter(default_limit=2)
    app = create_app(store, rt, lim, dispatcher, settings)
    return TestClient(app), store, lim


def tenant(c, name="a", limit=None):
    r = c.post("/admin/tenants", json={"name": name, "concurrencyLimit": limit}, headers={"X-Admin-Key": "adm"})
    assert r.status_code == 200
    return r.json()["apiKey"], r.json()["tenantId"]


def test_admin_and_api_key_auth():
    c, store, _ = make()
    assert c.post("/admin/tenants", json={"name": "x"}).status_code == 403
    assert c.post("/admin/tenants", json={"name": "x"}, headers={"X-Admin-Key": "no"}).status_code == 403
    assert c.get("/v1/agents").status_code == 401
    assert c.get("/v1/agents", headers={"X-API-Key": "vox_nope"}).status_code == 401
    key, tid = tenant(c)
    assert c.get("/v1/agents", headers={"X-API-Key": key}).json() == []
    # only the hash is stored
    assert key not in json.dumps(store._all("SELECT * FROM api_keys"))


def test_admin_disabled_when_no_admin_key():
    c, *_ = make(Settings(admin_key=""))
    assert c.post("/admin/tenants", json={"name": "x"}, headers={"X-Admin-Key": ""}).status_code == 403


def test_tenant_isolation():
    c, *_ = make()
    ka, _ = tenant(c, "a")
    kb, _ = tenant(c, "b")
    ha, hb = {"X-API-Key": ka}, {"X-API-Key": kb}
    agent = c.post("/v1/agents", json={"name": "n", "systemPrompt": "p"}, headers=ha).json()
    call = c.post("/v1/calls", json={"agentId": agent["id"]}, headers=ha).json()
    assert c.get(f"/v1/calls/{call['id']}", headers=ha).status_code == 200
    assert c.get(f"/v1/calls/{call['id']}", headers=hb).status_code == 404
    assert c.get(f"/v1/calls/{call['id']}/messages", headers=hb).status_code == 404
    assert c.post("/v1/calls", json={"agentId": agent["id"]}, headers=hb).status_code == 404
    assert c.get("/v1/calls", headers=hb).json() == [] and c.get("/v1/agents", headers=hb).json() == []


def test_create_call_merges_agent_config_and_overrides():
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "n", "systemPrompt": "base", "voice": "af_bella"}, headers=h).json()
    call = c.post("/v1/calls", json={"agentId": a["id"], "systemPrompt": "override"}, headers=h).json()
    cfg = store.get_call_any(call["id"])["config"]
    assert cfg["systemPrompt"] == "override" and cfg["voice"] == "af_bella"
    assert call["joinUrl"].startswith("wss://vox.example.com/ws/calls/" + call["id"] + "?token=")


def speech(n):  # n frames
    return [(np.full(512, 5000, dtype="<i2")).tobytes()] * n


def silence(n):
    return [np.zeros(512, dtype="<i2").tobytes()] * n


def test_websocket_call_end_to_end_gets_agent_audio_and_records():
    c, store, lim = make()
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers={"X-API-Key": k}).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    got = b""
    with c.websocket_connect(path) as ws:
        for f in silence(5) + speech(15) + silence(40):
            ws.send_bytes(f)
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            m = ws.receive()
            if m.get("bytes"):
                got += m["bytes"]
    assert len(got) >= 2400 * 2
    cur = store.get_call_any(call["id"])
    assert cur["status"] == "ended" and cur["end_reason"] == "hangup" and cur["joined"] and cur["ended"]
    assert lim.total_active() == 0
    roles = [m["role"] for m in store.messages(call["id"])]
    assert "agent" in roles
    # a second join with the same token is refused
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(path) as ws:
            ws.receive_text()
    assert e.value.code == 4401


def test_websocket_bad_token_and_unknown_call():
    c, *_ = make()
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={}, headers={"X-API-Key": k}).json()
    for p in (f"/ws/calls/{call['id']}?token=wrong", "/ws/calls/call_nope?token=x"):
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect(p) as ws:
                ws.receive_text()
        assert e.value.code == 4401


def test_concurrency_limit_rejects_with_1013_and_records_reason():
    c, store, lim = make()
    k, tid = tenant(c, limit=1)
    h = {"X-API-Key": k}
    c1 = c.post("/v1/calls", json={}, headers=h).json()
    c2 = c.post("/v1/calls", json={}, headers=h).json()
    p = lambda call: call["joinUrl"].replace("wss://vox.example.com", "")
    with c.websocket_connect(p(c1)) as ws1:
        ws1.send_bytes(silence(1)[0])
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect(p(c2)) as ws2:
                ws2.receive_text()
        assert e.value.code == 1013
        assert lim.active(tid) == 1
    assert store.get_call_any(c2["id"])["end_reason"] == "concurrency_limit"
    assert lim.active(tid) == 0


def twilio_form():
    return {"CallSid": "CA1", "From": "+15550001", "To": "+15550002"}


def sign(url, form, token="tok"):
    return {"X-Twilio-Signature": security.twilio_signature(token, url, form)}


def test_twilio_signature_matches_official_validator():
    form = twilio_form()
    url = "https://vox.example.com/twilio/voice/agt_1"
    assert security.twilio_signature("tok", url, form) == RequestValidator("tok").compute_signature(url, form)


def test_twilio_webhook_validates_signature_and_returns_twiml():
    c, store, _ = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    assert c.post(f"/twilio/voice/{a['id']}", data=form).status_code == 403
    assert c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form, "wrong")).status_code == 403
    r = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    assert r.status_code == 200 and "<Stream url=\"wss://vox.example.com/ws/twilio/call_" in r.text
    assert '<Parameter name="token"' in r.text
    calls = store.list_calls(tid)
    assert len(calls) == 1 and calls[0]["medium"] == "twilio" and calls[0]["metadata"]["from"] == "+15550001"


def test_twilio_webhook_fails_closed_without_token_and_can_be_opened_explicitly():
    c, *_ = make(Settings(admin_key="adm", twilio_auth_token=""))
    assert c.post("/twilio/voice/agt_x", data=twilio_form()).status_code == 403
    c2, *_ = make(Settings(admin_key="adm", twilio_auth_token="", allow_unsigned_twilio=True))
    assert c2.post("/twilio/voice/agt_missing", data=twilio_form()).status_code == 404


def test_twilio_media_stream_end_to_end():
    c, store, lim = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    xml = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form)).text
    call = store.list_calls(tid)[0]

    def media(pcm8):
        return json.dumps({"event": "media", "streamSid": "MZ1", "media": {
            "track": "inbound", "payload": base64.b64encode(pcm16_to_mulaw(pcm8)).decode()}})

    loud = np.full(160, 5000, dtype=np.int16)
    quiet = np.zeros(160, dtype=np.int16)
    outbound = []
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": call["join_token"]}}}))
        for pcm in [quiet] * 10 + [loud] * 60 + [quiet] * 200:
            ws.send_text(media(pcm))
        deadline = time.time() + 5
        while time.time() < deadline and not any(m.get("event") == "media" for m in outbound):
            outbound.append(json.loads(ws.receive_text()))
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    assert any(m["event"] == "media" and m["streamSid"] == "MZ1" for m in outbound)
    assert store.get_call_any(call["id"])["status"] == "ended" and lim.total_active() == 0


def test_twilio_stream_with_wrong_token_is_closed():
    c, store, lim = make()
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    c.post(f"/twilio/voice/{a['id']}", data=twilio_form(), headers=sign(url, twilio_form()))
    call = store.list_calls(tid)[0]
    assert lim.total_active() == 1                                          # the webhook reserved the slot
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
            ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
                "streamSid": "MZ1", "customParameters": {"token": "bad"}}}))
            ws.receive_text()
    assert e.value.code == 4401
    assert store.get_call_any(call["id"])["status"] == "created" and lim.total_active() == 0


def test_twilio_stream_that_drops_before_start_keeps_its_slot_until_resume():
    c, store, lim = make(Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                                  node_max_calls=1))
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    assert "<Stream" in c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form)).text
    call = store.list_calls(tid)[0]
    assert lim.total_active() == 1 and c.get("/healthz").status_code == 503
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
    assert store.get_call_any(call["id"])["status"] == "created" and lim.total_active() == 1
    rurl = f"https://vox.example.com/twilio/resume/{call['id']}"
    resumed = c.post(f"/twilio/resume/{call['id']}", data={"CallSid": "CA1"}, headers=sign(rurl, {"CallSid": "CA1"}))
    assert "<Stream" in resumed.text and lim.total_active() == 1
    new = [x for x in store.list_calls(tid) if x["id"] != call["id"]][0]
    assert lim.try_admit(tid, new["id"], 10, 1) == "ok" and lim.total_active() == 1
    assert c.get("/healthz").status_code == 503


# ---------------------------------------------------------------- webhooks
async def test_webhook_signed_and_retried_with_backoff():
    seen, sleeps = [], []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(500 if len(seen) < 3 else 200)

    async def fake_sleep(s):
        sleeps.append(s)

    d = security.WebhookDispatcher(httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                                   retries=3, backoff_s=0.5, sleep=fake_sleep)
    ok = await d.send("https://h.example/x", "sec", {"type": "call.ended", "n": 1})
    assert ok and len(seen) == 3 and sleeps == [0.5, 1.0]
    body = seen[-1].content
    assert seen[-1].headers["X-Vox-Signature"] == security.sign_body("sec", body)
    assert seen[-1].headers["X-Vox-Event"] == "call.ended"


async def test_webhook_gives_up_after_retries():
    async def fake_sleep(s):
        pass

    d = security.WebhookDispatcher(httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
                                   retries=2, sleep=fake_sleep)
    assert await d.send("https://h.example/x", "s", {"type": "t"}) is False


def test_limiter_basic():
    l = security.ConcurrencyLimiter(2)
    assert l.acquire("t", "1") and l.acquire("t", "2") and not l.acquire("t", "3")
    assert l.acquire("t", "1")             # idempotent
    l.release("t", "1")
    assert l.acquire("t", "3") and l.active("t") == 2
    assert l.acquire("u", "1")             # tenants independent


def test_try_admit_node_cap_is_atomic_and_a_refusal_takes_no_slot():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lim = security.ConcurrencyLimiter(10)
    barrier = threading.Barrier(16)

    def once(i):
        barrier.wait()
        return lim.try_admit("t", f"c{i}", 10, 1)

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(once, range(16)))
    ok = [i for i, r in enumerate(res) if r == "ok"]
    assert len(ok) == 1 and res.count("node") == 15 and lim.local_active() == 1
    assert lim.try_admit("t", f"c{ok[0]}", 10, 1) == "ok"          # already held, still one slot
    assert lim.local_active() == 1
    lim.release("t", f"c{ok[0]}")
    assert lim.local_active() == 0
    assert lim.try_admit("u", "x", 1, 5) == "ok"
    assert lim.try_admit("u", "y", 1, 5) == "tenant"               # tenant cap, node cap still has room
    assert lim.local_active() == 1 and lim.active("u") == 1
    assert lim.try_admit("t", "a", 10, 0) == "ok" and lim.try_admit("t", "b", 10, 0) == "ok"
    assert lim.local_active() == 3                                 # node_limit 0 does not cap this process


class _FakeRedis:
    """Just enough of redis-py for RedisLimiter.try_admit/release, with no server."""

    def __init__(self, fail_zrem: bool = False):
        self.sets: dict[str, dict] = {}
        self.fail_zrem = fail_zrem

    def register_script(self, _lua):
        def run(keys, args):
            _now, exp, member, cap = args
            members = self.sets.setdefault(keys[0], {})
            if member in members:
                members[member] = float(exp)
                return 1
            if len(members) >= int(float(cap)):
                return 0
            members[member] = float(exp)
            return 1
        return run

    def zrem(self, key, member):
        if self.fail_zrem:
            raise RuntimeError("zrem failed")
        self.sets.get(key, {}).pop(member, None)
        return 1


def test_redis_try_admit_node_cap_is_one_slot_and_release_survives_zrem_failure():
    client = _FakeRedis()
    lim = security.RedisLimiter(client=client, default_limit=10)
    assert lim.try_admit("t", "c1", 10, 1) == "ok"
    assert lim.try_admit("t", "c2", 10, 1) == "node"
    assert lim.local_active() == 1
    assert lim.try_admit("t", "c1", 10, 1) == "ok" and lim.local_active() == 1   # same id, still one slot
    lim.release("t", "c1")
    assert lim.local_active() == 0
    assert lim.try_admit("u", "x", 1, 5) == "ok"
    assert lim.try_admit("u", "y", 1, 5) == "tenant"                             # refusal keeps no node slot
    assert lim.local_active() == 1
    open_lim = security.RedisLimiter(client=_FakeRedis(), default_limit=10)
    assert open_lim.try_admit("t", "a", 10, 0) == "ok" and open_lim.try_admit("t", "b", 10, 0) == "ok"
    assert open_lim.local_active() == 2                                          # node_limit 0 disables the node cap
    broken = _FakeRedis(fail_zrem=True)
    stuck = security.RedisLimiter(client=broken, default_limit=10)
    assert stuck.try_admit("t", "a", 10, 1) == "ok"
    stuck.release("t", "a")                                                      # must not raise
    assert stuck.local_active() == 0
    assert stuck.try_admit("t", "b", 10, 1) == "ok" and stuck.local_active() == 1


def test_redis_release_failure_still_ends_the_call_and_frees_the_slot():
    client = _FakeRedis(fail_zrem=True)
    lim = security.RedisLimiter(client=client, default_limit=10)
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    app = create_app(store, rt, lim, None, Settings(admin_key="adm", public_base_url="https://vox.example.com",
                                                    node_max_calls=1))
    c = TestClient(app)
    key, _tid = tenant(c, limit=10)
    h = {"X-API-Key": key}
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=h).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    with c.websocket_connect(path) as ws:
        ws.send_bytes(silence(1)[0])
        assert store.get_call_any(call["id"])["status"] == "active"
        assert lim.local_active() == 1
    done = store.get_call_any(call["id"])
    assert done["status"] == "ended" and done["end_reason"] == "hangup"
    assert lim.local_active() == 0
    call2 = c.post("/v1/calls", json={}, headers=h).json()
    with c.websocket_connect(call2["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        ws.send_bytes(silence(1)[0])
        assert store.get_call_any(call2["id"])["status"] == "active"
        assert lim.local_active() == 1


# ---------------------------------------------------------------- tools through the API
def test_agent_tools_are_validated_and_credentials_never_returned():
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    r = c.post("/v1/agents", json={"name": "bad", "tools": ["no_such_tool"]}, headers=h)
    assert r.status_code == 422 and "unknown tool" in r.json()["detail"]
    r = c.post("/v1/agents", json={"name": "bad", "tools": [{"name": "x", "description": "d",
                                   "http": {"url": "ftp://nope"}}]}, headers=h)
    assert r.status_code == 422
    tool = {"name": "lookup", "description": "Find it.", "http": {"url": "https://api.example.com/x",
            "headers": {"Authorization": "Bearer s3cret"}}}
    a = c.post("/v1/agents", json={"name": "ok", "tools": [tool, "hangUp"]}, headers=h)
    assert a.status_code == 200
    assert "s3cret" not in json.dumps(a.json()) and "s3cret" not in json.dumps(c.get("/v1/agents", headers=h).json())
    assert "s3cret" in json.dumps(store.get_agent_any(a.json()["id"])["config"])     # stored, just write-only
    r = c.post("/v1/calls", json={"agentId": a.json()["id"], "tools": ["nope"]}, headers=h)
    assert r.status_code == 422                                                      # per-call override is validated too


class HangUpLLM:
    def __init__(self):
        self.n = 0

    async def stream(self, history, audio, followups, tools):
        from core.session import ToolCall
        self.n += 1
        if not followups:
            yield "Goodbye now. "
            yield ToolCall("h1", "hangUp", {"reason": "done"})
        else:
            yield "Take care."

    async def transcribe(self, audio):
        return "bye"


def test_hangUp_tool_ends_the_websocket_call_after_the_agent_has_spoken():
    store = Store()
    rt = Runtime(HangUpLLM(), lambda v: FakeTTS(), FakeVAD, store)
    c = TestClient(create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, SETTINGS))
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={"tools": ["hangUp"]}, headers={"X-API-Key": k}).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    got, closed = b"", False
    with c.websocket_connect(path) as ws:
        for f in silence(5) + speech(15) + silence(40):
            ws.send_bytes(f)
        deadline = time.time() + 8
        while not closed and time.time() < deadline:
            m = ws.receive()
            if m["type"] == "websocket.close":
                closed = True
            elif m.get("bytes"):
                got += m["bytes"]
    assert closed and len(got) >= 2 * 2400 * 2          # both sentences were heard before the line dropped
    cur = store.get_call_any(call["id"])
    assert cur["status"] == "ended"
    assert [m["text"] for m in store.messages(call["id"]) if m["role"] == "agent"] == ["Goodbye now.", "Take care."]


def test_agent_can_be_updated_in_place_and_new_calls_use_it():
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "r", "systemPrompt": "old", "tools": []}, headers=h).json()
    r = c.patch(f"/v1/agents/{a['id']}", json={"systemPrompt": "new", "tools": ["hangUp"]}, headers=h)
    assert r.status_code == 200 and r.json()["id"] == a["id"]
    cfg = store.get_agent_any(a["id"])["config"]
    assert cfg["systemPrompt"] == "new" and cfg["tools"] == ["hangUp"] and cfg["voice"] == "af_heart"
    assert c.patch(f"/v1/agents/{a['id']}", json={"tools": ["nope"]}, headers=h).status_code == 422
    assert c.patch("/v1/agents/agt_missing", json={}, headers=h).status_code == 404
    kb, _ = tenant(c, "other")
    assert c.patch(f"/v1/agents/{a['id']}", json={"systemPrompt": "x"}, headers={"X-API-Key": kb}).status_code == 404


def _make_twilio_rest(handler, **kw):
    """App whose Twilio REST client is a MockTransport. Inbound calls must register a status callback."""
    store = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store, http=http)
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", **kw)
    lim = security.ConcurrencyLimiter(4)
    return TestClient(create_app(store, rt, lim, None, settings)), store, lim


def test_empty_call_sid_does_not_reserve_a_slot():
    """A signed voice form with no CallSid has nothing a status callback can free."""
    c, store, lim = make(Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                                  node_max_calls=1))
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    for form in ({}, {"From": "+15550001", "To": "+15550002"}):
        r = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
        assert r.status_code == 200 and "<Hangup/>" in r.text and "<Stream" not in r.text
    assert store.list_calls(tid) == [] and lim.local_active() == 0
    assert c.get("/healthz").status_code == 200


def test_repeated_voice_webhook_reuses_the_open_call():
    c, store, lim = make(Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                                  node_max_calls=2))
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    first = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    second = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    assert "<Stream" in first.text and first.text == second.text
    assert len(store.list_calls(tid)) == 1 and lim.local_active() == 1
    assert c.get("/healthz").status_code == 200                    # cap is 2; a retry must not take the other slot
    call = store.list_calls(tid)[0]
    done = {"CallStatus": "completed"}
    surl = f"https://vox.example.com/twilio/status/{call['id']}"
    assert c.post(surl, data=done, headers=sign(surl, done)).status_code == 200
    assert lim.local_active() == 0 and store.get_call_any(call["id"])["status"] == "ended"
    again = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    assert "<Stream" in again.text and call["id"] not in again.text
    assert len(store.list_calls(tid)) == 2 and lim.local_active() == 1
    # Same CallSid on another tenant is a different call, not this row's stream token.
    k2, tid2 = tenant(c, "b", limit=10)
    a2 = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k2}).json()
    url2 = f"https://vox.example.com/twilio/voice/{a2['id']}"
    other = c.post(f"/twilio/voice/{a2['id']}", data=form, headers=sign(url2, form))
    assert "<Stream" in other.text and store.list_calls(tid)[0]["id"] not in other.text
    assert len(store.list_calls(tid2)) == 1 and lim.local_active() == 2


def test_inbound_status_callback_frees_a_leg_that_stops_before_start():
    """Hang-up before the socket's start event never reaches /twilio/resume. The call-level callback must."""
    from urllib.parse import parse_qs
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json={"sid": "CA1", "status": "in-progress"})

    c, store, lim = _make_twilio_rest(handler, node_max_calls=1)
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    xml = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    assert "<Stream" in xml.text
    call = store.list_calls(tid)[0]
    assert len(seen) == 1 and seen[0].url.path.endswith("/Calls/CA1.json")
    posted = {key: vals[0] for key, vals in parse_qs(seen[0].content.decode()).items()}
    assert posted["StatusCallback"] == f"https://vox.example.com/twilio/status/{call['id']}"
    assert posted["StatusCallbackMethod"] == "POST"
    assert lim.local_active() == 1 and c.get("/healthz").status_code == 503
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    assert store.get_call_any(call["id"])["status"] == "created" and lim.local_active() == 1
    done = {"CallStatus": "completed"}
    surl = f"https://vox.example.com/twilio/status/{call['id']}"
    assert c.post(surl, data=done, headers=sign(surl, done)).status_code == 200
    ended = store.get_call_any(call["id"])
    assert ended["status"] == "ended" and ended["end_reason"] == "completed"
    assert lim.local_active() == 0 and c.get("/healthz").status_code == 200
    assert len(store.list_calls(tid)) == 1                              # resume was never called


def test_failed_status_callback_does_not_hand_back_a_stream():
    def refuse(req: httpx.Request):
        return httpx.Response(500, text="no")

    c, store, lim = _make_twilio_rest(refuse, node_max_calls=1)
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    r = c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    assert r.status_code == 200 and "<Hangup/>" in r.text and "<Stream" not in r.text
    row = store.list_calls(tid)[0]
    assert row["status"] == "ended" and row["end_reason"] == "status_callback_failed"
    assert lim.local_active() == 0 and c.get("/healthz").status_code == 200

    def explode(req: httpx.Request):
        raise RuntimeError("twilio down")

    c2, store2, lim2 = _make_twilio_rest(explode, node_max_calls=1)
    k2, tid2 = tenant(c2, limit=10)
    a2 = c2.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k2}).json()
    url2 = f"https://vox.example.com/twilio/voice/{a2['id']}"
    r2 = c2.post(f"/twilio/voice/{a2['id']}", data=form, headers=sign(url2, form))
    assert "<Hangup/>" in r2.text and "<Stream" not in r2.text
    row2 = store2.list_calls(tid2)[0]
    assert row2["status"] == "ended" and row2["end_reason"] == "status_callback_failed"
    assert lim2.local_active() == 0 and c2.get("/healthz").status_code == 200


def test_resume_terminal_status_releases_and_in_progress_still_reconnects():
    c, store, lim = make(Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                                  node_max_calls=2))
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    assert "<Stream" in c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form)).text
    call = store.list_calls(tid)[0]
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    assert lim.local_active() == 1 and store.get_call_any(call["id"])["status"] == "created"
    done = {"CallSid": "CA1", "CallStatus": "completed"}
    rurl = f"https://vox.example.com/twilio/resume/{call['id']}"
    resumed = c.post(rurl, data=done, headers=sign(rurl, done))
    assert "<Hangup/>" in resumed.text and "<Stream" not in resumed.text
    assert len(store.list_calls(tid)) == 1
    assert store.get_call_any(call["id"])["status"] == "ended"
    assert store.get_call_any(call["id"])["end_reason"] == "completed"
    assert lim.local_active() == 0

    form2 = {"CallSid": "CA2", "From": "+15550001", "To": "+15550002"}
    assert "<Stream" in c.post(f"/twilio/voice/{a['id']}", data=form2, headers=sign(url, form2)).text
    live = [x for x in store.list_calls(tid) if x["status"] == "created"][0]
    with c.websocket_connect(f"/ws/twilio/{live['id']}") as ws:
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ2"}))
    mid = {"CallSid": "CA2", "CallStatus": "in-progress"}
    rurl2 = f"https://vox.example.com/twilio/resume/{live['id']}"
    again = c.post(rurl2, data=mid, headers=sign(rurl2, mid))
    assert "<Stream" in again.text and lim.local_active() == 1
    assert len(store.list_calls(tid)) == 3


def test_replacement_leg_is_not_left_held_when_its_status_callback_fails():
    n = {"i": 0}

    def handler(req: httpx.Request):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(200, json={"sid": "CA1"})
        return httpx.Response(503, text="later")

    c, store, lim = _make_twilio_rest(handler, node_max_calls=1)
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    assert "<Stream" in c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form)).text
    call = store.list_calls(tid)[0]
    rurl = f"https://vox.example.com/twilio/resume/{call['id']}"
    mid = {"CallSid": "CA1"}
    resumed = c.post(rurl, data=mid, headers=sign(rurl, mid))
    assert "<Hangup/>" in resumed.text and "<Stream" not in resumed.text
    assert lim.local_active() == 0 and c.get("/healthz").status_code == 200
    nxt = [x for x in store.list_calls(tid) if x["id"] != call["id"]]
    assert len(nxt) == 1 and nxt[0]["status"] == "ended" and nxt[0]["end_reason"] == "status_callback_failed"
    # The return leg of a missed transfer has the same hole.
    parent = store.create_call(tid, a["id"], "twilio", a["config"], {"from": "+1555", "to": "+1666", "callSid": "CA9"})
    path, q = f"/twilio/transfer-result/{parent['id']}", "to=billing"
    miss = {"DialCallStatus": "no-answer", "CallSid": "CA9"}
    tr = c.post(f"{path}?{q}", data=miss, headers=sign(f"https://vox.example.com{path}?{q}", miss))
    assert "<Hangup/>" in tr.text and "<Stream" not in tr.text
    assert lim.local_active() == 0
    back = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == parent["id"]]
    assert len(back) == 1 and back[0]["status"] == "ended" and back[0]["end_reason"] == "status_callback_failed"


def test_resume_points_the_new_leg_at_its_own_status_callback():
    from urllib.parse import parse_qs
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json={"sid": "CA1"})

    c, store, lim = _make_twilio_rest(handler, node_max_calls=1)
    k, tid = tenant(c, limit=10)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    form = twilio_form()
    assert "<Stream" in c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form)).text
    call = store.list_calls(tid)[0]
    rurl = f"https://vox.example.com/twilio/resume/{call['id']}"
    mid = {"CallSid": "CA1"}
    resumed = c.post(rurl, data=mid, headers=sign(rurl, mid))
    assert "<Stream" in resumed.text and lim.local_active() == 1
    new = [x for x in store.list_calls(tid) if x["id"] != call["id"]][0]
    assert len(seen) == 2
    posted = {key: vals[0] for key, vals in parse_qs(seen[1].content.decode()).items()}
    assert posted["StatusCallback"].endswith(f"/twilio/status/{new['id']}")
    assert call["id"] not in posted["StatusCallback"]


def test_greeting_is_part_of_the_agent_and_is_spoken_on_a_browser_call():
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    a = c.post("/v1/agents", json={"name": "g", "greeting": "Hi there."}, headers=h).json()
    call = c.post("/v1/calls", json={"agentId": a["id"]}, headers=h).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    got = b""
    with c.websocket_connect(path) as ws:           # the caller says nothing: the agent speaks first
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            m = ws.receive()
            if m.get("bytes"):
                got += m["bytes"]
    assert len(got) >= 2400 * 2
    assert [m["text"] for m in store.messages(call["id"]) if m["role"] == "agent"] == ["Hi there."]
    r = c.patch(f"/v1/agents/{a['id']}", json={"greeting": "Hello again."}, headers=h)
    assert r.json()["config"]["greeting"] == "Hello again."
