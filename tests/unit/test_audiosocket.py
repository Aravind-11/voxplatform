import asyncio
import struct
import time
import uuid

import numpy as np

from transports import audiosocket as A


class StubSession:
    def __init__(self):
        self.frames, self.closed, self.started, self.injected = [], False, False, []

    async def start(self):
        self.started = True

    async def feed(self, frame):
        self.frames.append(frame)

    async def close(self):
        self.closed = True

    async def inject_text(self, t):
        self.injected.append(t)


async def serve(handler):
    srv = await asyncio.start_server(handler, "127.0.0.1", 0)
    return srv, srv.sockets[0].getsockname()[1]


async def test_inbound_audio_uuid_dtmf_and_hangup():
    sess, seen = StubSession(), {}

    async def factory(tr, custom):
        seen.update(custom)
        return sess

    done = asyncio.Event()

    async def handler(r, w):
        tr = A.AudioSocketTransport(r, w, factory)
        seen["tr"] = tr
        await tr.run()
        done.set()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    u = uuid.uuid4()
    w.write(A.pack(A.KIND_UUID, u.bytes))
    tone = (8000 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)).astype("<i2")        # 1 s at 8 kHz
    for i in range(0, 8000, 160):
        w.write(A.pack(A.KIND_AUDIO, tone[i:i + 160].tobytes()))
    w.write(A.pack(A.KIND_DTMF, b"#"))
    w.write(A.pack(A.KIND_HANGUP))
    await w.drain()
    await asyncio.wait_for(done.wait(), 3)
    assert seen["uuid"] == str(u) and sess.started and sess.closed
    assert 29 <= len(sess.frames) <= 31 and all(len(f) == 512 for f in sess.frames)    # 1 s -> 16 kHz frames of 512 samples (32 ms)
    srv.close()


async def test_outbound_audio_is_paced_at_real_time_and_clear_drops_the_queue():
    sess = StubSession()

    async def factory(tr, custom):
        return sess

    box = {}

    async def handler(r, w):
        tr = A.AudioSocketTransport(r, w, factory)
        box["tr"] = tr
        await tr.run()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
    await w.drain()
    for _ in range(100):
        if "tr" in box and sess.started:
            break
        await asyncio.sleep(0.01)
    tr = box["tr"]
    await tr.send_audio(np.full(8000, 1000, dtype=np.int16), 8000)          # 1 s of audio = 50 frames
    t0, got = time.perf_counter(), 0
    while got < 10:
        kind, payload = await A.read_message(r)
        assert kind == A.KIND_AUDIO and len(payload) == 320
        got += 1
    elapsed = time.perf_counter() - t0
    assert 0.12 <= elapsed <= 0.35, f"10 frames should take about 180 ms of real time, took {elapsed:.3f}s"
    await tr.clear()                                                        # interruption: the rest must never be sent
    await asyncio.sleep(0.15)
    extra = 0
    try:
        while True:
            kind, _ = await asyncio.wait_for(A.read_message(r), 0.1)
            extra += kind == A.KIND_AUDIO
    except asyncio.TimeoutError:
        pass
    assert extra <= 2                                                       # at most a frame or two already in flight
    await tr.end()
    kind, _ = await asyncio.wait_for(A.read_message(r), 1)
    assert kind == A.KIND_HANGUP and tr.ended_by_us
    srv.close()


async def test_rejects_connection_that_does_not_start_with_uuid():
    sess = StubSession()

    async def factory(tr, custom):
        return sess

    done = asyncio.Event()

    async def handler(r, w):
        await A.AudioSocketTransport(r, w, factory).run()
        done.set()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(A.pack(A.KIND_AUDIO, b"\x00" * 320))
    await w.drain()
    await asyncio.wait_for(done.wait(), 2)
    assert not sess.started
    srv.close()


def test_full_sip_call_through_the_app_creates_a_call_speaks_back_and_hangs_up():
    """A simulated Asterisk connects to the app's AudioSocket port: a 'sip' call is created, the agent answers, and
    when Asterisk hangs up the call ends in the database."""
    import socket
    from fastapi.testclient import TestClient
    from api import security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a")
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hello from SIP", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    settings = Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"])
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, settings)
    with TestClient(app):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
        got_audio = False
        sock.settimeout(5)
        buf = b""
        end = time.time() + 5
        while time.time() < end and not got_audio:
            buf += sock.recv(4096)
            got_audio = len(buf) >= 3 + 320 and buf[0] == A.KIND_AUDIO
        assert got_audio, "the greeting should arrive as AudioSocket audio frames"
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        for _ in range(100):
            calls = store.list_calls(tid)
            if calls and calls[0]["status"] == "ended":
                break
            time.sleep(0.05)
        assert calls[0]["medium"] == "sip" and calls[0]["status"] == "ended"
        assert calls[0]["metadata"].get("from") == "" and calls[0]["metadata"].get("to") == ""


def test_sip_setup_puts_the_caller_on_the_call_and_into_the_prompt():
    """The setup row is what a later AudioSocket reads, including after the process cache is gone.
    A different UUID does not inherit it, and CallContext.caller is what an HTTP tool sends as `from`."""
    import asyncio
    import json
    import socket

    import httpx
    from fastapi.testclient import TestClient
    from api import security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a")
    caller, callee = "+15551212000", "+15551110000"
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"ok": True})

    lookup = {"name": "lookup_caller", "description": "Look up who is calling.",
              "parameters": {"type": "object", "properties": {}},
              "http": {"url": "https://93.184.216.34/lookup", "method": "POST"},
              "automatic": ["from", "to", "callSid"]}
    ag = store.create_agent(tid, "n", {"systemPrompt": "Caller is {{caller}}", "voice": "af_heart",
                                       "greeting": "", "tools": [lookup]})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    rt.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"], sip_token="sip-secret")
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, settings)
    u = uuid.uuid4()

    def live_call(call_sid: str):
        found = None
        for _ in range(100):
            for cand in store.list_calls(tid):
                if cand["metadata"].get("callSid") == call_sid and cand["id"] in rt.live:
                    found = cand
                    break
            if found:
                return found
            time.sleep(0.05)
        return None

    with TestClient(app) as c:
        denied = c.post("/internal/sip/setup", data={"uuid": str(u), "caller": caller, "callee": callee})
        assert denied.status_code == 403
        assert store.get_sip_setup(str(u)) is None and str(u) not in app.state.sip_setups
        ok = c.post("/internal/sip/setup", data={"uuid": str(u), "caller": caller, "callee": callee},
                    headers={"X-Vox-Sip-Token": "sip-secret"})
        assert ok.status_code == 200
        assert app.state.sip_setups[str(u)]["caller"] == caller
        assert app.state.sip_setups[str(u)]["callee"] == callee
        row = store.get_sip_setup(str(u))
        assert row is not None and row["caller"] == caller and row["callee"] == callee
        app.state.sip_setups.clear()

        other = uuid.uuid4()
        sock_other = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock_other.settimeout(5)
        sock_other.sendall(A.pack(A.KIND_UUID, other.bytes))
        other_call = live_call(str(other))
        assert other_call is not None, store.list_calls(tid)
        assert other_call["metadata"]["from"] == "" and other_call["metadata"]["to"] == ""
        sock_other.sendall(A.pack(A.KIND_HANGUP))
        sock_other.close()
        for _ in range(100):
            if store.get_call_any(other_call["id"])["status"] == "ended":
                break
            time.sleep(0.05)
        assert store.get_call_any(other_call["id"])["status"] == "ended"
        assert store.get_call_any(other_call["id"])["metadata"]["from"] == ""
        assert str(u) not in app.state.sip_setups

        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        sock.sendall(A.pack(A.KIND_UUID, u.bytes))
        call = live_call(str(u))
        assert call is not None, store.list_calls(tid)
        assert call["metadata"]["from"] == caller and call["metadata"]["to"] == callee
        assert call["metadata"]["callSid"] == str(u)
        assert call["metadata"]["channel"] == "sip" and call["medium"] == "sip"
        sess = rt.live[call["id"]][0]
        assert sess.history[0]["content"] == f"Caller is {caller}"          # {{caller}} rendered
        asyncio.run(sess.tools["lookup_caller"]({}))
        assert seen["body"]["from"] == caller and seen["body"]["to"] == callee
        assert seen["body"]["callSid"] == str(u)
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        for _ in range(100):
            if store.get_call_any(call["id"])["status"] == "ended":
                break
            time.sleep(0.05)
        assert store.get_call_any(call["id"])["status"] == "ended"
        assert store.get_call_any(call["id"])["metadata"]["from"] == caller

    store2 = Store()
    app2 = create_app(store2, Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store2),
                      security.ConcurrencyLimiter(2), None, Settings(sip_token=""))
    with TestClient(app2) as c2:
        missing = c2.post("/internal/sip/setup", data={"uuid": "z", "caller": caller, "callee": callee},
                          headers={"X-Vox-Sip-Token": "z"})
        assert missing.status_code == 404
        assert store2.get_sip_setup("z") is None and "z" not in app2.state.sip_setups


def test_dialplan_keeps_the_dialed_number_and_does_not_shell_out_caller_id():
    """_+X. must land in _X. with the digits still in ${EXTEN}. System() is a shell, and $() runs inside
    double quotes, so CALLERID(num) and EXTEN cannot be interpolated into that command.
    The curl and AudioSocket share ${CALLUUID}, and compose passes one token into both sides."""
    text = open("deploy/sip/extensions.conf", encoding="utf-8").read()
    code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith(";"))
    assert "exten => _+X.,1,Goto(${EXTEN:1},1)" in code
    assert "Goto(_X.,1)" not in code
    shell_lines = [ln for ln in code.splitlines() if "System(" in ln or "SHELL(" in ln]
    assert shell_lines, "the dialplan should still post the setup before AudioSocket"
    for ln in shell_lines:
        assert "${CALLERID(" not in ln
        assert "${EXTEN}" not in ln
    assert 'caller=${SAFE_CALLER}' in code and 'callee=${SAFE_CALLEE}' in code
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]

    def at(fragment: str) -> int:
        for i, ln in enumerate(lines):
            if fragment in ln:
                return i
        raise AssertionError(fragment)

    system_at = at("System(")
    assert at("Set(SAFE_CALLER=") < system_at and at("Set(SAFE_CALLEE=") < system_at
    caller_line, callee_line = lines[at("Set(SAFE_CALLER=")], lines[at("Set(SAFE_CALLEE=")]
    assert "FILTER(0-9+" in caller_line and "${CALLERID(num)}" in caller_line
    assert "FILTER(0-9+" in callee_line and "${EXTEN}" in callee_line
    assert at("Set(CALLUUID=") < system_at < at("AudioSocket(${CALLUUID}")
    system = lines[system_at]
    assert "uuid=${CALLUUID}" in system and "/internal/sip/setup" in system
    assert "X-Vox-Sip-Token: ${SIP_SETUP_TOKEN}" in system
    assert "AudioSocket(${UUID()}" not in code
    compose = open("deploy/docker-compose.yml", encoding="utf-8").read()
    assert "VOX_SIP_TOKEN: ${VOX_SIP_TOKEN:-}" in compose
    assert "SIP_SETUP_TOKEN: ${VOX_SIP_TOKEN:-}" in compose


def test_sip_caller_is_dropped_when_the_call_ends_is_deleted_or_expires():
    """The setup row is a second copy of the phone numbers. Finish, DELETE, and retention each remove it."""
    import socket
    from fastapi.testclient import TestClient
    from api import retention, security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store()
    tid = store.create_tenant("a")
    key = security.generate_key()
    store.add_key(tid, security.hash_key(key))
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "", "tools": []})
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None,
                     Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"],
                              sip_token="sip-secret"))
    caller, callee = "+15551212000", "+15551110000"
    u = uuid.uuid4()
    with TestClient(app) as c:
        ok = c.post("/internal/sip/setup", data={"uuid": str(u), "caller": caller, "callee": callee},
                    headers={"X-Vox-Sip-Token": "sip-secret"})
        assert ok.status_code == 200
        assert store.get_sip_setup(str(u))["caller"] == caller
        assert app.state.sip_setups[str(u)]["caller"] == caller
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        sock.sendall(A.pack(A.KIND_UUID, u.bytes))
        call = None
        for _ in range(100):
            for cand in store.list_calls(tid):
                if cand["metadata"].get("callSid") == str(u) and cand["id"] in rt.live:
                    call = cand
                    break
            if call:
                break
            time.sleep(0.05)
        assert call is not None, store.list_calls(tid)
        assert store.get_sip_setup(str(u)) is not None and str(u) in app.state.sip_setups
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        for _ in range(100):
            if store.get_call_any(call["id"])["status"] == "ended":
                break
            time.sleep(0.05)
        ended = store.get_call_any(call["id"])
        assert ended["status"] == "ended" and ended["metadata"]["from"] == caller
        assert store.get_sip_setup(str(u)) is None and str(u) not in app.state.sip_setups

        u2 = str(uuid.uuid4())
        store.put_sip_setup(u2, caller, callee)
        app.state.sip_setups[u2] = {"uuid": u2, "caller": caller, "callee": callee, "created": time.time()}
        parked = store.create_call(tid, ag["id"], "sip", {},
                                   {"callSid": u2, "from": caller, "to": callee, "channel": "sip"})
        store.update_call(parked["id"], status="ended")
        assert c.delete(f"/v1/calls/{parked['id']}", headers={"X-API-Key": key}).status_code == 200
        assert store.get_call_any(parked["id"]) is None
        assert store.get_sip_setup(u2) is None and u2 not in app.state.sip_setups

        now = time.time()
        old_orphan, fresh, linked = "old-orphan", "fresh-orphan", "linked-uuid"
        store.put_sip_setup(old_orphan, caller, callee, now - 40 * 86400)
        store.put_sip_setup(fresh, caller, callee, now - 86400)
        store.put_sip_setup(linked, caller, callee, now - 86400)
        app.state.sip_setups[old_orphan] = {"uuid": old_orphan, "caller": caller, "callee": callee,
                                            "created": now - 40 * 86400}
        app.state.sip_setups[fresh] = {"uuid": fresh, "caller": caller, "callee": callee, "created": now - 86400}
        app.state.sip_setups[linked] = {"uuid": linked, "caller": caller, "callee": callee, "created": now - 86400}
        expired = store.create_call(tid, ag["id"], "sip", {}, {"callSid": linked, "from": caller, "to": callee})
        store.update_call(expired["id"], status="ended", ended=now - 40 * 86400)
        store.update_tenant(tid, retention_days=30)
        assert retention.purge_expired(store, "", setups=app.state.sip_setups) == {tid: 1}
        assert store.get_call_any(expired["id"]) is None
        assert store.get_sip_setup(old_orphan) is None and old_orphan not in app.state.sip_setups
        assert store.get_sip_setup(linked) is None and linked not in app.state.sip_setups
        assert store.get_sip_setup(fresh)["caller"] == caller and fresh in app.state.sip_setups
