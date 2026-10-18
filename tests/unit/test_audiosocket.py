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
        assert app.state.limiter.local_active() == 0


def test_number_normalize_and_setup_rows_expire():
    from api.store import Store, normalize_e164
    assert normalize_e164(" +1 555 111 0000 ") == "+15551110000"
    s = Store()
    s.put_number("tnt", "+1 555 111 0000", "agt_b")
    got = s.get_number("+15551110000")
    assert got["tenant_id"] == "tnt" and got["agent_id"] == "agt_b" and got["e164"] == "+15551110000"
    # Asterisk's _X. pattern posts ${EXTEN} as digits. The row is stored with a leading '+'.
    bare = s.get_number("15551110000")
    assert bare is not None and bare["agent_id"] == "agt_b" and bare["e164"] == "+15551110000"
    dialplan = open("deploy/sip/extensions.conf", encoding="utf-8").read()
    assert "Goto(${EXTEN:1},1)" in dialplan
    assert "Goto(_X.,1)" not in dialplan
    assert s.delete_number("other", "+15551110000") is False
    assert s.delete_number("tnt", "+15551110000") is True and s.get_number("+15551110000") is None
    s.put_sip_setup("old", "+1", "+15551110000", created=time.time() - 90)
    assert s.get_sip_setup("old") is None
    s.put_sip_setup("new", "+1999", "+15551110000")
    assert s.get_sip_setup("new")["exten"] == "+15551110000" and s.get_sip_setup("new")["caller"] == "+1999"


def test_dialed_number_routes_to_the_mapped_agent_and_no_setup_uses_the_default():
    """Two agents: +15551110000 belongs to B. A setup row for that exten answers as B; a call with no setup uses the default."""
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
    key = security.generate_key()
    store.add_key(tid, security.hash_key(key), "k")
    cfg = {"systemPrompt": "", "voice": "af_heart", "greeting": "Hello from SIP", "tools": []}
    ag_a = store.create_agent(tid, "A", cfg)
    ag_b = store.create_agent(tid, "B", {**cfg, "systemPrompt": "mapped"})
    other = store.create_tenant("b")
    ag_other = store.create_agent(other, "C", cfg)
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    settings = Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag_a["id"], sip_token="sip-secret")
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=4), None, settings)
    H = {"X-API-Key": key}

    def wait_n(n):
        for _ in range(100):
            calls = store.list_calls(tid)
            if len(calls) >= n and calls[0]["status"] in ("active", "ended"):
                return calls
            time.sleep(0.05)
        raise AssertionError(store.list_calls(tid))

    def wait_ended():
        for _ in range(100):
            calls = store.list_calls(tid)
            if calls and all(c["status"] == "ended" for c in calls):
                return
            time.sleep(0.05)
        raise AssertionError([(c["id"], c["status"]) for c in store.list_calls(tid)])

    def connect(u):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        sock.sendall(A.pack(A.KIND_UUID, u.bytes))
        return sock

    with TestClient(app) as c:
        # The call row is created only after a UUID frame. Closing first must not insert one.
        early = socket.create_connection(("127.0.0.1", port), timeout=5)
        early.close()
        deadline = time.time() + 1.0
        while time.time() < deadline:
            assert store.list_calls(tid) == []
            time.sleep(0.05)

        assert c.post("/internal/sip/setup", data={"uuid": "x", "caller": "1", "exten": "+15551110000"}).status_code == 403
        assert c.post("/internal/sip/setup", data={"uuid": "x", "caller": "1", "exten": "+15551110000"},
                      headers={"X-Vox-Sip-Token": "nope"}).status_code == 403
        assert c.put("/v1/numbers/+15551110000", json={"agentId": "agt_missing"}, headers=H).status_code == 422
        assert c.put("/v1/numbers/+15551110000", json={"agentId": ag_other["id"]}, headers=H).status_code == 422
        put = c.put("/v1/numbers/+15551110000", json={"agentId": ag_b["id"]}, headers=H)
        assert put.status_code == 200 and put.json() == {"e164": "+15551110000", "agentId": ag_b["id"]}
        assert c.get("/v1/numbers", headers=H).json() == [{"e164": "+15551110000", "agentId": ag_b["id"]}]
        assert c.delete("/v1/numbers/+15552220000", headers=H).status_code == 404

        stale = uuid.uuid4()
        old = time.time() - 120
        store.put_sip_setup(str(stale), "+1999", "+15551110000", created=old)
        app.state.sip_setups[str(stale)] = {"uuid": str(stale), "caller": "+1999", "exten": "+15551110000", "created": old}
        sock = connect(stale)
        assert wait_n(1)[0]["agent_id"] == ag_a["id"]          # expired setup does not route
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        wait_ended()

        u = uuid.uuid4()
        ok = c.post("/internal/sip/setup", data={"uuid": str(u), "caller": "+19998887777", "exten": "+15551110000"},
                    headers={"X-Vox-Sip-Token": "sip-secret"})
        assert ok.status_code == 200 and ok.json()["ok"] is True
        assert app.state.sip_setups[str(u)]["exten"] == "+15551110000"
        assert store.get_sip_setup(str(u))["caller"] == "+19998887777"
        sock = connect(u)
        routed = wait_n(2)[0]
        assert routed["agent_id"] == ag_b["id"] and routed["medium"] == "sip"
        assert routed["metadata"]["callSid"] == str(u) and routed["metadata"]["channel"] == "sip"
        assert routed["metadata"]["from"] == "+19998887777" and routed["metadata"]["to"] == "+15551110000"
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        wait_ended()

        plain = uuid.uuid4()
        sock = connect(plain)
        assert wait_n(3)[0]["agent_id"] == ag_a["id"]           # no setup row: the configured default agent
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        wait_ended()

        bare = uuid.uuid4()
        ok = c.post("/internal/sip/setup", data={"uuid": str(bare), "caller": "+19998887777", "exten": "15551110000"},
                    headers={"X-Vox-Sip-Token": "sip-secret"})
        assert ok.status_code == 200
        sock = connect(bare)
        assert wait_n(4)[0]["agent_id"] == ag_b["id"]           # digits-only EXTEN still selects the E.164 mapping
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        wait_ended()

        unmapped = uuid.uuid4()
        ok = c.post("/internal/sip/setup", data={"uuid": str(unmapped), "caller": "+19990000000", "exten": "+15559990000"},
                    headers={"X-Vox-Sip-Token": "sip-secret"})
        assert ok.status_code == 200
        sock = connect(unmapped)
        assert wait_n(5)[0]["agent_id"] == ag_a["id"]           # fresh setup, no numbers row: the default agent
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        wait_ended()

    store2 = Store()
    rt2 = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store2)
    app2 = create_app(store2, rt2, security.ConcurrencyLimiter(2), None, Settings(sip_token=""))
    with TestClient(app2) as c2:
        assert c2.post("/internal/sip/setup", data={"uuid": "z", "caller": "1", "exten": "+1555"},
                       headers={"X-Vox-Sip-Token": "z"}).status_code == 404


def _dialplan_code(raw: str) -> str:
    """Drop an unescaped ';' comment. '\\;' is the dialplan's escaped semicolon, not a comment."""
    out = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            out.append(raw[i:i + 2])
            i += 2
            continue
        if raw[i] == ";":
            break
        out.append(raw[i])
        i += 1
    return "".join(out)


def _dialplan_shell_lines(text: str) -> str:
    """System() and SHELL() are /bin/sh -c. Comments are not executed."""
    lines = []
    for raw in text.splitlines():
        code = _dialplan_code(raw)
        if "System(" in code or "SHELL(" in code:
            lines.append(code)
    return "\n".join(lines)


def test_dialplan_system_does_not_interpolate_raw_caller_or_extension():
    text = open("deploy/sip/extensions.conf", encoding="utf-8").read()
    shell = _dialplan_shell_lines(text)
    assert "${CALLERID(num)}" not in shell
    assert "${EXTEN}" not in shell
    assert "${SIP_SETUP_TOKEN}" not in shell
    assert "FILTER(0-9+,${CALLERID(num)})" in text
    assert "FILTER(0-9+,${EXTEN})" in text
    assert "${SIP_CALLER}" in shell and "${SIP_EXTEN}" in shell
    assert "sip/sip_setup.py" in open("deploy/Dockerfile.asterisk", encoding="utf-8").read()
    code = [_dialplan_code(raw) for raw in text.splitlines()]
    audiosocket = [line for line in code if "AudioSocket(" in line]
    setup = [line for line in code if "System(" in line and "sip_setup.py" in line]
    assert len(audiosocket) == 1 and "${CALLUUID}" in audiosocket[0]
    assert len(setup) == 1 and "${CALLUUID}" in setup[0]
    assert "AudioSocket(${UUID()}" not in text


def test_sip_setup_encodes_plus_and_refuses_shell_characters(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("vox_sip_setup", "deploy/sip/sip_setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sent = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"ok"

    def fake_open(req, timeout=0):
        sent.append((req.data, req.get_header("X-vox-sip-token"), timeout))
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_open)
    token = 'sek"$(id)`rm`'
    assert mod.post_setup("+15551212", "15550000", "abc-DEF", token) is True
    body, header, timeout = sent[0]
    assert b"caller=%2B15551212" in body and b"caller=+" not in body
    assert b"exten=15550000" in body and b"uuid=abc-DEF" in body
    assert header == token and timeout == 2
    assert mod.post_setup('+1";id;echo "', "1555", "abc", "t") is False
    assert mod.post_setup("1555", "$(id)", "abc", "t") is False
    assert mod.post_setup("1555", "1555", "not a uuid", "t") is False
    assert len(sent) == 1

    seen = {}

    def fake_post(caller, exten, uuid, tok, url=mod.DEFAULT_URL, timeout=2.0):
        seen.update(caller=caller, exten=exten, uuid=uuid, token=tok, url=url)
        return False

    monkeypatch.setattr(mod, "post_setup", fake_post)
    monkeypatch.setenv("SIP_SETUP_TOKEN", token)
    assert mod.main(["+15551212", "15550000", "abc-DEF", "http://api1:8080/internal/sip/setup"]) == 0
    assert seen == {
        "caller": "+15551212", "exten": "15550000", "uuid": "abc-DEF",
        "token": token, "url": "http://api1:8080/internal/sip/setup",
    }
    assert mod.main(["+1", "2"]) == 2


def test_put_number_does_not_give_the_number_to_another_tenant():
    from api.store import Store
    s = Store()
    assert s.put_number("A", "+15551110000", "agt_a") is True
    assert s.put_number("B", "+15551110000", "agt_b") is False
    got = s.get_number("+15551110000")
    assert got["tenant_id"] == "A" and got["agent_id"] == "agt_a"
    assert s.put_number("A", "+1 555 111 0000", "agt_a2") is True
    got = s.get_number("+15551110000")
    assert got["tenant_id"] == "A" and got["agent_id"] == "agt_a2"


def test_number_put_is_409_when_another_tenant_owns_it_and_the_agent_stays():
    from fastapi.testclient import TestClient
    from api import security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD

    store = Store()
    tenant_a = store.create_tenant("a")
    tenant_b = store.create_tenant("b")
    key_a = security.generate_key()
    key_b = security.generate_key()
    store.add_key(tenant_a, security.hash_key(key_a), "a")
    store.add_key(tenant_b, security.hash_key(key_b), "b")
    cfg = {"systemPrompt": "", "voice": "af_heart", "greeting": "", "tools": []}
    agent_a = store.create_agent(tenant_a, "A", cfg)
    agent_a2 = store.create_agent(tenant_a, "A2", cfg)
    agent_b = store.create_agent(tenant_b, "B", cfg)
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(2), None, Settings(admin_key="adm"))
    with TestClient(app) as client:
        first = client.put("/v1/numbers/+15551110000", json={"agentId": agent_a["id"]}, headers={"X-API-Key": key_a})
        assert first.status_code == 200
        stolen = client.put("/v1/numbers/+15551110000", json={"agentId": agent_b["id"]}, headers={"X-API-Key": key_b})
        assert stolen.status_code == 409
        again = client.put("/v1/numbers/+15551110000", json={"agentId": agent_a2["id"]}, headers={"X-API-Key": key_a})
        assert again.status_code == 200 and again.json()["agentId"] == agent_a2["id"]
    row = store.get_number("+15551110000")
    assert row["tenant_id"] == tenant_a and row["agent_id"] == agent_a2["id"]
    assert store.list_numbers(tenant_b) == []


def test_sip_doc_routes_by_number_and_passes_caller():
    doc = open("docs/SIP.md", encoding="utf-8").read()
    assert "Caller number is not passed" not in doc
    assert "another way; not built" not in doc
    assert "VOX_SIP_TOKEN" in doc
    assert "PUT /v1/numbers/{e164}" in doc
    assert "{{caller}}" in doc and "{{callee}}" in doc
    assessment = open("docs/ASSESSMENT.md", encoding="utf-8").read()
    assert "pass the caller number" not in assessment
