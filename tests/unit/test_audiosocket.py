import asyncio
import re
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
        tr = A.AudioSocketTransport(r, w, factory, sample_rate=8000)
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
        tr = A.AudioSocketTransport(r, w, factory, sample_rate=8000)
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
        await A.AudioSocketTransport(r, w, factory, sample_rate=8000).run()
        done.set()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(A.pack(A.KIND_AUDIO, b"\x00" * 320))
    await w.drain()
    await asyncio.wait_for(done.wait(), 2)
    assert not sess.started
    srv.close()


async def test_wideband_audiosocket_uses_640_byte_frames_and_does_not_resample():
    """16 kHz AudioSocket frames are 640 bytes, and 16 kHz input is not resampled on the way to the session."""
    from core.audio import StreamResampler
    block = np.arange(320, dtype=np.int16)
    assert np.array_equal(StreamResampler(16000, 16000).push(block), block)

    sess = StubSession()

    async def factory(tr, custom):
        return sess

    box = {}

    async def handler(r, w):
        tr = A.AudioSocketTransport(r, w, factory, sample_rate=16000)
        box["tr"] = tr
        await tr.run()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
    payload = np.full(320, 1000, dtype="<i2").tobytes()
    assert len(payload) == 640
    w.write(A.pack(A.KIND_AUDIO, payload) + A.pack(A.KIND_AUDIO, payload))
    await w.drain()
    for _ in range(100):
        if sess.frames:
            break
        await asyncio.sleep(0.01)
    # 640 samples at 16 kHz fill one 512-sample frame. Upsampling them as 8 kHz yields two.
    assert len(sess.frames) == 1, f"expected one unresampled frame, got {len(sess.frames)}"
    assert np.array_equal(sess.frames[0], np.full(512, 1000, dtype=np.int16))
    tr = box["tr"]
    assert tr.out_rate == 16000
    await tr.send_audio(np.full(16000, 700, dtype=np.int16), 16000)       # 1 s = 50 frames of 640 bytes
    kind, got = await asyncio.wait_for(A.read_message(r), 1)
    assert kind == 0x12 and len(got) == 640
    await tr.end()
    srv.close()


async def test_slin16_kind_is_wideband_and_outbound_16khz_is_not_tagged_8khz():
    """Kind 0x12 is 16 kHz slin. Asterisk plays kind 0x10 as 8 kHz, so a 16 kHz reply must not use it."""
    sess = StubSession()

    async def factory(tr, custom):
        return sess

    box = {}

    async def handler(r, w):
        tr = A.AudioSocketTransport(r, w, factory, sample_rate=16000)
        box["tr"] = tr
        await tr.run()

    srv, port = await serve(handler)
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
    payload = np.full(320, 1000, dtype="<i2").tobytes()
    assert len(payload) == 640
    w.write(A.pack(0x12, payload) + A.pack(0x12, payload))
    await w.drain()
    for _ in range(100):
        if sess.frames:
            break
        await asyncio.sleep(0.01)
    assert sess.frames, "kind 0x12 was ignored, so the 16 kHz caller audio never reached the session"
    assert int(sess.frames[0][0]) == 1000 and len(sess.frames[0]) == 512
    tr = box["tr"]
    await tr.send_audio(np.full(640, 700, dtype=np.int16), 16000)
    kind, got = await asyncio.wait_for(A.read_message(r), 1)
    assert kind == 0x12 and len(got) == 640
    await tr.end()
    srv.close()


def test_dialplan_requests_slin16_on_the_audiosocket_channel():
    """AudioSocket() forces 8 kHz slin and ignores CHANNEL(audioreadformat). The channel driver does not."""
    text = open("deploy/sip/extensions.conf", encoding="utf-8").read()
    code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith(";"))
    assert "Dial(AudioSocket/api1:9092/${UUID()}/c(slin16))" in code
    assert "AudioSocket(" not in code
    assert "audioreadformat" not in code
    assert "Goto(_X.,1)" not in code
    assert "Goto(${EXTEN:1},1)" in code
    modules = open("deploy/sip/modules.conf", encoding="utf-8").read()
    assert "load = chan_audiosocket.so" in modules


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
    settings = Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"], audiosocket_rate=8000)
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


def _asterisk_plays_slin16(major: int, minor: int) -> bool:
    """Kinds 0x11-0x18 (0x12 is 16 kHz slin) shipped in 20.17, 21.12, and 22.7."""
    if major > 22:
        return True
    if major == 22:
        return minor >= 7
    if major == 21:
        return minor >= 12
    if major == 20:
        return minor >= 17
    return False


def test_asterisk_image_plays_slin16_and_bare_metal_sets_the_rate():
    """Debian 12 has no asterisk package, and 20.6 drops kind 0x12. The image must build a release that plays it,
    and the non-docker command must ask the engine for 16 kHz or TTS stays 8 kHz against a slin16 dialplan."""
    docker = open("deploy/Dockerfile.asterisk", encoding="utf-8").read()
    m = re.search(r"(?m)^ARG\s+ASTERISK_VERSION=(\d+)\.(\d+)\.(\d+)\s*$", docker)
    assert m, "Dockerfile.asterisk must pin ASTERISK_VERSION"
    major, minor = int(m.group(1)), int(m.group(2))
    assert _asterisk_plays_slin16(major, minor), (major, minor)
    assert "downloads.asterisk.org" in docker
    assert "asterisk-${ASTERISK_VERSION}.tar.gz" in docker
    assert "chan_audiosocket" in docker and "res_audiosocket" in docker
    flat = docker.replace("\\\n", " ")
    installs = re.findall(r"apt-get\s+install\b([^;&|]*)", flat)
    assert installs, "Dockerfile.asterisk lost its package install"
    for blob in installs:
        pkgs = {tok.split("=")[0] for tok in blob.split() if not tok.startswith("-")}
        assert "asterisk" not in pkgs and "asterisk-modules" not in pkgs
    sip = open("docs/SIP.md", encoding="utf-8").read()
    bare = next(ln for ln in sip.splitlines() if ln.startswith("Without Docker:"))
    assert "18+" not in bare
    assert "20.17" in bare and "21.12" in bare and "22.7" in bare
    assert "VOX_AUDIOSOCKET_PORT=9092" in bare
    assert "VOX_AUDIOSOCKET_AGENT=" in bare
    assert "VOX_AUDIOSOCKET_HOST=127.0.0.1" in bare
    assert "VOX_AUDIOSOCKET_RATE=16000" in bare


def test_wideband_sip_call_through_the_app_speaks_kind_0x12_at_the_session_rate():
    """create_app must pass audiosocket_rate into the transport and build_session must copy out_rate.
    The greeting is then kind 0x12, 640 bytes, not an 8 kHz frame Asterisk would play at half speed."""
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
    settings = Settings(admin_key="adm", audiosocket_port=port, audiosocket_agent=ag["id"], audiosocket_rate=16000)
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, settings)
    with TestClient(app):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
        sock.settimeout(5)
        buf, audio = b"", None
        end = time.time() + 5
        while time.time() < end and audio is None:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 3 and audio is None:
                kind, n = struct.unpack_from(">BH", buf)
                if len(buf) < 3 + n:
                    break
                payload, buf = buf[3:3 + n], buf[3 + n:]
                if kind in (A.KIND_AUDIO, A.KIND_SLIN16):
                    audio = (kind, payload)
        assert audio is not None, "the greeting should arrive as an AudioSocket audio frame"
        kind, payload = audio
        assert kind == A.KIND_SLIN16 == 0x12 and len(payload) == 640
        assert abs(int(np.median(np.frombuffer(payload, dtype="<i2"))) - 500) < 50
        assert rt.live, "build_session did not keep the live session"
        for sess, tr in rt.live.values():
            assert sess.cfg.out_rate == 16000 and tr.out_rate == 16000
        sock.sendall(A.pack(A.KIND_HANGUP))
        sock.close()
        for _ in range(100):
            calls = store.list_calls(tid)
            if calls and calls[0]["status"] == "ended":
                break
            time.sleep(0.05)
        assert calls[0]["medium"] == "sip" and calls[0]["status"] == "ended"
