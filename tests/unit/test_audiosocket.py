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
    assert len(StreamResampler(16000, 16000).push(block)) == len(block)

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
    assert sess.frames and all(len(f) == 512 for f in sess.frames)
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
