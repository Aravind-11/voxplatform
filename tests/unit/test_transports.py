import asyncio
import base64
import json

import numpy as np
import pytest

from core.audio import mulaw_to_pcm16, pcm16_to_mulaw
from core.session import Session, SessionConfig
from core.tts import KokoroTTS
from tests.unit.test_session import Clock, EnergyVAD, FakeLLM, FakeTTS
from transports.browser import BrowserTransport
from transports.twilio import TwilioTransport


class StubSession:
    def __init__(self):
        self.frames, self.closed, self.started = [], False, False

    async def start(self):
        self.started = True

    async def feed(self, frame):
        self.frames.append(frame)

    async def close(self):
        self.closed = True


class FakeTwilioWS:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []

    async def receive_text(self):
        return json.dumps(self.incoming.pop(0))

    async def send_text(self, s):
        self.sent.append(json.loads(s))


def media_msgs(pcm8, n_per=160):
    out = []
    for i in range(0, len(pcm8), n_per):
        out.append({"event": "media", "streamSid": "MZ1",
                    "media": {"track": "inbound", "chunk": str(i), "timestamp": "0",
                              "payload": base64.b64encode(pcm16_to_mulaw(pcm8[i:i + n_per])).decode()}})
    return out


async def test_twilio_inbound_audio_becomes_16k_frames_and_session_closes():
    t = np.arange(8000) / 8000
    tone = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)         # 1 s at 8 kHz
    msgs = ([{"event": "connected", "protocol": "Call", "version": "1.0.0"},
             {"event": "start", "streamSid": "MZ1",
              "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {"agent": "a1"}}}]
            + media_msgs(tone)
            + [{"event": "media", "streamSid": "MZ1",
                "media": {"track": "outbound", "payload": base64.b64encode(b"\xff" * 160).decode()}},
               {"event": "stop", "streamSid": "MZ1"}])
    sess = StubSession()
    seen = {}

    async def factory(tr, custom):
        seen["custom"] = custom
        return sess

    ws = FakeTwilioWS(msgs)
    tr = TwilioTransport(ws, factory)
    await tr.run()
    assert seen["custom"] == {"agent": "a1"} and tr.stream_sid == "MZ1" and tr.call_sid == "CA1"
    assert sess.closed
    # 1 s of 8 kHz -> 16000 samples, minus 192 withheld by the resampler lookahead -> 30 full frames
    assert len(sess.frames) == 30 and all(len(f) == 512 for f in sess.frames)
    audio = np.concatenate(sess.frames).astype(np.float64)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    assert abs(np.argmax(spec) * 16000 / len(audio) - 440) < 5       # tone survives 8k->16k
    assert 6000 < np.max(np.abs(audio)) < 9000


async def test_twilio_outbound_media_mark_clear_and_chunking():
    ws = FakeTwilioWS([])
    tr = TwilioTransport(ws, None)
    tr.stream_sid = "MZ9"
    pcm = (np.arange(400) * 20).astype(np.int16)
    name = await tr.send_audio(pcm, 8000)
    media, mark = ws.sent
    assert media["event"] == "media" and media["streamSid"] == "MZ9"
    raw = base64.b64decode(media["media"]["payload"])
    assert len(raw) == 400 and np.array_equal(mulaw_to_pcm16(raw), mulaw_to_pcm16(pcm16_to_mulaw(pcm)))
    assert mark == {"event": "mark", "streamSid": "MZ9", "mark": {"name": "m1"}} and name == "m1"
    ws.sent.clear()
    name = await tr.send_audio(np.zeros(20000, dtype=np.int16), 8000)
    sizes = [len(base64.b64decode(m["media"]["payload"])) for m in ws.sent if m["event"] == "media"]
    assert sizes == [8000, 8000, 4000] and ws.sent[-1]["mark"]["name"] == "m2" and name == "m2"
    ws.sent.clear()
    await tr.clear()
    assert ws.sent == [{"event": "clear", "streamSid": "MZ9"}]


async def test_twilio_inbound_mark_reaches_the_handler():
    seen = []
    msgs = [{"event": "start", "streamSid": "MZ1",
             "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {}}},
            {"event": "mark", "streamSid": "MZ1", "mark": {"name": "m1"}},
            {"event": "stop", "streamSid": "MZ1"}]
    sess = StubSession()

    async def factory(tr, custom):
        tr.set_mark_handler(seen.append)
        return sess

    tr = TwilioTransport(FakeTwilioWS(msgs), factory)
    await tr.run()
    assert seen == ["m1"] and tr.marks_acked == 1 and sess.closed


async def test_twilio_rejects_non_8k_output():
    tr = TwilioTransport(FakeTwilioWS([]), None)
    with pytest.raises(AssertionError):
        await tr.send_audio(np.zeros(10, dtype=np.int16), 24000)


class MarkFailWS:
    """Accepts media frames, then raises when the playback mark is written."""

    def __init__(self):
        self.sent = []

    async def send_text(self, s):
        obj = json.loads(s)
        if obj.get("event") == "mark":
            raise RuntimeError("mark failed")
        self.sent.append(obj["event"])

    async def close(self):
        return None


async def test_failed_mark_write_holds_say_and_wait_until_the_mark_timeout():
    """Media is already queued when the mark write raises. say_and_wait must not return on the
    local clock: voicemail would hang up while that audio is still in Twilio's buffer."""
    ws = MarkFailWS()
    tr = TwilioTransport(ws, None)
    tr.stream_sid = "MZ1"
    clk = Clock()
    events = []
    sess = Session(llm=FakeLLM([["x."]]), tts=FakeTTS(word_ms=500), vad=EnergyVAD(),
                   send_audio=tr.send_audio, clear_audio=tr.clear, clock=clk,
                   on_event=events.append,
                   cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    task = asyncio.create_task(sess.say_and_wait("Please call us back."))
    for _ in range(400):
        await asyncio.sleep(0)
    assert ws.sent == ["media", "media"]
    assert [str(c.text) for c in sess.tracker._chunks] == ["Please call us back."]
    assert sess.tracker.pending_marks == {}
    assert any(e["type"] == "error" and e["where"] == "send_audio" for e in events)
    assert not task.done()
    clk.t = sess.tracker.ends_at() + 1
    await asyncio.sleep(0.3)
    assert not task.done() and sess.tracker.is_playing()
    clk.t = sess.tracker.ends_at() + sess.cfg.mark_timeout_ms
    await asyncio.sleep(0.3)
    assert task.done() and task.exception() is None


async def test_session_arms_the_mark_name_twilio_writes_and_releases_on_that_echo():
    """The string on the wire is the string send_audio returns, and that is what the session arms.

    Echoing it while the clock is still before ends_at marks the chunk heard and lets say_and_wait return.
    """
    ws = FakeTwilioWS([])
    tr = TwilioTransport(ws, None)
    tr.stream_sid = "MZ9"
    returned = []

    async def send(pcm, rate):
        name = await tr.send_audio(pcm, rate)
        returned.append(name)
        return name

    clk = Clock()
    sess = Session(llm=FakeLLM([["x."]]), tts=FakeTTS(word_ms=200), vad=EnergyVAD(),
                   send_audio=send, clear_audio=tr.clear, clock=clk,
                   cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    sess.attach_transport_marks(tr)
    task = asyncio.create_task(sess.say_and_wait("Hello there friend."))
    for _ in range(400):
        await asyncio.sleep(0)
    wire = [m["mark"]["name"] for m in ws.sent if m["event"] == "mark"]
    assert returned == wire == ["m1"]
    assert set(sess.tracker.pending_marks) == {"m1"}
    assert clk.t < sess.tracker.ends_at() and not task.done()
    tr._on_mark(returned[0])
    assert sess.tracker.heard() == ("Hello there friend.", False)
    assert sess.tracker.buffered_ms() == 0 and not sess.tracker.is_playing()
    await asyncio.sleep(0.3)
    assert task.done() and task.exception() is None
    assert clk.t < sess.tracker.ends_at()


class FakeBrowserWS:
    def __init__(self, chunks):
        self.chunks, self.bytes_sent, self.texts = chunks, [], []

    async def iter_bytes(self):
        for c in self.chunks:
            yield c

    async def send_bytes(self, b):
        self.bytes_sent.append(b)

    async def send_text(self, s):
        self.texts.append(json.loads(s))


async def test_browser_reassembles_frames_across_odd_sized_messages():
    pcm = (np.arange(512 * 3) % 1000).astype("<i2")
    raw = pcm.tobytes()
    cuts = [0, 7, 1000, 1001, 1500, len(raw)]        # includes odd splits mid-sample
    chunks = [raw[a:b] for a, b in zip(cuts, cuts[1:])]
    sess = StubSession()

    async def factory(tr):
        return sess

    tr = BrowserTransport(FakeBrowserWS(chunks), factory)
    await tr.run()
    assert sess.closed and len(sess.frames) == 3
    assert np.array_equal(np.concatenate(sess.frames), pcm)


async def test_browser_outbound_audio_and_clear_event():
    ws = FakeBrowserWS([])
    tr = BrowserTransport(ws, None)
    await tr.send_audio(np.array([1, -2, 3], dtype=np.int16), 24000)
    await tr.clear()
    assert np.frombuffer(ws.bytes_sent[0], dtype="<i2").tolist() == [1, -2, 3]
    assert ws.texts == [{"type": "clear"}]


class FakePipeline:
    def __init__(self, pieces, fail=False):
        self.pieces, self.fail = pieces, fail

    def __call__(self, text, voice, speed):
        for p in self.pieces:
            yield ("g", "p", p)
        if self.fail:
            raise RuntimeError("tts boom")


async def test_kokoro_adapter_converts_float_to_int16_in_order():
    a = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    b = np.array([1.0, -1.0, 2.0], dtype=np.float32)       # 2.0 must clip, not wrap
    tts = KokoroTTS(pipeline=FakePipeline([a, b]))
    out = [x async for x in tts.synthesize("hi")]
    assert tts.sample_rate == 24000 and len(out) == 2
    assert out[0].dtype == np.int16 and out[0].tolist() == [0, 16384, -16384]
    assert out[1].tolist() == [32767, -32767, 32767]


async def test_kokoro_adapter_propagates_errors():
    tts = KokoroTTS(pipeline=FakePipeline([np.zeros(4, dtype=np.float32)], fail=True))
    with pytest.raises(RuntimeError, match="tts boom"):
        _ = [x async for x in tts.synthesize("hi")]
