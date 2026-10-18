"""Backchannels: short overlaps must not take the floor, and the labelled set is scorable."""
import asyncio

import numpy as np

from api.runtime import Runtime
from core.backchannel import BACKCHANNEL_MS, is_backchannel
from core.session import SessionConfig
from core.turn import Ev, TurnConfig, TurnManager
from harness.backchannel_set import score_set
from tests.unit.test_api import FakeLLM as ApiLLM, FakeTTS as ApiTTS, FakeVAD, make, tenant
from tests.unit.test_session import FakeLLM, FakeTTS, Rig

F = 32.0


def feed(tm, pattern):
    out = []
    for speech, n in pattern:
        for _ in range(n):
            for e in tm.process(0.9 if speech else 0.1):
                out.append((e.kind, e.t_ms))
    return out


def kinds(events):
    return [k for k, _ in events]


def test_classifier_accepts_the_fixed_set_only_when_short():
    for phrase in ("mm-hm", "mm", "mhm", "uh-huh", "uh huh", "yeah", "yep", "yes",
                   "okay", "ok", "right", "sure", "sí", "si", "oui", "haan"):
        assert is_backchannel(phrase, 450) is True
        assert is_backchannel(phrase, 451) is False
    assert is_backchannel("Yeah!", 200) is True
    assert is_backchannel("  OK. ", 100) is True
    assert is_backchannel("Sí", 300) is True
    assert is_backchannel("", 319) is True
    assert is_backchannel("   ", 0) is True
    assert is_backchannel("", 320) is False          # empty only under 320 ms
    assert is_backchannel("", 450) is False
    assert is_backchannel("yeah yeah", 200) is False
    assert is_backchannel("I need to move the appointment", 300) is False
    assert is_backchannel("yeah", 500, max_ms=700) is True


def test_turn_holds_speech_start_for_backchannel_ms_while_agent_speaks():
    tm = TurnManager()
    tm.set_agent_speaking(True)
    ev = feed(tm, [(True, 14), (False, 12)])                         # 448 ms, then silence
    assert kinds(ev) == [Ev.OVERLAP_CHECK] and ev[0][1] == 14 * F    # transcribed, not a turn yet
    fresh = TurnManager()
    fresh.set_agent_speaking(True)
    ev = feed(fresh, [(True, 16)])
    assert kinds(ev) == [Ev.SPEECH_START] and ev[0][1] == 14 * F     # 15th frame reaches 480 ms

    low = TurnManager(TurnConfig(backchannel_ms=BACKCHANNEL_MS["low"]))
    low.set_agent_speaking(True)
    ev = feed(low, [(True, 21), (False, 8)])                         # 672 ms < 700
    assert kinds(ev) == [Ev.OVERLAP_CHECK] and ev[0][1] == 21 * F
    # not speaking: backchannel_ms does not move the normal 96 ms start
    ev = feed(TurnManager(TurnConfig(backchannel_ms=700)), [(True, 4)])
    assert kinds(ev) == [Ev.SPEECH_START] and ev[0][1] == 2 * F

    high = TurnManager(TurnConfig(backchannel_ms=BACKCHANNEL_MS["high"]))
    high.set_agent_speaking(True)
    ev = feed(high, [(True, 7)])
    assert kinds(ev) == [Ev.SPEECH_START] and ev[0][1] == 5 * F      # unchanged 192 ms floor


def test_score_set_counts_five_labelled_rows():
    rows = [
        {"text": "yeah", "duration_ms": 300, "label": True},
        {"text": "yeah", "duration_ms": 800, "label": False},
        {"text": "I need a different day", "duration_ms": 400, "label": False},
        {"text": "", "duration_ms": 200, "label": True},
        {"text": "uh-huh", "duration_ms": 450, "label": True},
    ]
    assert score_set(rows) == {"correct": 5, "total": 5}
    rows[0]["label"] = False
    assert score_set(rows) == {"correct": 4, "total": 5}


def test_note_overlap_transcript_emits_backchannel_and_does_not_count():
    r = Rig(FakeLLM([["Hi."]]))
    r.s.stats["interruptions"] = 1
    r.s.history.append({"role": "assistant", "content": "Hi there [interrupted]"})
    r.s.note_overlap_transcript("yeah", 300)
    assert r.s.stats["interruptions"] == 1                       # not counted again, and not undone
    assert r.last("backchannel") == {"type": "backchannel", "t": 0.0, "text": "yeah", "duration_ms": 300}
    assert r.s.history[-1]["content"].endswith("[interrupted]")  # cannot un-interrupt
    r.s.note_overlap_transcript("um, what about Tuesday instead", 300)
    r.s.note_overlap_transcript("yeah", 900)
    assert [e["type"] for e in r.events] == ["backchannel"]


async def test_short_overlap_does_not_barge_and_a_yeah_after_a_real_cut_is_marked():
    llm = FakeLLM([["Hold on, I am still talking through this whole reply with you."]])

    async def transcribe(audio):
        return "yeah"

    llm.transcribe = transcribe
    r = Rig(llm, tts=FakeTTS(word_ms=300))
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.sent and len(llm.calls) == 1
    n_before = len(r.events)
    await r.frames("s", 12)                                      # 384 ms < 450
    await r.frames("q", 8)
    await r.settle()
    extra = r.types()[n_before:]
    assert r.clears == 0 and r.s.stats["interruptions"] == 0 and len(llm.calls) == 1
    assert "user_interrupted" not in extra and "user_speech_start" not in extra

    llm2 = FakeLLM([["Hello there, friend of mine today."]])

    async def transcribe(audio):
        llm2.n_transcribe += 1
        return "yeah"

    llm2.transcribe = transcribe
    r2 = Rig(llm2, tts=FakeTTS(word_ms=400),
             cfg=SessionConfig(tts_poll_s=0, turn=TurnConfig(backchannel_ms=192)))
    await r2.frames("s", 15)
    await r2.frames("q", 22)
    await r2.settle()
    await r2.frames("s", 6)                                      # high sensitivity still cuts at 192 ms
    await r2.settle()
    assert r2.clears == 1 and r2.s.stats["interruptions"] == 1
    assert all(m["role"] != "assistant" for m in r2.s.history)  # cut already applied; nothing heard to keep
    await r2.frames("q", 30)
    await r2.settle()
    ev = r2.last("backchannel")
    assert ev["text"] == "yeah" and ev["duration_ms"] == 192
    assert r2.clears == 1                                  # playback clear cannot be undone
    assert r2.s.stats["interruptions"] == 0                # a backchannel is not counted


async def test_short_real_request_during_playback_interrupts():
    """384 ms is under the 450 ms gate. "stop" still takes the floor; the reply is cleared."""
    llm = FakeLLM([
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Sure, I stopped."],
    ])

    async def transcribe(audio):
        return "stop"

    llm.transcribe = transcribe
    r = Rig(llm, tts=FakeTTS(word_ms=300))
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.sent and len(llm.calls) == 1
    await r.frames("s", 12)                                      # 384 ms < 450
    await r.frames("q", 10)
    await r.settle()
    assert r.clears >= 1 and r.s.stats["interruptions"] >= 1
    assert "user_interrupted" in r.types()
    assert any(m.get("role") == "user" and m.get("content") == "stop" for m in r.s.history)
    assert len(llm.calls) >= 2


async def test_yeah_after_a_dropped_cough_is_not_glued_into_a_barge():
    """A cough under the 192 ms floor is not a turn. It must not stay in the buffer and be
    concatenated, with the silence removed, onto the following yeah."""
    llm = FakeLLM([["Hold on, I am still talking through this whole reply with you right now."]])

    async def transcribe(audio):
        if len(audio) and bool((audio == 9000).any()):
            return "stop yeah"
        return "yeah"

    llm.transcribe = transcribe
    r = Rig(llm, tts=FakeTTS(word_ms=300))
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    cough = np.full(512, 9000, np.int16)
    for _ in range(4):                                          # 128 ms < 192 ms barge floor
        await r.s.feed(cough)
        await r.settle(5)
        r.clk.t += 32
    await r.frames("q", 6)
    await r.frames("s", 12)                                     # 384 ms yeah
    await r.frames("q", 8)
    await r.settle()
    assert r.clears == 0 and r.s.stats["interruptions"] == 0 and len(llm.calls) == 1


async def test_barge_longer_than_preroll_keeps_the_start_of_the_utterance():
    """The backchannel gate holds SPEECH_START until 480 ms, past the 320 ms pre-roll.
    The model has to hear the start of that barge, not only the last 320 ms."""
    llm = FakeLLM([
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Okay, go on."],
    ])
    audios = []
    orig = llm.stream

    async def stream(history, audio, followups, tools):
        audios.append(audio.copy())
        async for item in orig(history, audio, followups, tools):
            yield item

    llm.stream = stream
    r = Rig(llm, tts=FakeTTS(word_ms=300))
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    marker = np.full(512, 7000, np.int16)
    rest = np.full(512, 5000, np.int16)
    for i in range(16):                                         # 512 ms, opens a real turn
        await r.s.feed(marker if i == 0 else rest)
        await r.settle(5)
        r.clk.t += 32
    await r.frames("q", 10)
    await r.settle()
    assert len(audios) >= 2 and bool((audios[-1] == 7000).any())


async def test_stop_overlap_waits_for_the_transcript_already_in_flight():
    """Classifying "stop" commits a turn at once. The model request must still carry the
    previous caller transcript, not the "[caller audio]" placeholder."""
    llm = FakeLLM([
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Sure, I stopped."],
    ])
    gate = asyncio.Event()
    n = {"n": 0}

    async def transcribe(audio):
        n["n"] += 1
        if n["n"] == 1:
            await gate.wait()
            return "move my appointment"
        return "stop"

    llm.transcribe = transcribe
    r = Rig(llm, tts=FakeTTS(word_ms=300))
    try:
        await r.frames("s", 15)
        await r.frames("q", 22)
        await r.settle()
        assert r.sent and len(llm.calls) == 1
        await r.frames("s", 12)
        await r.frames("q", 8)
        await r.settle()
        gate.set()
        await r.settle()
        assert len(llm.calls) >= 2
        hist = llm.calls[1]["history"]
        assert any(m.get("role") == "user" and m.get("content") == "move my appointment" for m in hist)
        assert all(m.get("content") != "[caller audio]" for m in hist)
    finally:
        gate.set()


def test_agent_barge_in_is_validated_and_copied_onto_the_turn():
    c, store, _ = make()
    k, tid = tenant(c)
    h = {"X-API-Key": k}
    assert c.post("/v1/agents", json={"name": "a", "bargeIn": "loud"}, headers=h).status_code == 422
    a = c.post("/v1/agents", json={"name": "a", "bargeIn": "low"}, headers=h)
    assert a.status_code == 200 and a.json()["config"]["bargeIn"] == "low"
    aid = a.json()["id"]
    assert c.patch(f"/v1/agents/{aid}", json={"bargeIn": "sometimes"}, headers=h).status_code == 422
    patched = c.patch(f"/v1/agents/{aid}", json={"bargeIn": "high"}, headers=h)
    assert patched.status_code == 200 and patched.json()["config"]["bargeIn"] == "high"

    class T:
        out_rate = 8000

        async def send_audio(self, b): ...
        async def clear(self): ...
        async def end(self): ...

    rt = Runtime(ApiLLM(), lambda v: ApiTTS(), FakeVAD, store)
    call = store.create_call(tid, aid, "webSocket", store.get_agent(tid, aid)["config"], {})
    assert rt.build_session(call, T()).cfg.turn.backchannel_ms == BACKCHANNEL_MS["high"]
    low = store.create_call(tid, None, "webSocket", {"systemPrompt": "x", "bargeIn": "low"}, {})
    assert rt.build_session(low, T()).cfg.turn.backchannel_ms == BACKCHANNEL_MS["low"]
    plain = store.create_call(tid, None, "webSocket", {"systemPrompt": "x"}, {})
    assert rt.build_session(plain, T()).cfg.turn.backchannel_ms == BACKCHANNEL_MS["normal"]
