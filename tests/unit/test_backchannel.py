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
    assert is_backchannel("no", 200) is False        # a short refusal still takes the floor
    assert is_backchannel("no", 450) is False
    assert is_backchannel("stop", 200) is False
    assert is_backchannel("stop", 450) is False
    assert is_backchannel("yeah yeah", 200) is False
    assert is_backchannel("I need to move the appointment", 300) is False
    assert is_backchannel("yeah", 500, max_ms=700) is True


def test_turn_holds_speech_start_for_backchannel_ms_while_agent_speaks():
    tm = TurnManager()
    tm.set_agent_speaking(True)
    ev = feed(tm, [(True, 14), (False, 12)])                         # 448 ms, then silence
    assert kinds(ev) == [Ev.OVERLAP_CHECK] and ev[0][1] == (14 + 5) * F  # after speculate_ms
    fresh = TurnManager()
    fresh.set_agent_speaking(True)
    ev = feed(fresh, [(True, 16)])
    assert kinds(ev) == [Ev.SPEECH_START] and ev[0][1] == 14 * F     # 15th frame reaches 480 ms

    low = TurnManager(TurnConfig(backchannel_ms=BACKCHANNEL_MS["low"]))
    low.set_agent_speaking(True)
    ev = feed(low, [(True, 21), (False, 8)])                         # 672 ms < 700
    assert kinds(ev) == [Ev.OVERLAP_CHECK] and ev[0][1] == (21 + 5) * F
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
    """384 ms is under the 450 ms gate. "stop" and "no" still take the floor."""
    for word in ("stop", "no"):
        llm = FakeLLM([
            ["Hold on, I am still talking through this whole reply with you right now."],
            ["Sure, I stopped."],
        ])

        async def transcribe(audio, word=word):
            return word

        llm.transcribe = transcribe
        r = Rig(llm, tts=FakeTTS(word_ms=300))
        await r.frames("s", 15)
        await r.frames("q", 22)
        await r.settle()
        assert r.sent and len(llm.calls) == 1
        await r.frames("s", 12)                                  # 384 ms < 450
        await r.frames("q", 10)
        await r.settle()
        assert r.clears >= 1 and r.s.stats["interruptions"] >= 1
        assert "user_interrupted" in r.types()
        assert any(m.get("role") == "user" and m.get("content") == word for m in r.s.history)
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


async def _agent_playing(transcribe, replies=None):
    llm = FakeLLM(replies or [
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Okay, go on."],
    ])
    llm.transcribe = transcribe
    r = Rig(llm, tts=FakeTTS(word_ms=1000))
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.sent and len(llm.calls) == 1
    return r


async def _feed_value(r, frame, n):
    for _ in range(n):
        await r.s.feed(frame)
        await r.settle(5)
        r.clk.t += 32


async def test_vad_dip_during_hold_keeps_the_rest_of_the_phrase():
    """13 continuous frames of "yeah I need" clear playback. The same audio with one
    silent frame between "yeah" and "I need" must do the same: the tail is under the
    192 ms floor, so splitting on the dip drops it."""
    prefix = np.full(512, 8000, np.int16)
    tail = np.full(512, 3000, np.int16)

    async def transcribe(audio):
        if len(audio) == 0:
            return ""
        has_prefix = bool((audio == 8000).any())
        has_tail = bool((audio == 3000).any())
        if has_prefix and has_tail:
            return "yeah I need"
        if has_prefix:
            return "yeah"
        if has_tail:
            return "I need"
        return "hello"

    async def run(split):
        r = await _agent_playing(transcribe)
        if split:
            await _feed_value(r, prefix, 8)
            await r.frames("q", 1)
            await _feed_value(r, tail, 5)
        else:
            await _feed_value(r, prefix, 8)
            await _feed_value(r, tail, 5)
        await r.frames("q", 8)
        await r.settle()
        assert r.clears >= 1
        assert any(m.get("role") == "user" and m.get("content") == "yeah I need"
                   for m in r.s.history)

    await run(False)
    await run(True)


async def test_stop_becomes_a_turn_when_playback_ends_before_the_pause():
    """384 ms of "stop" while the agent is playing, then playback ends, then silence.
    The pause must open a user turn. It must not be deleted with the overlap buffer."""
    n = {"n": 0}

    async def transcribe(audio):
        n["n"] += 1
        return "hello" if n["n"] == 1 else "no"

    r = await _agent_playing(transcribe, [
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Sure, I stopped."],
    ])
    await r.frames("s", 12)                                      # 384 ms, gate is 450
    r.s._gen.produce_done = True
    r.s.tracker.reset()
    assert r.s._agent_active() is False
    await r.frames("q", 25)
    await r.settle()
    assert any(m.get("role") == "user" and m.get("content") == "no" for m in r.s.history)
    assert len(r.llm.calls) >= 2


async def test_barge_onset_survives_playback_ending():
    """More than the 320 ms pre-roll is already buffered when playback stops. The first
    frame has fallen out of _pre and must still reach the model."""

    async def transcribe(audio):
        return "hello"

    r = await _agent_playing(transcribe)
    audios = []
    orig = r.llm.stream

    async def stream(history, audio, followups, tools):
        audios.append(np.array(audio, copy=True))
        async for item in orig(history, audio, followups, tools):
            yield item

    r.llm.stream = stream
    marker = np.full(512, 7000, np.int16)
    rest = np.full(512, 5000, np.int16)
    await _feed_value(r, marker, 1)
    await _feed_value(r, rest, 11)                               # 384 ms total, past pre-roll
    r.s._gen.produce_done = True
    r.s.tracker.reset()
    await _feed_value(r, rest, 6)
    await r.frames("q", 25)
    await r.settle()
    assert audios and bool((audios[-1] == 7000).any())


async def test_inflight_stop_is_kept_when_a_later_yeah_finishes_first():
    """A later short "yeah" must not cancel the overlap check that is still transcribing
    "stop". When that transcript arrives it still clears playback and is the user turn."""
    gate = asyncio.Event()
    stop = np.full(512, 7000, np.int16)
    yeah = np.full(512, 4000, np.int16)

    async def transcribe(audio):
        if len(audio) and bool((audio == 7000).any()) and not bool((audio == 4000).any()):
            await gate.wait()
            return "stop"
        if len(audio) and bool((audio == 4000).any()) and not bool((audio == 7000).any()):
            return "yeah"
        return "hello"

    r = await _agent_playing(transcribe, [
        ["Hold on, I am still talking through this whole reply with you right now."],
        ["Sure, I stopped."],
    ])
    try:
        await _feed_value(r, stop, 12)
        await r.frames("q", 8)
        await r.settle()
        assert r.clears == 0
        await _feed_value(r, yeah, 12)
        await r.frames("q", 8)
        await r.settle()
        assert r.clears == 0
        assert not any(m.get("content") == "stop" for m in r.s.history)
        gate.set()
        await r.settle()
        assert r.clears >= 1
        assert any(m.get("role") == "user" and m.get("content") == "stop" for m in r.s.history)
    finally:
        gate.set()


async def test_real_turn_keeps_audio_from_an_inflight_overlap_check():
    """Speech that resumes into a real turn must still contain the samples already handed
    to an in-flight overlap check. Dropping that result must not drop the onset."""
    gate = asyncio.Event()
    audios = []

    async def transcribe(audio):
        if len(audio) and bool((audio == 7000).any()) and not bool((audio == 4000).any()):
            await gate.wait()
            return "stop"
        return "hello"

    r = await _agent_playing(transcribe)
    orig = r.llm.stream

    async def stream(history, audio, followups, tools):
        audios.append(np.array(audio, copy=True))
        async for item in orig(history, audio, followups, tools):
            yield item

    r.llm.stream = stream
    try:
        await _feed_value(r, np.full(512, 7000, np.int16), 12)
        await r.frames("q", 8)
        await r.settle()
        await _feed_value(r, np.full(512, 4000, np.int16), 16)   # opens a real turn at 480 ms
        await r.frames("q", 10)
        await r.settle()
        assert any(len(a) and bool((a == 7000).any()) for a in audios)
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
