import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from core.session import Session, SessionConfig, ToolCall
from core.vad_exec import VADScheduler

SPEECH = (np.ones(512) * 5000).astype(np.int16)
SILENCE = np.zeros(512, dtype=np.int16)
HANG = "<hang>"


class EnergyVAD:
    def prob(self, frame):
        return 0.9 if np.abs(frame).max() > 1000 else 0.1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeTTS:
    sample_rate = 24000

    def __init__(self, word_ms=80):
        self.word_ms = word_ms
        self.texts = []

    async def synthesize(self, text):
        self.texts.append(text)
        await asyncio.sleep(0)
        n = len(text.split()) * self.word_ms * 24
        yield np.zeros(n, dtype=np.int16)


class FakeLLM:
    """replies: one list per stream() call; items are text deltas, ToolCall, or HANG."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.cancelled = 0
        self.n_transcribe = 0

    async def stream(self, history, audio, followups, tools):
        i = len(self.calls)
        self.calls.append({"history": copy.deepcopy(history), "followups": copy.deepcopy(followups),
                           "n_audio": len(audio)})
        done = False
        try:
            for item in self.replies[min(i, len(self.replies) - 1)]:
                await asyncio.sleep(0)
                if item == HANG:
                    await asyncio.Event().wait()
                yield item
            done = True
        finally:
            if not done:
                self.cancelled += 1

    async def transcribe(self, audio):
        self.n_transcribe += 1
        return f"caller said {self.n_transcribe}"


class Rig:
    def __init__(self, llm, tts=None, tools=None, cfg=None):
        self.clk = Clock()
        self.sent, self.clears, self.events = [], 0, []
        self.llm, self.tts = llm, tts or FakeTTS()

        async def send(pcm, rate):
            self.sent.append((self.clk.t, len(pcm), rate))

        async def clear():
            self.clears += 1

        self.s = Session(llm=llm, tts=self.tts, vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                         clock=self.clk, on_event=self.events.append, tools=tools,
                         cfg=cfg or SessionConfig(tts_poll_s=0))

    async def settle(self, n=300):
        for _ in range(n):
            await asyncio.sleep(0)

    async def frames(self, kind, n):
        for _ in range(n):
            await self.s.feed(SPEECH if kind == "s" else SILENCE)
            await self.settle(5)
            self.clk.t += 32

    def types(self):
        return [e["type"] for e in self.events]

    def last(self, type_):
        return [e for e in self.events if e["type"] == type_][-1]


async def test_basic_turn_holds_audio_until_end_of_turn():
    llm = FakeLLM([["Sure, ", "I ", "can ", "help ", "you ", "with ", "that. ", "What ", "do ", "you ", "need?"]])
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)            # 256 ms silence: speculative reply starts, must stay silent
    await r.settle()
    assert "speculate_start" in r.types() and len(llm.calls) == 1
    assert r.sent == []
    await r.frames("q", 15)           # reaches 640 ms: turn ends, reply plays
    await r.settle()
    assert [n for _, n, _ in r.sent] == [4480, 2560]   # 560 ms + 320 ms at 8 kHz
    assert all(rate == 8000 for _, _, rate in r.sent)
    assert r.tts.texts == ["Sure, I can help you with that.", "What do you need?"]
    assert r.last("agent_first_audio")["latency_ms"] == 640
    assert r.s.history[1:] == [
        {"role": "user", "content": "caller said 1"},
        {"role": "assistant", "content": "Sure, I can help you with that. What do you need?"}]


async def test_resume_cancels_speculation_and_only_final_reply_is_spoken():
    llm = FakeLLM([["Hello ", HANG], ["Final ", "answer ", "here."]])
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)            # speculate
    await r.settle()
    await r.frames("s", 10)           # caller continues: cancel speculation
    await r.settle()
    assert llm.cancelled == 1 and r.sent == []
    await r.frames("q", 30)
    await r.settle()
    assert r.s.stats["speculations"] == 2 and r.s.stats["speculations_cancelled"] == 1
    assert r.tts.texts == ["Final answer here."]
    assert r.s.history[-1] == {"role": "assistant", "content": "Final answer here."}
    assert len([e for e in r.events if e["type"] == "user_transcript"]) == 1


async def test_barge_in_clears_audio_and_trims_history_to_what_was_heard():
    reply = ["One ", "two ", "three ", "four ", "five. ", "Six ", "seven ", "eight ", "nine ", "ten. ",
             "Eleven ", "twelve ", "thirteen ", "fourteen ", "fifteen."]
    llm = FakeLLM([reply, ["Okay."]])
    r = Rig(llm, tts=FakeTTS(word_ms=200))    # 1000 ms per sentence, 3000 ms total
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert len(r.sent) == 2            # 2 s queued > 1.5 s lookahead: sentence 3 not yet synthesized
    await r.frames("q", 45)           # 1440 ms of playback elapse
    await r.settle()
    assert len(r.sent) == 3            # synthesized just in time, before the listener runs out
    sent_before = len(r.sent)
    await r.frames("s", 16)           # 512 ms: backchannel gate is 450 ms, so this still interrupts
    await r.settle()
    assert r.clears == 1
    ev = r.last("user_interrupted")
    assert ev["cut"] is True
    # the longer gate lets one more word play than the old 192 ms barge threshold
    assert ev["heard"] == "One two three four five. Six seven eight nine"
    assert r.s.history[-1] == {"role": "assistant",
                               "content": "One two three four five. Six seven eight nine [interrupted]"}
    assert len(r.sent) == sent_before                 # nothing more is sent after the interruption
    await r.frames("s", 10)
    await r.frames("q", 30)
    await r.settle()
    # the next model call sees the trimmed assistant turn, not the full planned reply
    second = llm.calls[1]["history"]
    assert second[-1]["content"].endswith("[interrupted]")
    assert second[1]["content"] == "caller said 1"


async def test_no_barge_in_after_agent_finished_playing():
    llm = FakeLLM([["Short ", "reply."], ["Again."]])
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    await r.frames("q", 40)           # far longer than the 160 ms reply
    await r.frames("s", 15)
    await r.frames("q", 30)
    await r.settle()
    assert r.clears == 0 and r.s.stats["interruptions"] == 0
    assert r.tts.texts == ["Short reply.", "Again."]


async def test_tool_call_does_not_run_until_turn_is_committed():
    calls = []

    async def lookup(args):
        calls.append(args)
        return {"slot": "10:00"}

    llm = FakeLLM([["Let ", "me ", "check. ", ToolCall("t1", "lookup", {"day": "mon"})],
                   ["Booked ", "for ", "ten."]])
    r = Rig(llm, tools={"lookup": lookup})
    await r.frames("s", 15)
    await r.frames("q", 8)            # speculative reply reaches the tool call
    await r.settle()
    assert calls == []                # side effect must wait for commit
    await r.frames("q", 15)
    await r.settle()
    assert calls == [{"day": "mon"}]
    followups = llm.calls[1]["followups"]
    assert followups[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert followups[1] == {"role": "tool", "tool_call_id": "t1", "content": '{"slot": "10:00"}'}
    assert r.tts.texts == ["Let me check.", "Booked for ten."]


async def test_slow_tool_speaks_its_filler_before_returning():
    release = asyncio.Event()
    seen = {}

    async def lookup(args):
        await release.wait()
        seen["texts"] = list(r.tts.texts)
        return {"slot": "10:00"}

    llm = FakeLLM([["Let me check. ", ToolCall("t1", "lookup", {"day": "mon"})],
                   ["Booked for ten."]])
    phrase = "Checking that order."
    r = Rig(llm, tools={"lookup": lookup},
            cfg=SessionConfig(tts_poll_s=0, filler_delay_s=0.01, tool_fillers={"lookup": phrase}))
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        for _ in range(100):
            if phrase in r.tts.texts:
                break
            await asyncio.sleep(0.01)
        await r.settle()
        assert phrase in r.tts.texts
        assert "One moment, let me check that." not in r.tts.texts
    finally:
        release.set()
    await r.settle()
    assert phrase in seen["texts"]          # synthesized while the tool was still blocked
    assert "Booked for ten." in r.tts.texts


async def test_blank_per_tool_filler_does_not_speak_the_default():
    """fillerText "" is a disable, not a missing value. The default line must not play."""
    from api.tools import filler_texts

    release = asyncio.Event()

    async def lookup(args):
        await release.wait()
        return {"slot": "10:00"}

    fillers = filler_texts([{
        "name": "lookup", "description": "Find a slot.",
        "parameters": {"type": "object", "properties": {}},
        "http": {"url": "https://api.example.com/slots"},
        "fillerText": "   ",
    }])
    llm = FakeLLM([["Let me check. ", ToolCall("t1", "lookup", {"day": "mon"})],
                   ["Booked for ten."]])
    r = Rig(llm, tools={"lookup": lookup},
            cfg=SessionConfig(tts_poll_s=0, filler_delay_s=0.01, tool_fillers=fillers))
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        await asyncio.sleep(0.05)           # past filler_delay_s while the tool is still blocked
        assert fillers == {"lookup": ""}
        assert "One moment, let me check that." not in r.tts.texts
    finally:
        release.set()
    await r.settle()
    assert "One moment, let me check that." not in r.tts.texts
    assert "Booked for ten." in r.tts.texts


async def test_fast_tool_does_not_speak_a_filler():
    async def lookup(args):
        return {"slot": "10:00"}

    llm = FakeLLM([["Let me check. ", ToolCall("t1", "lookup", {"day": "mon"})],
                   ["Booked for ten."]])
    r = Rig(llm, tools={"lookup": lookup}, cfg=SessionConfig(tts_poll_s=0, filler_delay_s=0.01))
    await r.frames("s", 15)
    await r.frames("q", 25)
    await r.settle()
    await asyncio.sleep(0.05)               # longer than filler_delay_s; the sleeper was cancelled
    assert "One moment, let me check that." not in r.tts.texts
    assert r.tts.texts == ["Let me check.", "Booked for ten."]


def _blocked_lookup(release):
    async def lookup(args):
        await release.wait()
        return {"slot": "10:00"}

    return lookup


def _two_sentence_tool_turn(release, phrase):
    """First sentence is 40 words (3.2s) so it stays above the 1.5s lookahead while the tool blocks."""
    first = " ".join(["alpha"] * 40) + "."
    second = "Beta sentence follows."
    llm = FakeLLM([[f"{first} ", f"{second} ", ToolCall("t1", "lookup", {"day": "mon"})],
                   ["All done now."]])
    cfg = SessionConfig(tts_poll_s=0, tts_lookahead_ms=1500, filler_delay_s=0.01,
                        tool_fillers={"lookup": phrase})
    return first, second, Rig(llm, tools={"lookup": _blocked_lookup(release)}, cfg=cfg)


async def test_filler_does_not_speak_ahead_of_a_queued_sentence():
    """The filler must not be synthesized while a pre-tool sentence is still unspoken, and a tool
    that returns before that sentence is released must not leave the filler in tts.texts."""
    release = asyncio.Event()
    phrase = "FILLER"
    first, second, r = _two_sentence_tool_turn(release, phrase)
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        await asyncio.sleep(0.05)          # past filler_delay_s; sentence 2 is held for lookahead
        assert r.tts.texts == [first]      # not [first, FILLER] while Beta is still unspoken
        assert second not in r.tts.texts and phrase not in r.tts.texts
    finally:
        release.set()                      # tool returns before the filler is allowed to start
    r.clk.t += 10_000
    for _ in range(20):
        await asyncio.sleep(0.03)
    await r.settle()
    assert phrase not in r.tts.texts
    assert r.tts.texts == [first, second, "All done now."]


async def test_tool_only_turn_still_speaks_a_filler():
    """No pre-tool sentence: the queue is empty and nothing is held, so the filler still plays."""
    release = asyncio.Event()
    llm = FakeLLM([[ToolCall("t1", "lookup", {})], ["Done."]])
    r = Rig(llm, tools={"lookup": _blocked_lookup(release)},
            cfg=SessionConfig(tts_poll_s=0, filler_delay_s=0.01))
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        for _ in range(100):
            if r.tts.texts:
                break
            await asyncio.sleep(0.01)
        assert r.tts.texts == ["One moment, let me check that."]
    finally:
        release.set()
    await r.settle()
    assert r.tts.texts == ["One moment, let me check that.", "Done."]


async def test_filler_follows_queued_sentences_when_the_tool_is_still_running():
    release = asyncio.Event()
    phrase = "FILLER"
    first, second, r = _two_sentence_tool_turn(release, phrase)
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        await asyncio.sleep(0.05)          # delay already elapsed, but Beta is not spoken yet
        assert phrase not in r.tts.texts
        r.clk.t += 10_000                  # playback catches up; the queued sentence may be synthesized
        for _ in range(50):
            if phrase in r.tts.texts and second in r.tts.texts:
                break
            await asyncio.sleep(0.02)
        assert r.tts.texts == [first, second, phrase]
    finally:
        release.set()
    await r.settle()


async def test_slow_first_sentence_does_not_let_the_filler_cut_in():
    """Filler delay expires during the first synthesize. It must not take speak_lock ahead of
    the next sentence already sitting on textq."""
    release = asyncio.Event()
    phrase = "FILLER"

    class SlowFirstTTS(FakeTTS):
        async def synthesize(self, text):
            self.texts.append(text)
            await asyncio.sleep(0.08 if len(self.texts) == 1 else 0)
            n = len(text.split()) * self.word_ms * 24
            yield np.zeros(n, dtype=np.int16)

    llm = FakeLLM([["Alpha sentence here. ", "Beta sentence follows. ",
                    ToolCall("t1", "lookup", {"day": "mon"})],
                   ["All done now."]])
    r = Rig(llm, tts=SlowFirstTTS(), tools={"lookup": _blocked_lookup(release)},
            cfg=SessionConfig(tts_poll_s=0, filler_delay_s=0.01, tool_fillers={"lookup": phrase}))
    await r.frames("s", 15)
    await r.frames("q", 25)
    try:
        for _ in range(50):
            if len(r.tts.texts) >= 3:
                break
            await asyncio.sleep(0.02)
        assert r.tts.texts[:3] == ["Alpha sentence here.", "Beta sentence follows.", phrase]
    finally:
        release.set()
    await r.settle()


async def test_tool_call_cancelled_speculation_never_runs_tool():
    calls = []

    async def lookup(args):
        calls.append(args)
        return {}

    llm = FakeLLM([["Checking. ", ToolCall("t1", "lookup", {})], ["Plain ", "answer."]])
    r = Rig(llm, tools={"lookup": lookup})
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await r.frames("s", 10)           # resume: speculation (with pending tool call) is cancelled
    await r.frames("q", 30)
    await r.settle()
    assert calls == []
    # "Checking." was pre-synthesised during speculation (by design) but never sent;
    # only the committed reply reaches the caller: "Plain answer." = 160 ms = 1280 samples at 8 kHz
    assert [n for _, n, _ in r.sent] == [1280]
    assert r.s.history[-1] == {"role": "assistant", "content": "Plain answer."}


async def test_model_error_after_commit_speaks_an_apology():
    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Partial answer. "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)             # speculation starts; the model is held before it fails
    await r.settle()
    await r.frames("q", 15)            # turn commits while the model is still in flight
    await r.settle()
    llm.release.set()
    await r.settle()
    assert "Partial answer." in r.tts.texts
    assert r.tts.texts.count("Sorry, could you say that again?") == 1
    assert any(e["type"] == "error" and e["where"] == "produce" for e in r.events)


async def test_model_error_after_commit_keeps_a_finished_sentence_then_apologizes():
    """A sentence that ends on .!? is held until another character arrives. The success
    path flushes it; a model error must too, then apologize once, not instead."""

    class LLM:
        def __init__(self, sentence):
            self.sentence = sentence
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield self.sentence
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM("You are booked.")
    r = Rig(llm)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert r.tts.texts == []          # no trailing character yet, so nothing is spoken
    await r.frames("q", 15)           # commit while the model is still held
    await r.settle()
    llm.release.set()
    await r.settle()
    assert r.tts.texts == ["You are booked.", phrase]
    assert r.tts.texts.count(phrase) == 1
    assert r.s.history[-1] == {"role": "assistant", "content": f"You are booked. {phrase}"}


async def test_model_error_after_commit_drops_an_unfinished_fragment():
    """The error path must not flush a mid-sentence tail the way a clean finish would."""

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "on Tuesday"
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 25)
    await r.settle()
    llm.release.set()
    await r.settle()
    assert r.tts.texts == [phrase]
    assert "on Tuesday" not in " ".join(r.tts.texts)
    assert r.s.history[-1] == {"role": "assistant", "content": phrase}


async def _speak_tail_then_apology(sentence: str):
    """Commit, then raise, with `sentence` still held by the chunker (no trailing character)."""

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield sentence
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert r.tts.texts == []
    await r.frames("q", 15)
    await r.settle()
    llm.release.set()
    await r.settle()
    return r


async def test_model_error_keeps_a_question_then_one_apology():
    r = await _speak_tail_then_apology("Do you agree?")
    phrase = r.s.cfg.error_phrase
    assert r.tts.texts == ["Do you agree?", phrase]
    assert r.tts.texts.count(phrase) == 1
    assert r.s.history[-1] == {"role": "assistant", "content": f"Do you agree? {phrase}"}


async def test_model_error_keeps_a_newline_sentence_then_one_apology():
    r = await _speak_tail_then_apology("Booked for Tuesday\n")
    phrase = r.s.cfg.error_phrase
    assert r.tts.texts == ["Booked for Tuesday", phrase]
    assert r.tts.texts.count(phrase) == 1
    assert r.s.history[-1] == {"role": "assistant", "content": f"Booked for Tuesday {phrase}"}


async def test_committed_failure_speaks_the_configured_error_phrase():
    """The synth path must speak SessionConfig.error_phrase, not a hardcoded English sentence."""
    from api.runtime import ERROR_PHRASE

    phrase = ERROR_PHRASE["es"]
    assert phrase == "Perdón, ¿puede repetirlo?"

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Partial answer. "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm, cfg=SessionConfig(tts_poll_s=0, error_phrase=phrase))
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await r.frames("q", 15)
    await r.settle()
    llm.release.set()
    await r.settle()
    assert r.tts.texts.count(phrase) == 1
    assert "Partial answer." in r.tts.texts
    assert "Sorry, could you say that again?" not in r.tts.texts
    assert r.s.history[-1]["content"].endswith(phrase)


async def test_tts_failure_speaks_the_configured_error_phrase():
    """A committed TTS failure speaks via _speak_fallback, which must use the configured phrase."""
    from api.runtime import ERROR_PHRASE

    phrase = ERROR_PHRASE["es"]

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.started = asyncio.Event()
            self.fail = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            if text != phrase:
                self.started.set()
                await self.fail.wait()
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    tts = TTS()
    r = Rig(FakeLLM([["Hello there."]]), tts=tts, cfg=SessionConfig(tts_poll_s=0, error_phrase=phrase))
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await asyncio.wait_for(tts.started.wait(), timeout=2)
    await r.frames("q", 15)            # commit while the reply synthesize is still blocked
    await r.settle()
    tts.fail.set()
    await r.settle()
    assert tts.texts == ["Hello there.", phrase]
    assert "Sorry, could you say that again?" not in tts.texts
    assert len(r.sent) == 1


async def test_queued_apology_is_spoken_when_earlier_tts_fails():
    """The model path queues the apology and marks it used. A TTS error on the partial
    must not drop that phrase, and must not speak it twice."""
    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Partial answer. "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.partial_started = asyncio.Event()
            self.fail_partial = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            if text != "Sorry, could you say that again?":
                self.partial_started.set()
                await self.fail_partial.wait()
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    llm, tts = LLM(), TTS()
    r = Rig(llm, tts=tts)
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await asyncio.wait_for(tts.partial_started.wait(), timeout=2)
    await r.frames("q", 15)            # commit while both the model and the partial TTS are blocked
    await r.settle()
    llm.release.set()
    await r.settle()                   # model fails and queues the apology before the partial TTS raises
    tts.fail_partial.set()
    await r.settle()
    assert r.tts.texts.count("Sorry, could you say that again?") == 1
    assert "Partial answer." in r.tts.texts
    assert any(e["type"] == "error" and e["where"] == "produce" for e in r.events)


async def test_tts_failure_speaks_one_fallback_and_does_not_loop():
    class BoomTTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.fail = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            await self.fail.wait()
            raise RuntimeError("tts down")
            yield np.zeros(1, dtype=np.int16)  # pragma: no cover  (keeps this an async generator)

    tts = BoomTTS()
    r = Rig(FakeLLM([["Hello there."]]), tts=tts)
    await r.frames("s", 15)
    await r.frames("q", 25)            # commit while synthesis is still blocked
    await r.settle()
    tts.fail.set()
    await r.settle()
    assert r.tts.texts == ["Hello there.", "Sorry, could you say that again?"]
    assert any(e["type"] == "error" and e["where"] == "tts" for e in r.events)
    assert any(e["type"] == "error" and e["where"] == "tts_fallback" for e in r.events)


async def test_cancelled_speculation_does_not_speak_an_apology():
    llm = FakeLLM([["Hello ", HANG], ["Fine."]])
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await r.frames("s", 10)            # caller resumes: the speculative reply is cancelled
    await r.settle()
    assert "Sorry, could you say that again?" not in r.tts.texts


async def test_model_error_before_commit_stays_silent():
    """A speculative stream that fails before END_OF_TURN must not apologize once the turn commits."""

    class LLM:
        async def stream(self, history, audio, followups, tools):
            raise RuntimeError("model down")
            yield ""  # pragma: no cover

        async def transcribe(self, audio):
            return "hello"

    r = Rig(LLM())
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)             # speculation starts and the model fails before commit
    await r.settle()
    assert any(e["type"] == "error" and e["where"] == "produce" for e in r.events)
    assert phrase not in r.tts.texts and r.sent == []
    await r.frames("q", 15)            # silence reaches END_OF_TURN
    await r.settle()
    assert phrase not in r.tts.texts and r.sent == []
    assert r.s.history[-1] == {"role": "assistant", "content": ""}


async def test_model_error_after_speculative_tts_failure_still_speaks():
    """TTS dying during speculation used to end synth and drain, so a model error after
    commit queued an apology nobody read."""

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Hello there. "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            if text != "Sorry, could you say that again?":
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    llm, tts = LLM(), TTS()
    r = Rig(llm, tts=tts)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert tts.texts == ["Hello there."]
    assert r.sent == []
    assert "user_end_of_turn" not in r.types()
    await r.frames("q", 15)            # commit; the model has not failed yet
    await r.settle()
    assert tts.texts == ["Hello there."] and r.sent == []
    llm.release.set()
    await r.settle()
    assert tts.texts == ["Hello there.", phrase]
    assert len(r.sent) == 1
    assert any(e["type"] == "error" and e["where"] == "produce" for e in r.events)
    assert any(e["type"] == "error" and e["where"] == "tts" for e in r.events)


async def test_precommit_tts_failure_of_a_successful_reply_stays_silent():
    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            raise RuntimeError("tts down")
            yield np.zeros(1, dtype=np.int16)  # pragma: no cover

    tts = TTS()
    r = Rig(FakeLLM([["Hello there. More words here."]]), tts=tts)
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await r.frames("q", 15)
    await r.settle()
    assert tts.texts == ["Hello there."]
    assert r.s.cfg.error_phrase not in tts.texts
    assert r.sent == []


async def test_speculative_tts_failure_cancelled_by_resume_stays_silent():
    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            raise RuntimeError("tts down")
            yield np.zeros(1, dtype=np.int16)  # pragma: no cover

    tts = TTS()
    r = Rig(FakeLLM([["Hello there. ", HANG]]), tts=tts)
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert tts.texts == ["Hello there."]
    await r.frames("s", 10)            # resume cancels the uncommitted reply
    await r.settle()
    assert r.s.cfg.error_phrase not in tts.texts
    assert r.sent == []


async def test_model_error_does_not_repeat_its_own_closing_apology():
    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Sorry, could you say that again? "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await r.frames("q", 15)            # commit and play the model's own sentence
    await r.settle()
    assert r.tts.texts.count(phrase) == 1
    assert r.s.history[-1]["content"] == phrase
    llm.release.set()
    await r.settle()
    assert r.tts.texts.count(phrase) == 1
    assert r.s.history[-1]["content"] == phrase
    assert [e["text"] for e in r.events if e["type"] == "agent_speech"].count(phrase) == 1


async def test_model_error_does_not_repeat_a_closing_apology_after_other_text():
    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Hello there. Sorry, could you say that again? "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    llm = LLM()
    r = Rig(llm)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 25)
    await r.settle()
    llm.release.set()
    await r.settle()
    assert r.tts.texts.count(phrase) == 1
    assert "Hello there." in r.tts.texts
    assert r.s.history[-1]["content"].count(phrase) == 1


async def test_model_error_suppresses_a_second_apology_while_the_phrase_is_still_synthesizing():
    """spoken is still empty while TTS is blocked, so only sentences[-1] == phrase can suppress."""

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Sorry, could you say that again? "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.started = asyncio.Event()
            self.finish = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            self.started.set()
            await self.finish.wait()
            yield np.zeros(2400, dtype=np.int16)

    llm, tts = LLM(), TTS()
    r = Rig(llm, tts=tts)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await asyncio.wait_for(tts.started.wait(), timeout=2)
    assert tts.texts == [phrase] and r.sent == []
    await r.frames("q", 15)            # commit while that synthesize is still blocked
    await r.settle()
    llm.release.set()                  # model raises before the phrase has been sent
    await r.settle()
    assert r.sent == [] and tts.texts == [phrase]
    tts.finish.set()
    await r.settle()
    assert tts.texts == [phrase]
    assert len(r.sent) == 1
    assert [e["text"] for e in r.events if e["type"] == "agent_speech"] == [phrase]


async def test_tts_failure_on_the_phrase_still_speaks_once_after_the_model_errors():
    """A pre-commit TTS failure drops the ordinary sentence, even when that sentence is the
    apology. The model error must still queue one real fallback."""

    class LLM:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream(self, history, audio, followups, tools):
            yield "Sorry, could you say that again? "
            await self.release.wait()
            raise RuntimeError("model down")

        async def transcribe(self, audio):
            return "hello"

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            if len(self.texts) == 1:
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    llm, tts = LLM(), TTS()
    r = Rig(llm, tts=tts)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)             # TTS fails on the phrase before commit
    await r.settle()
    assert tts.texts == [phrase] and r.sent == []
    assert "user_end_of_turn" not in r.types()
    await r.frames("q", 15)            # commit; the model has not raised yet
    await r.settle()
    assert r.sent == []
    llm.release.set()
    await r.settle()
    assert tts.texts == [phrase, phrase]
    assert len(r.sent) == 1
    assert [e["text"] for e in r.events if e["type"] == "agent_speech"] == [phrase]
    assert any(e["type"] == "error" and e["where"] == "tts" for e in r.events)
    assert any(e["type"] == "error" and e["where"] == "produce" for e in r.events)


async def test_model_apology_sentence_does_not_suppress_later_tts_fallback():
    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            if text == "I will transfer you now.":
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    tts = TTS()
    r = Rig(FakeLLM([["Sorry, could you say that again? I will transfer you now."]]), tts=tts)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert tts.texts == [phrase]       # second sentence waits for commit
    await r.frames("q", 15)
    await r.settle()
    assert tts.texts == [phrase, "I will transfer you now.", phrase]
    assert len(r.sent) == 2
    assert any(e["type"] == "error" and e["where"] == "tts" for e in r.events)


async def test_tts_failure_of_model_apology_sentence_retries_once():
    """The model's own copy of the apology is not the fallback attempt. A TTS error on it
    is where=tts and is retried once, not relabelled where=tts_fallback with no retry."""

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.fail = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            if len(self.texts) == 1:
                await self.fail.wait()
                raise RuntimeError("tts down")
            yield np.zeros(2400, dtype=np.int16)

    tts = TTS()
    r = Rig(FakeLLM([["Sorry, could you say that again?"]]), tts=tts)
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    assert tts.texts == [phrase]
    await r.frames("q", 15)            # commit while that synthesize is blocked
    await r.settle()
    tts.fail.set()
    await r.settle()
    assert tts.texts == [phrase, phrase]
    assert len(r.sent) == 1
    assert any(e["type"] == "error" and e["where"] == "tts" for e in r.events)
    assert not any(e["type"] == "error" and e["where"] == "tts_fallback" for e in r.events)


async def test_noise_burst_triggers_nothing():
    llm = FakeLLM([["Should ", "not ", "happen."]])
    r = Rig(llm)
    await r.frames("q", 5)
    await r.frames("s", 2)
    await r.frames("q", 40)
    await r.settle()
    assert llm.calls == [] and r.sent == []


def test_default_model_context_budget_is_6000_tokens():
    assert SessionConfig().max_context_tokens == 6000


async def test_long_history_is_fitted_for_the_model_but_stored_in_full():
    llm = FakeLLM([["Okay."]])
    cfg = SessionConfig(max_context_tokens=200, tts_poll_s=0,
                        system_prompt="You are a helpful receptionist.")
    r = Rig(llm, cfg=cfg)
    seeded = []
    for i in range(60):
        seeded.append({"role": "user", "text": f"user-{i:02d}-" + ("abcd" * 8)})
        seeded.append({"role": "agent", "text": f"agent-{i:02d}-" + ("efgh" * 8)})
    r.s.seed_history(seeded)
    await r.frames("s", 15)
    await r.frames("q", 25)
    await r.settle()
    shown = llm.calls[0]["history"]
    from core.history import estimate_messages
    assert estimate_messages(shown) <= 200
    assert shown[0]["content"].startswith("You are a helpful")
    assert any(m["role"] == "user" and "user-59-" in m["content"] for m in shown)
    stored_users = [m for m in r.s.history if m["role"] == "user" and str(m["content"]).startswith("user-")]
    assert len(stored_users) == 60
    assert not any(str(m.get("content", "")).startswith("Earlier in this call:") for m in r.s.history)
    assert estimate_messages(r.s.history) > 200


async def test_tool_round_leaves_room_for_followups_in_the_token_budget():
    """Tool-call arguments are part of the request. A later round must shrink history to fit them."""
    from core.history import estimate_messages

    async def lookup(args):
        return "ok"

    llm = FakeLLM([["Checking. ", ToolCall("t1", "lookup", {"q": "x" * 3000})], ["Done."]])
    cfg = SessionConfig(max_context_tokens=200, tts_poll_s=0,
                        system_prompt="You are a helpful receptionist.")
    r = Rig(llm, cfg=cfg, tools={"lookup": lookup})
    seeded = []
    for i in range(30):
        seeded.append({"role": "user", "text": f"user-{i:02d}-" + ("abcd" * 8)})
        seeded.append({"role": "agent", "text": f"agent-{i:02d}-" + ("efgh" * 8)})
    r.s.seed_history(seeded)
    await r.frames("s", 15)
    await r.frames("q", 25)
    await r.settle()
    assert len(llm.calls) >= 2
    assert estimate_messages(llm.calls[0]["history"]) <= 200
    followups = llm.calls[1]["followups"]
    assert followups[0]["tool_calls"][0]["id"] == "t1"
    assert followups[1]["role"] == "tool" and followups[1]["tool_call_id"] == "t1"
    assert estimate_messages(followups) > 200          # arguments alone exceed the budget
    shown = llm.calls[1]["history"]
    assert shown[0]["content"].startswith("You are a helpful")
    assert [m for m in shown if m["role"] == "user"] == [
        m for m in shown if "user-29-" in (m.get("content") or "")]
    # Nothing older than the system prompt and that latest user message is still attached.
    assert len(shown) == 2
    stored_users = [m for m in r.s.history if str(m.get("content", "")).startswith("user-")]
    assert len(stored_users) == 30


async def test_second_turn_history_contains_first_transcript():
    llm = FakeLLM([["First ", "reply."], ["Second ", "reply."]])
    r = Rig(llm)
    for _ in range(2):
        await r.frames("s", 15)
        await r.frames("q", 25)
        await r.settle()
        await r.frames("q", 20)
    second = llm.calls[1]["history"]
    assert [m["role"] for m in second] == ["system", "user", "assistant"]
    assert second[1]["content"] == "caller said 1"
    assert second[2]["content"] == "First reply."


async def test_speculation_synthesizes_only_first_sentence_before_commit():
    llm = FakeLLM([["One two three four. ", "Five six seven. ", "Eight nine."]])
    r = Rig(llm)
    await r.frames("s", 15)
    await r.frames("q", 8)            # speculating, not committed
    await r.settle()
    assert r.tts.texts == ["One two three four."]          # later sentences wait for commit
    await r.frames("q", 15)           # commit
    await r.settle()
    assert len(r.tts.texts) == 3 and len(r.sent) == 3


async def test_later_sentences_wait_until_playback_needs_them():
    words = " ".join(["w"] * 30)      # 30 words x 80 ms = 2.4 s of audio per sentence
    llm = FakeLLM([[f"{words}. ", f"{words}. ", f"{words}."]])
    r = Rig(llm, cfg=SessionConfig(tts_lookahead_ms=1500, tts_poll_s=0))
    await r.frames("s", 15)
    await r.frames("q", 23)           # commit; first sentence sent (2.4 s buffered)
    await r.settle()
    assert len(r.tts.texts) == 1 and len(r.sent) == 1      # 2.4 s queued > 1.5 s lookahead
    r.clk.t += 1000                    # listener has heard 1 s: 1.4 s left < 1.5 s
    for _ in range(20):
        await asyncio.sleep(0.03)
    assert len(r.tts.texts) == 2
    r.clk.t += 10_000
    for _ in range(20):
        await asyncio.sleep(0.03)
    assert len(r.tts.texts) == 3 and r.s.history[-1]["content"].count("w") == 90


class Verdict:
    """Async end-of-turn scorer fake: returns `score` after `delay` asyncio ticks."""

    def __init__(self, score, delay=0):
        self.score, self.delay, self.calls = score, delay, 0

    async def __call__(self, audio, history):
        self.calls += 1
        for _ in range(self.delay):
            await asyncio.sleep(0)
        return self.score


def _eot_rig(score, delay=0):
    llm = FakeLLM([["Sure thing."]])
    v = Verdict(score, delay)
    r = Rig(llm)
    r.s.eot_async = v
    return r, v


async def test_confident_verdict_commits_the_reply_at_320ms_instead_of_640():
    r, v = _eot_rig(0.95)
    await r.frames("s", 15)
    await r.frames("q", 12)           # 384 ms of silence: past 320 ms, short of 640 ms
    await r.settle()
    assert v.calls == 1 and r.last("eot_score")["score"] == 0.95
    assert r.sent and "user_end_of_turn" in r.types()
    assert r.last("agent_first_audio")["latency_ms"] <= 352


async def test_confident_not_done_holds_the_reply_past_the_normal_640ms():
    r, v = _eot_rig(0.05)
    await r.frames("s", 15)
    await r.frames("q", 25)           # 800 ms of silence: normal timing would have committed at 640
    await r.settle()
    assert r.sent == [] and "user_end_of_turn" not in r.types()
    await r.frames("q", 10)           # past 1000 ms
    await r.settle()
    assert r.sent and "user_end_of_turn" in r.types()


async def test_verdict_for_a_resumed_pause_is_dropped_and_scorer_errors_fall_back():
    r, v = _eot_rig(0.95, delay=50)
    await r.frames("s", 15)
    await r.frames("q", 7)            # speculate: scorer starts but has not answered
    await r.frames("s", 8)            # caller resumes: scorer task must be cancelled
    await r.settle(400)
    assert "eot_score" not in r.types() and r.sent == []

    async def boom(audio, history):
        raise RuntimeError("scorer down")

    r2 = Rig(FakeLLM([["Hello there."]]))
    r2.s.eot_async = boom
    await r2.frames("s", 15)
    await r2.frames("q", 25)          # no verdict: default 640 ms timing still commits
    await r2.settle()
    assert r2.sent and any(e["type"] == "error" and e["where"] == "eot" for e in r2.events)


class _GateVAD:
    """Energy VAD. After arm(), the next prob() blocks the worker until release()."""

    def __init__(self):
        self.entered = threading.Event()
        self.release_ev = threading.Event()
        self._arm = False

    def arm(self) -> None:
        self.entered.clear()
        self.release_ev.clear()
        self._arm = True

    def prob(self, frame):
        if self._arm:
            self._arm = False
            self.entered.set()
            if not self.release_ev.wait(5):
                raise TimeoutError("vad gate was not released")
        return 0.9 if np.abs(frame).max() > 1000 else 0.1


class _HoldVerdict:
    """End-of-turn scorer that stays pending until go() so a test can release it mid-score."""

    def __init__(self, score=0.95):
        self.score = score
        self.calls = 0
        self.started = asyncio.Event()
        self._go = asyncio.Event()

    async def __call__(self, audio, history):
        self.calls += 1
        self.started.set()
        await self._go.wait()
        return self.score

    def go(self) -> None:
        self._go.set()


async def _wait_gate(gate: _GateVAD) -> None:
    for _ in range(200):
        if gate.entered.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("scoring did not reach the worker")


def _scheduler_rig(llm, eot=None, tts=None):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad-race")
    gate = _GateVAD()
    r = Rig(llm, tts=tts)
    r.s.vad = gate
    r.s.vad_scheduler = VADScheduler(pool)
    if eot is not None:
        r.s.eot_async = eot
    return r, gate, pool


async def test_inflight_speech_frame_drops_a_ready_eot_verdict():
    """A speech frame already awaiting VAD must cancel a 0.95 verdict, not lose the race to it.

    15 speech + 10 silence leaves the turn speculated at 320 ms (end target still 640). Releasing
    the verdict while the next speech frame is on the worker used to commit before turn.process.
    """
    eot = _HoldVerdict(0.95)
    llm = FakeLLM([["Sure thing."]])
    r, gate, pool = _scheduler_rig(llm, eot)
    feed_task = None
    try:
        await r.frames("s", 15)
        await r.frames("q", 10)             # 320 ms silence: speculated, not yet the 640 ms end
        assert eot.started.is_set() and eot.calls == 1
        assert r.s.turn._speculated and r.s.turn._silence_ms == 320 and r.s.turn._end_target == 640
        assert "user_end_of_turn" not in r.types()
        gate.arm()
        feed_task = asyncio.create_task(r.s.feed(SPEECH))
        await _wait_gate(gate)
        eot.go()
        for _ in range(5):                  # the verdict is runnable while this frame is still scoring
            await asyncio.sleep(0)
        assert "user_end_of_turn" not in r.types() and "eot_score" not in r.types()
        gate.release_ev.set()
        await asyncio.wait_for(feed_task, 2)
        await r.settle()
        assert "speculate_cancelled" in r.types()
        assert "user_end_of_turn" not in r.types()
        assert "eot_score" not in r.types()
        assert r.sent == [] and llm.n_transcribe == 0
        assert r.s.stats["speculations_cancelled"] == 1
        assert r.s.history == [{"role": "system", "content": r.s.cfg.system_prompt}]
    finally:
        gate.release_ev.set()
        if feed_task is not None and not feed_task.done():
            feed_task.cancel()
            await asyncio.gather(feed_task, return_exceptions=True)
        await r.s.close()
        pool.shutdown(wait=False, cancel_futures=True)


async def test_inflight_speech_frame_does_not_interrupt_inject_text():
    """Keypad text injected while a barge-in frame is scoring must not be barged by that frame.

    The agent is already speaking, so the backchannel gate is 450 ms. Fourteen speech frames
    leave run_ms at 448. The fifteenth is on the worker; inject_text during that score used to
    install the digit turn and then lose it when the frame resumed and interrupted the new generation.
    """
    llm = FakeLLM([["Got it."]])
    r, gate, pool = _scheduler_rig(llm, tts=FakeTTS(word_ms=400))
    feed_task = inject_task = None
    try:
        await r.s.say("Hello from the agent this is long.")
        await r.settle()
        await r.frames("s", 14)             # 448 ms < 450 ms backchannel gate while the greeting still plays
        assert r.s._agent_active() and r.s.turn._run_ms == 448 and not r.s.turn.in_turn
        assert "user_interrupted" not in r.types()
        gate.arm()
        feed_task = asyncio.create_task(r.s.feed(SPEECH))
        await _wait_gate(gate)
        inject_task = asyncio.create_task(
            r.s.inject_text("[The caller pressed these keypad digits: 42]"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert all("42" not in str(m.get("content")) for m in r.s.history)
        gate.release_ev.set()
        await asyncio.wait_for(asyncio.gather(feed_task, inject_task), 2)
        await r.settle()
        assert r.types().count("user_interrupted") == 1   # the greeting only, not the digit turn
        assert r.s._gen is not None and not r.s._gen.cancelled
        assert r.s.history[-2:] == [
            {"role": "user", "content": "[The caller pressed these keypad digits: 42]"},
            {"role": "assistant", "content": "Got it."}]
    finally:
        gate.release_ev.set()
        for task in (feed_task, inject_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await r.s.close()
        pool.shutdown(wait=False, cancel_futures=True)


async def test_inflight_speech_frame_does_not_interrupt_say():
    """A fixed line started while a barge-in frame is scoring must not be barged by that frame."""
    llm = FakeLLM([["nope"]])
    r, gate, pool = _scheduler_rig(llm, tts=FakeTTS(word_ms=400))
    feed_task = say_task = None
    try:
        await r.s.say("Hello from the agent this is long.")
        await r.settle()
        await r.frames("s", 14)             # 448 ms < 450 ms backchannel gate while the greeting still plays
        assert r.s._agent_active() and r.s.turn._run_ms == 448 and not r.s.turn.in_turn
        gate.arm()
        feed_task = asyncio.create_task(r.s.feed(SPEECH))
        await _wait_gate(gate)
        say_task = asyncio.create_task(r.s.say("Please call us back now friend."))
        for _ in range(10):
            await asyncio.sleep(0)
        assert all("Please call us back" not in str(m.get("content")) for m in r.s.history)
        gate.release_ev.set()
        await asyncio.wait_for(asyncio.gather(feed_task, say_task), 2)
        await r.settle()
        assert r.types().count("user_interrupted") == 1
        assert r.s._gen is not None and not r.s._gen.cancelled
        assert r.s.history[-1] == {"role": "assistant", "content": "Please call us back now friend."}
    finally:
        gate.release_ev.set()
        for task in (feed_task, say_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await r.s.close()
        pool.shutdown(wait=False, cancel_futures=True)


async def test_say_and_wait_playback_stays_interruptible():
    """The turn lock must not be held while say_and_wait waits out playback, or barge-in cannot run."""
    r = Rig(FakeLLM([["x."]]), tts=FakeTTS(word_ms=200))
    task = asyncio.create_task(r.s.say_and_wait("Please call us back later today."))
    await r.settle()
    assert r.sent
    await asyncio.wait_for(r.frames("s", 16), 2)  # 512 ms crosses the 450 ms backchannel gate while audio is queued
    await r.settle()
    assert r.clears == 1 and r.s.stats["interruptions"] == 1
    await asyncio.wait_for(task, 2)


# ------------------------------------------------------- tool-driven call actions (hang up / transfer)
from core.session import ToolResult  # noqa: E402


def _hangup_rig(strict):
    ended = []

    async def hang(args):
        async def after():
            ended.append(1)
        return ToolResult({"ok": True}, after=after, strict=strict)

    llm = FakeLLM([["Goodbye ", "now. ", ToolCall("h", "hang", {})], ["Bye ", "then."]])
    return Rig(llm, tts=FakeTTS(word_ms=400), tools={"hang": hang}), ended


async def test_hangup_runs_only_after_the_goodbye_has_finished_playing():
    r, ended = _hangup_rig(strict=True)
    await r.frames("s", 15)
    await r.frames("q", 22)            # commit: reply is sent, tool ran
    await r.settle()
    assert r.tts.texts == ["Goodbye now.", "Bye then."] and ended == []   # audio still playing: line must stay up
    r.clk.t += 5000
    await asyncio.sleep(0.2)
    assert ended == [1]
    assert "call_action_done" in r.types() and "tool_result" in r.types()


async def test_speculative_tts_failure_speaks_apology_before_deferred_hangup():
    """TTS dying before commit used to let a tool hang-up run against dead air. The apology
    is spoken once, ordinary sentences stay dropped, and the hang-up waits out that playback."""
    ended = []

    async def hang(args):
        async def after():
            ended.append(1)
        return ToolResult({"ok": True}, after=after, strict=True)

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        async def synthesize(self, text):
            self.texts.append(text)
            if text != "Sorry, could you say that again?":
                raise RuntimeError("tts down")
            n = len(text.split()) * 400 * 24   # 400 ms per word, long enough to still be playing
            yield np.zeros(n, dtype=np.int16)

    tts = TTS()
    llm = FakeLLM([["Goodbye ", "now. ", ToolCall("h", "hang", {})], ["Bye ", "then."]])
    r = Rig(llm, tts=tts, tools={"hang": hang})
    phrase = r.s.cfg.error_phrase
    await r.frames("s", 15)
    await r.frames("q", 8)             # speculate: TTS fails before the turn commits
    await r.settle()
    assert tts.texts == ["Goodbye now."] and r.sent == [] and ended == []
    await r.frames("q", 15)            # commit; the tool sets a deferred hang-up
    await r.settle()
    assert tts.texts == ["Goodbye now.", phrase]
    assert "Bye then." not in tts.texts
    assert len(r.sent) == 1 and ended == []
    t_sent, n, rate = r.sent[0]
    dur = 1000.0 * n / rate
    assert dur > 0 and r.clk.t < t_sent + dur
    r.clk.t = t_sent + dur - 1         # apology is still playing: the line must stay up
    await asyncio.sleep(0.2)
    assert ended == []
    r.clk.t = t_sent + dur + 1         # playback clock passes the apology
    await asyncio.sleep(0.2)
    assert ended == [1]
    assert [e["text"] for e in r.events if e["type"] == "agent_speech"] == [phrase]


def _held_hangup_tts():
    """TTS blocks the goodbye until `fail` is set, then the apology plays for a long time.
    The hang-up tool waits on `release` so it can outlive that synthesize."""
    ended = []
    entered = asyncio.Event()
    release = asyncio.Event()
    rounds = []

    async def hang(args):
        entered.set()
        await release.wait()

        async def after():
            ended.append(1)
        return ToolResult({"ok": True}, after=after, strict=True)

    class LLM:
        async def stream(self, history, audio, followups, tools):
            rounds.append(copy.deepcopy(followups))
            if not followups and len(rounds) == 1:
                yield "Goodbye now. "
                yield ToolCall("h", "hang", {})
                return
            if followups:
                yield "Still talking. "
                return
            yield "Next reply."

        async def transcribe(self, audio):
            return "hello"

    class TTS:
        sample_rate = 24000

        def __init__(self):
            self.texts = []
            self.started = asyncio.Event()
            self.fail = asyncio.Event()

        async def synthesize(self, text):
            self.texts.append(text)
            if text != "Sorry, could you say that again?":
                self.started.set()
                await self.fail.wait()
                raise RuntimeError("tts down")
            n = len(text.split()) * 400 * 24
            yield np.zeros(n, dtype=np.int16)

    tts = TTS()
    llm = LLM()
    r = Rig(llm, tts=tts, tools={"hang": hang})
    return r, tts, llm, entered, release, ended, rounds


async def _commit_while_goodbye_tts_blocked(r, tts, entered):
    await r.frames("s", 15)
    await r.frames("q", 8)
    await r.settle()
    await asyncio.wait_for(tts.started.wait(), timeout=2)
    await r.frames("q", 15)            # commit; the tool is awaiting and TTS has not raised
    await r.settle()
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert r.sent == []


async def test_postcommit_tts_failure_holds_hangup_until_the_apology_finishes():
    """TTS that raises after commit used to end synth before a slow tool stored `after`,
    so the hang-up never ran."""
    r, tts, llm, entered, release, ended, rounds = _held_hangup_tts()
    phrase = r.s.cfg.error_phrase
    await _commit_while_goodbye_tts_blocked(r, tts, entered)
    tts.fail.set()
    await r.settle()
    assert tts.texts == ["Goodbye now.", phrase] and len(r.sent) == 1 and ended == []
    release.set()                      # tool returns while the apology is still playing
    await r.settle()
    assert "Still talking." not in tts.texts
    assert ended == []
    t_sent, n, rate = r.sent[0]
    dur = 1000.0 * n / rate
    assert dur > 0 and r.clk.t < t_sent + dur
    r.clk.t = t_sent + dur - 1
    await asyncio.sleep(0.2)
    assert ended == []
    r.clk.t = t_sent + dur + 1
    await asyncio.sleep(0.2)
    assert ended == [1]
    assert [e["text"] for e in r.events if e["type"] == "agent_speech"] == [phrase]


async def test_second_utterance_cancels_produce_waiting_on_a_tool_after_tts_failure():
    """Once the apology has finished, a produce task still inside the tool must stay the
    active generation. A new utterance cancels it instead of letting it stream again."""
    r, tts, llm, entered, release, ended, rounds = _held_hangup_tts()
    await _commit_while_goodbye_tts_blocked(r, tts, entered)
    tts.fail.set()
    await r.settle()
    assert len(r.sent) == 1 and ended == []
    old = r.s._gen
    t_sent, n, rate = r.sent[0]
    r.clk.t = t_sent + (1000.0 * n / rate) + 1    # apology clock already past; tool still awaiting
    await asyncio.sleep(0.05)
    await r.frames("s", 10)            # caller speaks again: must barge in, not orphan produce
    await r.frames("q", 10)
    await r.settle()
    assert old.cancelled and all(t.done() for t in old.tasks)
    assert r.s.stats["interruptions"] == 1
    assert r.s._gen is not old and len(rounds) >= 2
    assert all(not f for f in rounds)
    n_rounds = len(rounds)
    release.set()
    await r.settle()
    assert len(rounds) == n_rounds and all(not f for f in rounds)
    assert ended == []


async def test_strict_hangup_still_happens_if_caller_talks_over_the_goodbye():
    r, ended = _hangup_rig(strict=True)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    await r.frames("s", 16)            # 512 ms: past the 450 ms backchannel gate, so the goodbye is cut
    await r.settle()
    assert r.clears == 1 and ended == [1]


async def test_non_strict_hangup_is_cancelled_by_the_caller_talking():
    r, ended = _hangup_rig(strict=False)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    await r.frames("s", 16)            # 512 ms of overlap: a real interruption, not a backchannel
    await r.settle()
    r.clk.t += 5000
    await asyncio.sleep(0.2)
    assert r.clears == 1 and ended == []


# ---------------------------------------------------------------------------------- greeting
async def test_greeting_is_spoken_first_without_a_model_call_and_enters_history():
    llm = FakeLLM([["Sure thing."]])
    r = Rig(llm, cfg=SessionConfig(tts_poll_s=0, greeting="Hi, this is Arvindh's voice agent. How can I help you today?"))
    await r.s.start()
    await r.settle()
    assert llm.calls == []                                           # no model involved
    assert r.tts.texts == ["Hi, this is Arvindh's voice agent.", "How can I help you today?"]
    assert len(r.sent) == 2 and "greeting" in r.types() and "agent_first_audio" in r.types()
    assert r.s.history[-1] == {"role": "assistant",
                               "content": "Hi, this is Arvindh's voice agent. How can I help you today?"}


async def test_no_greeting_means_silence_until_the_caller_speaks():
    r = Rig(FakeLLM([["x."]]))
    await r.s.start()
    await r.settle()
    assert r.sent == [] and r.tts.texts == []


async def test_caller_can_talk_over_the_greeting_and_the_next_turn_proceeds():
    llm = FakeLLM([["Booked."]])
    r = Rig(llm, tts=FakeTTS(word_ms=300),
            cfg=SessionConfig(tts_poll_s=0, greeting="Hi, this is the voice agent. How can I help you today?"))
    await r.s.start()
    await r.settle()
    await r.frames("q", 30)            # ~1 s of the greeting plays...
    await r.frames("s", 16)            # 512 ms over the greeting: past the 450 ms backchannel gate
    await r.settle()
    assert r.clears == 1 and r.last("user_interrupted")["cut"] is True
    await r.frames("s", 10)
    await r.frames("q", 30)
    await r.settle()
    assert len(llm.calls) == 1 and r.tts.texts[-1] == "Booked."
    h = llm.calls[0]["history"][1]                                   # the greeting precedes the caller's first turn,
    assert h["role"] == "assistant" and h["content"].endswith("[interrupted]")   # trimmed to what was heard
