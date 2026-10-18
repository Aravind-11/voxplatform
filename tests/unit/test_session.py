import asyncio
import copy

import numpy as np
import pytest

from core.session import Session, SessionConfig, ToolCall

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
    await r.frames("s", 6)            # caller talks over the agent (needs 192 ms while it speaks)
    await r.settle()
    assert r.clears == 1
    ev = r.last("user_interrupted")
    assert ev["cut"] is True
    assert ev["heard"] == "One two three four five. Six seven eight"
    assert r.s.history[-1] == {"role": "assistant",
                               "content": "One two three four five. Six seven eight [interrupted]"}
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


async def test_strict_hangup_still_happens_if_caller_talks_over_the_goodbye():
    r, ended = _hangup_rig(strict=True)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    await r.frames("s", 6)             # barge-in while the goodbye is playing
    await r.settle()
    assert r.clears == 1 and ended == [1]


async def test_non_strict_hangup_is_cancelled_by_the_caller_talking():
    r, ended = _hangup_rig(strict=False)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    await r.frames("s", 6)
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
    await r.frames("s", 8)             # ...then the caller starts talking over it
    await r.settle()
    assert r.clears == 1 and r.last("user_interrupted")["cut"] is True
    await r.frames("s", 10)
    await r.frames("q", 30)
    await r.settle()
    assert len(llm.calls) == 1 and r.tts.texts[-1] == "Booked."
    h = llm.calls[0]["history"][1]                                   # the greeting precedes the caller's first turn,
    assert h["role"] == "assistant" and h["content"].endswith("[interrupted]")   # trimmed to what was heard
