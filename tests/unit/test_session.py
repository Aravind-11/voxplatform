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

    The agent is already speaking, so the barge threshold is 192 ms. Five speech frames leave
    run_ms at 160. The sixth is on the worker; inject_text during that score used to install the
    digit turn and then lose it when the frame resumed and interrupted the new generation.
    """
    llm = FakeLLM([["Got it."]])
    r, gate, pool = _scheduler_rig(llm, tts=FakeTTS(word_ms=400))
    feed_task = inject_task = None
    try:
        await r.s.say("Hello from the agent this is long.")
        await r.settle()
        await r.frames("s", 5)              # 160 ms < 192 ms barge-in while the greeting still plays
        assert r.s._agent_active() and r.s.turn._run_ms == 160 and not r.s.turn.in_turn
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
        await r.frames("s", 5)
        assert r.s._agent_active() and r.s.turn._run_ms == 160 and not r.s.turn.in_turn
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
    await asyncio.wait_for(r.frames("s", 6), 2)   # 192 ms crosses the barge-in threshold while audio is queued
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
