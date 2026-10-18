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
