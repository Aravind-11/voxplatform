"""Conversation replay scores tool use without a paid baseline API."""
import asyncio
import json

import numpy as np

from core.session import Session, SessionConfig, ToolCall, TurnConfig
from harness import eval_conv
from harness.eval_conv import Case, _score, compare, replay
from scripts.eval_compare import main as compare_main

SPEECH = np.full(15 * 512, 5000, dtype=np.int16)


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

    async def synthesize(self, text):
        await asyncio.sleep(0)
        yield np.zeros(len(text.split()) * 80 * 24, dtype=np.int16)


class FakeLLM:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def stream(self, history, audio, followups, tools):
        i = len(self.calls)
        self.calls.append(i)
        for item in self.replies[min(i, len(self.replies) - 1)]:
            await asyncio.sleep(0)
            yield item

    async def transcribe(self, audio):
        return "book monday"


def factory(llm, tools):
    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=llm, tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools=tools, cfg=SessionConfig(tts_poll_s=0))

    return make


async def _book(args):
    return {"ok": True}


async def _cancel(args):
    return {"ok": True}


def _case(**kw):
    base = dict(audio=SPEECH, expect_tools=["book"], forbid_tools=["cancel"], max_interruptions=0)
    base.update(kw)
    return Case(**base)


async def test_replay_waits_long_enough_for_the_sessions_end_of_turn():
    """22 silent frames are 704 ms. end_ms=1000, and patient_end_ms after a low end-of-turn
    score, never fire, so the expected tool is not run and the case is scored a failure."""
    tools = {"book": _book, "cancel": _cancel}

    def slow_end():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools=tools,
                       cfg=SessionConfig(tts_poll_s=0, turn=TurnConfig(end_ms=1000)))

    async def not_done(audio, history):
        return 0.0

    def patient():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools=tools, cfg=SessionConfig(tts_poll_s=0), eot_async=not_done)

    slow = await replay(slow_end, _case())
    assert slow["task_success"] is True and slow["first_audio_ms"] >= 1000
    waited = await replay(patient, _case())
    assert waited["task_success"] is True and waited["first_audio_ms"] >= 1000


async def test_replay_finishes_a_slow_tool_and_the_first_audio_chunk():
    """A tool and the synthesizer that are still awaiting used to be cancelled by close().
    The tool_call was already emitted, so the case scored success with no audio."""
    state = {"finished": False}

    async def book(args):
        await asyncio.sleep(0.2)
        state["finished"] = True
        return {"ok": True}

    class SlowTTS:
        sample_rate = 24000

        async def synthesize(self, text):
            await asyncio.sleep(0.05)
            yield np.zeros(160, dtype=np.int16)

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=SlowTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0))

    out = await replay(make, _case())
    assert state["finished"] is True
    assert out["first_audio_ms"] is not None
    assert out["task_success"] is True
    assert out["wrong_actions"] == []


async def test_replay_keeps_silence_until_speculate_ms():
    """END_OF_TURN cannot fire before speculation. speculate_ms=1500 used to keep the
    patient_end tail (1000 ms), so the expected tool never ran."""
    tools = {"book": _book, "cancel": _cancel}

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools=tools,
                       cfg=SessionConfig(tts_poll_s=0, turn=TurnConfig(speculate_ms=1500)))

    out = await replay(make, _case())
    assert out["task_success"] is True
    assert out["wrong_actions"] == []
    assert out["first_audio_ms"] is not None


async def test_replay_keeps_silence_until_end_ms_when_it_is_the_longest_timer():
    """end_ms=1500 with patient_end_ms=400. A tail that drops end_ms never reaches the
    turn timer, so the expected tool does not run."""
    turn = TurnConfig(end_ms=1500, patient_end_ms=400)
    tools = {"book": _book, "cancel": _cancel}

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools=tools,
                       cfg=SessionConfig(tts_poll_s=0, turn=turn))

    out = await replay(make, _case())
    assert out["task_success"] is True
    assert out["wrong_actions"] == []
    assert out["first_audio_ms"] >= turn.end_ms


async def test_replay_keeps_silence_until_fast_end_ms_when_the_scorer_is_confident():
    """A sync eot_fn above eot_threshold commits at fast_end_ms. That timer is the
    longest one here; leaving it out of the tail drops the expected tool."""
    turn = TurnConfig(fast_end_ms=2000)

    def confident(audio, history):
        return 0.95

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": _book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0, turn=turn), eot_fn=confident)

    out = await replay(make, _case())
    assert out["task_success"] is True
    assert out["wrong_actions"] == []
    assert out["first_audio_ms"] >= turn.fast_end_ms


async def test_replay_fails_when_expected_and_forbidden_tools_both_finish():
    llm = FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"}),
                    ToolCall("t2", "cancel", {})], ["All set."]])
    out = await replay(factory(llm, {"book": _book, "cancel": _cancel}), _case())
    assert out["task_success"] is False
    assert out["wrong_actions"] == ["cancel"]


async def test_replay_scores_a_forbidden_tool_still_running_at_the_cap(monkeypatch):
    """book's tool_result used to be enough for a pass. cancel was already in the
    committed turn, still awaiting when the cap fired, and close() dropped it with
    no tool_result, so _score never saw the forbidden call."""
    monkeypatch.setattr(eval_conv, "_TURN_WAIT_S", 0.2)
    ran = {"started": False, "finished": False}

    async def book(args):
        return {"ok": True}

    async def cancel(args):
        ran["started"] = True
        await asyncio.sleep(30)
        ran["finished"] = True
        return {"ok": True}

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {}),
                                     ToolCall("t2", "cancel", {})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": book, "cancel": cancel},
                       cfg=SessionConfig(tts_poll_s=0))

    out = await replay(make, _case())
    assert ran["started"] is True
    assert ran["finished"] is False
    assert out["task_success"] is False
    assert out["wrong_actions"] == ["cancel"]


async def test_replay_does_not_pass_when_produce_is_still_running_at_the_cap(monkeypatch):
    """The follow-up model round is not a tool_call yet. Expiring the cap there used to
    leave task_success true off the earlier tool_result."""
    monkeypatch.setattr(eval_conv, "_TURN_WAIT_S", 0.15)
    ran = {"book": False}

    class HangFollowup(FakeLLM):
        async def stream(self, history, audio, followups, tools):
            i = len(self.calls)
            self.calls.append(i)
            if followups:
                await asyncio.sleep(30)
            for item in self.replies[min(i, len(self.replies) - 1)]:
                await asyncio.sleep(0)
                yield item

    async def book(args):
        ran["book"] = True
        return {"ok": True}

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=HangFollowup([["Done. ", ToolCall("t1", "book", {})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0))

    out = await replay(make, _case())
    assert ran["book"] is True
    assert out["task_success"] is False
    assert out["wrong_actions"] == []


async def test_replay_waits_for_a_slow_end_of_turn_verdict_before_more_silence():
    """eot_async that awaits real time used to lose to the sleep(0) tail: end_ms
    committed, the turn reset, and a 0.0 score never raised patient_end_ms."""
    async def not_done(audio, history):
        await asyncio.sleep(0.05)
        return 0.0

    def patient():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": _book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0), eot_async=not_done)

    out = await replay(patient, _case())
    assert out["task_success"] is True
    assert out["first_audio_ms"] is not None
    assert out["first_audio_ms"] >= TurnConfig().patient_end_ms


async def test_replay_keeps_end_ms_when_the_end_of_turn_verdict_misses_the_cap(monkeypatch):
    """A verdict still pending at the cap must not block the tail. end_ms stands."""
    monkeypatch.setattr(eval_conv, "_TURN_WAIT_S", 0.1)

    async def not_done(audio, history):
        await asyncio.sleep(5)
        return 0.0

    def patient():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": _book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0), eot_async=not_done)

    out = await replay(patient, _case())
    assert out["task_success"] is True
    assert out["first_audio_ms"] == 640


async def test_replay_cancels_a_late_end_of_turn_verdict_before_the_next_frame(monkeypatch):
    """The cap used to leave the verdict pending. The next silence frame then ran it,
    apply_eot(0.0) raised patient_end_ms, and first_audio_ms was 1024 instead of end_ms."""
    monkeypatch.setattr(eval_conv, "_TURN_WAIT_S", 0.1)
    held = {}

    async def not_done(audio, history):
        turn = held["session"].turn
        mark = turn._silence_ms
        while turn._silence_ms <= mark:
            await asyncio.sleep(0)
        return 0.0

    def patient():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        session = Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                          tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                          clock=Clock(), tools={"book": _book, "cancel": _cancel},
                          cfg=SessionConfig(tts_poll_s=0), eot_async=not_done)
        held["session"] = session
        return session

    out = await replay(patient, _case())
    assert out["task_success"] is True
    assert out["first_audio_ms"] == 640


async def test_replay_counts_first_audio_only_after_the_write_returns():
    """agent_first_audio is emitted before send_audio returns. close() used to cancel
    that write and still report its latency."""
    state = {"finished": False}

    async def send(pcm, rate):
        await asyncio.sleep(0.3)
        state["finished"] = True

    def make():
        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": _book, "cancel": _cancel},
                       cfg=SessionConfig(tts_poll_s=0))

    out = await replay(make, _case())
    assert state["finished"] is True
    assert out["first_audio_ms"] is not None
    assert out["task_success"] is True


async def test_replay_runs_a_forbidden_tool_queued_behind_a_slow_call():
    """book's tool_call is emitted before its body returns. Closing there used to score
    success and never start the cancel that was already in the committed turn."""
    ran = {"book": False, "cancel": False}

    async def book(args):
        await asyncio.sleep(0.2)
        ran["book"] = True
        return {"ok": True}

    async def cancel(args):
        ran["cancel"] = True
        return {"ok": True}

    def make():
        async def send(pcm, rate):
            return None

        async def clear():
            return None

        return Session(llm=FakeLLM([["Done. ", ToolCall("t1", "book", {}),
                                     ToolCall("t2", "cancel", {})], ["All set."]]),
                       tts=FakeTTS(), vad=EnergyVAD(), send_audio=send, clear_audio=clear,
                       clock=Clock(), tools={"book": book, "cancel": cancel},
                       cfg=SessionConfig(tts_poll_s=0))

    out = await replay(make, _case())
    assert ran["book"] is True and ran["cancel"] is True
    assert out["task_success"] is False
    assert out["wrong_actions"] == ["cancel"]


async def test_replay_scores_expected_tool_and_rejects_a_forbidden_one():
    ok_llm = FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"})], ["All set."]])
    bad_llm = FakeLLM([["No. ", ToolCall("t1", "cancel", {})], ["Cancelled."]])
    tools = {"book": _book, "cancel": _cancel}
    good = await replay(factory(ok_llm, tools), _case())
    assert good["task_success"] is True
    assert good["wrong_actions"] == []
    assert good["interruptions"] == 0
    assert good["first_audio_ms"] == 640
    bad = await replay(factory(bad_llm, tools), _case())
    assert bad["task_success"] is False
    assert bad["wrong_actions"] == ["cancel"]
    assert "book" not in bad["wrong_actions"]


def test_task_success_requires_expected_tools_and_the_interruption_budget():
    case = Case(audio=SPEECH, expect_tools=["book"], forbid_tools=["cancel"], max_interruptions=0)
    over = _score(case, [{"type": "tool_result", "name": "book"}, {"type": "user_interrupted"},
                         {"type": "agent_first_audio", "latency_ms": 640},
                         {"type": "agent_speech", "text": "Done."}])
    assert over["task_success"] is False
    assert over["wrong_actions"] == []
    assert over["interruptions"] == 1
    assert over["first_audio_ms"] == 640
    # The event is emitted before send_audio returns. A cancelled write is not a latency.
    dropped = _score(case, [{"type": "tool_result", "name": "book"},
                            {"type": "agent_first_audio", "latency_ms": 640}])
    assert dropped["first_audio_ms"] is None
    assert dropped["task_success"] is True
    assert _score(case, [{"type": "tool_result", "name": "book"}])["task_success"] is True
    # Emitted before the body runs. A cancelled call must not pass the case.
    assert _score(case, [{"type": "tool_call", "name": "book"}])["task_success"] is False
    hung = _score(case, [{"type": "tool_call", "name": "book"}, {"type": "tool_result", "name": "book"},
                         {"type": "tool_call", "name": "cancel"},
                         {"type": "agent_first_audio", "latency_ms": 640},
                         {"type": "agent_speech", "text": "Done."}])
    assert hung["task_success"] is False
    assert hung["wrong_actions"] == ["cancel"]
    assert hung["first_audio_ms"] == 640
    other = _score(case, [{"type": "tool_call", "name": "book"}, {"type": "tool_result", "name": "book"},
                          {"type": "tool_call", "name": "lookup"}])
    assert other["task_success"] is False
    assert other["wrong_actions"] == []
    # Produce still running at the cap, with no open tool_call left to see in the events.
    unsettled = _score(case, [{"type": "tool_result", "name": "book"},
                              {"type": "agent_first_audio", "latency_ms": 640},
                              {"type": "agent_speech", "text": "Done."}],
                       produce_settled=False)
    assert unsettled["task_success"] is False
    assert unsettled["wrong_actions"] == []
    assert unsettled["first_audio_ms"] == 640
    assert _score(case, [])["task_success"] is False
    allowed = Case(audio=SPEECH, expect_tools=["book"], forbid_tools=[], max_interruptions=1)
    assert _score(allowed, [{"type": "tool_result", "name": "book"},
                            {"type": "user_interrupted"}])["task_success"] is True
    both = _score(case, [{"type": "tool_result", "name": "book"},
                         {"type": "tool_result", "name": "cancel"}])
    assert both["task_success"] is False
    assert both["wrong_actions"] == ["cancel"]
    # Every expected name has to finish. The first result alone is not the case.
    two = Case(audio=SPEECH, expect_tools=["book", "lookup"], forbid_tools=["cancel"])
    assert _score(two, [{"type": "tool_result", "name": "book"}])["task_success"] is False
    assert _score(two, [{"type": "tool_result", "name": "book"},
                        {"type": "tool_result", "name": "lookup"}])["task_success"] is True
    # Every forbidden result is listed, in the order the events arrived.
    many = Case(audio=SPEECH, expect_tools=["book"], forbid_tools=["cancel", "transfer"])
    listed = _score(many, [{"type": "tool_result", "name": "cancel"},
                           {"type": "tool_result", "name": "book"},
                           {"type": "tool_result", "name": "transfer"}])
    assert listed["task_success"] is False
    assert listed["wrong_actions"] == ["cancel", "transfer"]


def test_compare_averages_task_success_and_first_audio():
    ours = [{"task_success": True, "first_audio_ms": 100.0},
            {"task_success": False, "first_audio_ms": 300.0},
            {"task_success": True, "first_audio_ms": None}]
    baseline = [{"task_success": True, "first_audio_ms": 200.0}]
    out = compare(ours, baseline)
    assert out["ours"]["task_success"] == 2 / 3
    assert out["ours"]["first_audio_ms"] == 200.0
    assert out["baseline"]["task_success"] == 1.0
    assert out["baseline"]["first_audio_ms"] == 200.0
    assert compare([], []) == {"ours": {"task_success": None, "first_audio_ms": None},
                               "baseline": {"task_success": None, "first_audio_ms": None}}


def test_eval_compare_script_reads_jsonl(tmp_path, capsys):
    ours = tmp_path / "ours.jsonl"
    base = tmp_path / "base.jsonl"
    ours.write_text('{"task_success": true, "first_audio_ms": 100}\n{"task_success": false, "first_audio_ms": 300}\n')
    base.write_text('{"task_success": true, "first_audio_ms": 200}\n')
    assert compare_main([str(ours), str(base)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ours"]["task_success"] == 0.5
    assert out["ours"]["first_audio_ms"] == 200.0
    assert out["baseline"]["task_success"] == 1.0
