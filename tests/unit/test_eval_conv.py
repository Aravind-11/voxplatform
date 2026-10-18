"""Conversation replay scores tool use without a paid baseline API."""
import asyncio
import json

import numpy as np

from core.session import Session, SessionConfig, ToolCall, TurnConfig
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


async def test_replay_fails_when_expected_and_forbidden_tools_both_finish():
    llm = FakeLLM([["Done. ", ToolCall("t1", "book", {"day": "mon"}),
                    ToolCall("t2", "cancel", {})], ["All set."]])
    out = await replay(factory(llm, {"book": _book, "cancel": _cancel}), _case())
    assert out["task_success"] is False
    assert out["wrong_actions"] == ["cancel"]


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
                         {"type": "agent_first_audio", "latency_ms": 640}])
    assert over["task_success"] is False
    assert over["wrong_actions"] == []
    assert over["interruptions"] == 1
    assert over["first_audio_ms"] == 640
    assert _score(case, [{"type": "tool_result", "name": "book"}])["task_success"] is True
    # Emitted before the body runs. A cancelled call must not pass the case.
    assert _score(case, [{"type": "tool_call", "name": "book"}])["task_success"] is False
    assert _score(case, [])["task_success"] is False
    allowed = Case(audio=SPEECH, expect_tools=["book"], forbid_tools=[], max_interruptions=1)
    assert _score(allowed, [{"type": "tool_result", "name": "book"},
                            {"type": "user_interrupted"}])["task_success"] is True
    both = _score(case, [{"type": "tool_result", "name": "book"},
                         {"type": "tool_result", "name": "cancel"}])
    assert both["task_success"] is False
    assert both["wrong_actions"] == ["cancel"]


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
