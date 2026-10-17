"""Conversation replay scores tool use without a paid baseline API."""
import asyncio
import json

import numpy as np

from core.session import Session, SessionConfig, ToolCall
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
    over = _score(case, [{"type": "tool_call", "name": "book"}, {"type": "user_interrupted"},
                         {"type": "agent_first_audio", "latency_ms": 640}])
    assert over["task_success"] is False
    assert over["wrong_actions"] == []
    assert over["interruptions"] == 1
    assert over["first_audio_ms"] == 640
    assert _score(case, [{"type": "tool_call", "name": "book"}])["task_success"] is True
    assert _score(case, [])["task_success"] is False
    allowed = Case(audio=SPEECH, expect_tools=["book"], forbid_tools=[], max_interruptions=1)
    assert _score(allowed, [{"type": "tool_call", "name": "book"},
                            {"type": "user_interrupted"}])["task_success"] is True


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
