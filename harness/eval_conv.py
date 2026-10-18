"""Replay caller audio through a Session and score the call. No paid API.

`compare` takes score dicts that were already produced (ours, and a baseline such as a
paid service). It only averages those dicts.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import numpy as np

FRAME = 512          # Session.feed wants 32 ms at 16 kHz
FRAME_MS = 32.0
# Minimum tail. The real length follows the session: end_ms is 640 by default (20 frames)
# but patient_end_ms is 1000 when an end-of-turn score says the caller is not done.
# Two extra frames cover a clip that already ends on a partial frame.
SILENCE_TAIL_FRAMES = 22
_SETTLE_YIELDS = 20
# After the tail, a tool or the first TTS chunk may still be awaiting. close()
# cancels that work. Cap the wait; a rig clock (.t) keeps moving inside it.
_TURN_WAIT_S = 5.0
_TURN_POLL_S = 0.02


@dataclass
class Case:
    audio: np.ndarray
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    max_interruptions: int = 0


def _tail_frames(session) -> int:
    """How many silent frames this session needs before its turn timer can commit.

    A fixed 22-frame tail is 704 ms. That never reaches ``end_ms`` or ``patient_end_ms``
    when either is 1000, so the reply stays speculative and the expected tool never runs.
    END_OF_TURN cannot fire until silence has also reached ``speculate_ms``.
    """
    turn = getattr(getattr(session, "cfg", None), "turn", None)
    frame_ms = float(getattr(turn, "frame_ms", FRAME_MS) or FRAME_MS)
    need = max(
        float(getattr(turn, "end_ms", 640.0) or 0.0),
        float(getattr(turn, "patient_end_ms", 0.0) or 0.0),
        float(getattr(turn, "fast_end_ms", 0.0) or 0.0),
        float(getattr(turn, "speculate_ms", 0.0) or 0.0),
    )
    if need <= 0 or frame_ms <= 0:
        return SILENCE_TAIL_FRAMES
    return max(SILENCE_TAIL_FRAMES, int(math.ceil(need / frame_ms)) + 2)


def _frames(audio: np.ndarray, tail: int = SILENCE_TAIL_FRAMES) -> list[np.ndarray]:
    pcm = np.asarray(audio, dtype=np.int16).reshape(-1)
    if len(pcm) == 0:
        chunks: list[np.ndarray] = []
    else:
        pad = (-len(pcm)) % FRAME
        if pad:
            pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
        chunks = [pcm[i:i + FRAME] for i in range(0, len(pcm), FRAME)]
    silence = np.zeros(FRAME, dtype=np.int16)
    chunks.extend(silence.copy() for _ in range(tail))
    return chunks


def _advance_clock(session, ms: float) -> None:
    """Rig-style clocks store milliseconds on `.t`. A wall clock has no `.t` and is left alone."""
    clock = getattr(session, "clock", None)
    t = getattr(clock, "t", None)
    if isinstance(t, (int, float)):
        clock.t = float(t) + ms


def _turn_ready(session) -> bool:
    """True when close() would not cancel an in-flight tool body or the first audio chunk.

    An uncommitted reply never runs tools. A committed one is ready once its produce
    task has finished (every tool_result is out) and the first chunk has been sent,
    or synthesis has ended without audio.
    """
    gen = getattr(session, "_gen", None)
    if gen is None or not gen.committed or gen.cancelled:
        return True
    tasks = gen.tasks or []
    produce = tasks[0] if tasks else None
    if produce is not None and not produce.done():
        return False
    if gen.first_audio_sent or gen.produce_done:
        return True
    synth = tasks[1] if len(tasks) > 1 else None
    return synth is None or synth.done()


async def _await_turn(session) -> None:
    """Let tools return and the first TTS chunk go out before close() cancels them.

    Yields alone never finish a real ``asyncio.sleep`` in the tool or synthesizer.
    While that wait runs, the rig clock (``.t``) advances so a later sentence blocked
    on ``buffered_ms`` / ``tts_poll_s`` cannot spin until the cap.
    """
    for _ in range(100):
        if _turn_ready(session):
            return
        await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TURN_WAIT_S
    while not _turn_ready(session):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(_TURN_POLL_S, remaining))
        if _turn_ready(session):
            return
        _advance_clock(session, FRAME_MS)


async def _drive(session, audio: np.ndarray) -> None:
    for frame in _frames(audio, _tail_frames(session)):
        await session.feed(frame)
        for _ in range(_SETTLE_YIELDS):
            await asyncio.sleep(0)
        _advance_clock(session, FRAME_MS)
    await _await_turn(session)


def _score(case: Case, events: list[dict]) -> dict:
    # tool_call is emitted before the body runs. close() can cancel it; only a
    # tool_result means that call finished, for expect_tools and forbid_tools.
    tools = [e["name"] for e in events if e.get("type") == "tool_result" and e.get("name")]
    forbidden = set(case.forbid_tools)
    wrong = [name for name in tools if name in forbidden]
    interruptions = sum(1 for e in events if e.get("type") == "user_interrupted")
    first = next((e.get("latency_ms") for e in events if e.get("type") == "agent_first_audio"), None)
    tools_ok = all(name in tools for name in case.expect_tools) and not wrong
    # The interruption budget is part of the case: a tool-correct call that barges in too often
    # is not a success. The count is still returned on its own.
    return {"task_success": bool(tools_ok and interruptions <= case.max_interruptions),
            "wrong_actions": wrong, "interruptions": interruptions, "first_audio_ms": first}


async def replay(session_factory, case: Case) -> dict:
    """Feed `case.audio` in 512-sample frames plus a silence tail. `session_factory` returns a Session."""
    session = session_factory()
    if asyncio.iscoroutine(session):
        session = await session
    events: list[dict] = []
    previous = session.on_event

    def collect(event: dict) -> None:
        events.append(event)
        if previous is not None:
            previous(event)

    session.on_event = collect
    try:
        await _drive(session, case.audio)
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            await close()
    return _score(case, events)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compare(ours: list[dict], baseline: list[dict]) -> dict:
    """Mean task_success (0..1) and mean first_audio_ms. `baseline` is already scored."""
    def side(rows: list[dict]) -> dict:
        return {"task_success": _mean(rows, "task_success"), "first_audio_ms": _mean(rows, "first_audio_ms")}

    return {"ours": side(list(ours)), "baseline": side(list(baseline))}
