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


def _produce_settled(session) -> bool:
    """False when a committed produce task is still running (tool body or a later round)."""
    gen = getattr(session, "_gen", None)
    if gen is None or not gen.committed or gen.cancelled:
        return True
    tasks = gen.tasks or []
    produce = tasks[0] if tasks else None
    return produce is None or produce.done()


def _turn_ready(session) -> bool:
    """True when close() would not cancel an in-flight tool body or the first audio write.

    An uncommitted reply never runs tools. A committed one is ready once its produce
    task has finished (every tool_result is out) and the first ``send_audio`` has
    returned (``gen.spoken``), or synthesis has ended without audio.

    ``first_audio_sent`` flips before that await. A chunk still in the write is not ready.
    """
    gen = getattr(session, "_gen", None)
    if gen is None or not gen.committed or gen.cancelled:
        return True
    tasks = gen.tasks or []
    produce = tasks[0] if tasks else None
    if produce is not None and not produce.done():
        return False
    if gen.spoken:
        return True
    drain = tasks[2] if len(tasks) > 2 else None
    return drain is None or drain.done()


async def _await_turn(session) -> bool:
    """Let tools return and the first TTS write finish before close() cancels them.

    Yields alone never finish a real ``asyncio.sleep`` in the tool or synthesizer.
    While that wait runs, the rig clock (``.t``) advances so a later sentence blocked
    on ``buffered_ms`` / ``tts_poll_s`` cannot spin until the cap.

    Returns False when the cap expires with the committed produce task still running.
    """
    for _ in range(100):
        if _turn_ready(session):
            return True
        await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TURN_WAIT_S
    while not _turn_ready(session):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return _produce_settled(session)
        await asyncio.sleep(min(_TURN_POLL_S, remaining))
        if _turn_ready(session):
            return True
        _advance_clock(session, FRAME_MS)
    return True


async def _await_pending_eot(session, seen: set[asyncio.Task]) -> None:
    """Hold the next silence frame so a verdict can move the end target.

    The tail is otherwise ``sleep(0)``, so ``end_ms`` commits and resets the turn
    before a real await in ``eot_async`` can call ``apply_eot``. One cap per task:
    if it is still pending, later frames continue and ``end_ms`` stands. This wait
    does not feed silence and does not move a rig clock.
    """
    task = getattr(session, "_eot_task", None)
    if task is None or task.done() or task in seen:
        return
    seen.add(task)
    await asyncio.wait({task}, timeout=_TURN_WAIT_S)


async def _drive(session, audio: np.ndarray) -> bool:
    seen: set[asyncio.Task] = set()
    for frame in _frames(audio, _tail_frames(session)):
        await session.feed(frame)
        for _ in range(_SETTLE_YIELDS):
            await asyncio.sleep(0)
        await _await_pending_eot(session, seen)
        _advance_clock(session, FRAME_MS)
    return await _await_turn(session)


def _open_tool_calls(events: list[dict]) -> list[str]:
    """tool_call names with no later tool_result. close() can cancel the body in between."""
    pending: list[str | None] = []
    for event in events:
        name = event.get("name")
        if not isinstance(name, str) or not name:
            continue
        kind = event.get("type")
        if kind == "tool_call":
            pending.append(name)
        elif kind == "tool_result":
            for i, open_name in enumerate(pending):
                if open_name == name:
                    pending[i] = None
                    break
    return [name for name in pending if name]


def _score(case: Case, events: list[dict], *, produce_settled: bool = True) -> dict:
    # tool_call is emitted before the body runs. close() can cancel it; only a
    # tool_result means that call finished. A call still open at the cap is unfinished:
    # a forbidden name is a wrong action, and the case is not a success.
    tools = [e["name"] for e in events if e.get("type") == "tool_result" and e.get("name")]
    unfinished = _open_tool_calls(events)
    forbidden = set(case.forbid_tools)
    wrong = [name for name in tools if name in forbidden]
    wrong.extend(name for name in unfinished if name in forbidden)
    interruptions = sum(1 for e in events if e.get("type") == "user_interrupted")
    # agent_first_audio is emitted before send_audio returns. Without agent_speech the
    # write was cancelled and the latency is not real.
    first = None
    if any(e.get("type") == "agent_speech" for e in events):
        first = next((e.get("latency_ms") for e in events if e.get("type") == "agent_first_audio"), None)
    tools_ok = (all(name in tools for name in case.expect_tools) and not wrong and not unfinished
                and produce_settled)
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
    settled = True
    try:
        settled = await _drive(session, case.audio)
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            await close()
    return _score(case, events, produce_settled=settled)


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
