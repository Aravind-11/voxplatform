"""Per-call session orchestrator: VAD -> turn manager -> Ultravox -> chunker -> TTS -> transport.

Design points
  * Speculative generation: a reply starts on a short pause (SPECULATE) but its audio is held
    until END_OF_TURN commits it; SPEECH_RESUME cancels it. Tool calls also wait for commit so a
    cancelled speculation can never cause a side effect.
  * Barge-in: caller speech during playback pauses the far end at once. If the burst
    ends before barge_confirm_ms it was a cough: playback resumes and history is not
    trimmed. If speech keeps going, that pause becomes a real interrupt.
  * The speech model emits text only and the server accepts one audio clip per request, so
    earlier caller turns live in history as transcripts (produced by a side request that runs in
    parallel with the reply and is awaited before the next turn).
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol

import numpy as np

from .audio import resample, pcm_duration_ms
from .barge import PlaybackTracker, trim_history
from .chunker import SentenceChunker
from .turn import Ev, TurnConfig, TurnEvent, TurnManager

FRAME_SAMPLES = 512   # 32 ms at 16 kHz (Silero VAD frame)
IN_RATE = 16000


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


class VAD(Protocol):
    def prob(self, frame: np.ndarray) -> float: ...


class LLM(Protocol):
    def stream(self, history: list[dict], audio: np.ndarray, followups: list[dict],
               tools: list[dict]) -> AsyncIterator[str | ToolCall]: ...

    async def transcribe(self, audio: np.ndarray) -> str: ...


class TTS(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]: ...


@dataclass
class ToolResult:
    """What a tool returns when it also wants to change the call (hang up, transfer).
    `content` goes back to the model. `after` runs once, only after the agent has finished saying its
    reply, so a goodbye is spoken in full before the line drops. strict=True: runs even if the caller
    talks over that reply; strict=False: the caller interrupting cancels it."""
    content: Any
    after: Optional[Callable[[], Awaitable[None]]] = None
    strict: bool = True


Tool = Callable[[dict], Awaitable[Any]]
SendAudio = Callable[[np.ndarray, int], Awaitable[None]]
ClearAudio = Callable[[], Awaitable[None]]
OnEvent = Callable[[dict], None]


@dataclass
class SessionConfig:
    system_prompt: str = "You are a helpful voice assistant. Keep replies short and conversational."
    out_rate: int = 8000
    pre_roll_frames: int = 10
    max_tool_rounds: int = 3
    output_latency_ms: float = 0.0
    tts_lookahead_ms: float = 1500.0   # synthesize sentence k+1 only when < this much audio is queued
    tts_poll_s: float = 0.02           # how often a waiting sentence re-checks the playback buffer
    tools_schema: list[dict] = field(default_factory=list)
    greeting: str = ""                 # spoken by the agent as soon as the call connects (no model call)
    turn: TurnConfig = field(default_factory=TurnConfig)


class _Generation:
    def __init__(self, audio: np.ndarray, base_history: list[dict], speech_end_clock: float):
        self.audio = audio
        self.base_history = base_history
        self.speech_end_clock = speech_end_clock
        self.commit = asyncio.Event()
        self.committed = False
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.textq: asyncio.Queue = asyncio.Queue()
        self.pending_ms = 0.0          # synthesized audio waiting in outbox (not yet sent)
        self.tasks: list[asyncio.Task] = []
        self.spoken: list[str] = []
        self.produce_done = False
        self.first_audio_sent = False
        self.cancelled = False
        self.holding = False           # tentative barge-in: drain must not send until this clears
        self.barge_paused = False
        self.paused_heard = ""
        self.paused_cut = False
        self.paused_spoken: list[str] = []
        self.spoken_mark = 0           # len(spoken) when a tentative barge-in began
        self.sending = False           # drain is inside send_audio; pause waits so clear cannot race it
        self.after: Optional[ToolResult] = None   # deferred call action requested by a tool this reply


class Session:
    def __init__(self, *, llm: LLM, tts: TTS, vad: VAD, send_audio: SendAudio,
                 clear_audio: ClearAudio, clock: Callable[[], float],
                 cfg: SessionConfig | None = None, tools: dict[str, Tool] | None = None,
                 on_event: OnEvent | None = None,
                 eot_fn: Optional[Callable[[np.ndarray, list[dict]], float]] = None,
                 eot_async: Optional[Callable[[np.ndarray, list[dict]], Awaitable[float]]] = None,
                 recorder: Any = None,
                 on_utterance: Optional[Callable[[np.ndarray, str, Optional[float]], None]] = None):
        self.llm, self.tts, self.vad = llm, tts, vad
        self.send_audio, self.clear_audio = send_audio, clear_audio
        self.clock = clock
        self.cfg = cfg or SessionConfig()
        self.tools = tools or {}
        self.on_event = on_event or (lambda e: None)
        self.eot_fn = eot_fn
        self.eot_async = eot_async
        self.recorder = recorder              # optional core.recorder.CallRecorder: mic(frame), agent(pcm, rate), clear()
        self.on_utterance = on_utterance      # optional dataset hook: (audio16k, transcript, eot_score)
        self._last_eot: Optional[float] = None
        self._eot_task: Optional[asyncio.Task] = None
        self.turn = TurnManager(self.cfg.turn, eot_scorer=self._eot_score if eot_fn else None)
        self.tracker = PlaybackTracker(clock, self.cfg.output_latency_ms)
        self.history: list[dict] = [{"role": "system", "content": self.cfg.system_prompt}]
        self._pre: deque[np.ndarray] = deque(maxlen=self.cfg.pre_roll_frames)
        self._utt: list[np.ndarray] = []
        self._gen: Optional[_Generation] = None
        self._last_speech_clock = 0.0
        self._transcript_task: Optional[asyncio.Task] = None
        self._bg: set[asyncio.Task] = set()
        self.stats = {"turns": 0, "speculations": 0, "speculations_cancelled": 0, "interruptions": 0}

    # ------------------------------------------------------------------ input
    async def feed(self, frame: np.ndarray) -> None:
        """Feed one 512-sample 16 kHz int16 frame (32 ms)."""
        if self.recorder is not None:
            self.recorder.mic(frame)
        prob = self.vad.prob(frame)
        if prob >= self.cfg.turn.threshold:
            self._last_speech_clock = self.clock()
        self.turn.set_agent_speaking(self._agent_active())
        (self._utt if self.turn.in_turn else self._pre).append(frame)
        for ev in self.turn.process(prob):
            await self._handle(ev)
        g = self._gen
        # barge_confirm_ms is measured from the start of the utterance, not from the pause.
        if g and g.holding and not g.cancelled and self.turn.speech_ms >= self.cfg.turn.barge_confirm_ms:
            await self._interrupt(g)

    async def start(self) -> None:
        """Called by the transport once the call is connected: speaks the configured greeting, if any."""
        if self.cfg.greeting.strip():
            self.say(self.cfg.greeting.strip())

    def say(self, text: str) -> None:
        """Speaks fixed text with no model call. It is a normal committed reply: the caller can talk over it
        (barge-in) and the history keeps only what they heard."""
        gen = _Generation(np.zeros(0, np.int16), list(self.history), self.clock())
        gen.committed = True
        gen.commit.set()
        self.history.append({"role": "assistant", "content": ""})

        async def produce() -> None:
            chunker = SentenceChunker()
            for piece in [*chunker.push(text), *chunker.flush()]:
                gen.textq.put_nowait(piece)
            gen.textq.put_nowait(None)

        gen.tasks = [asyncio.create_task(produce()), asyncio.create_task(self._synth(gen)),
                     asyncio.create_task(self._drain(gen))]
        self._gen = gen
        self._emit("greeting", text=text)

    async def say_and_wait(self, text: str, timeout_s: float = 60.0) -> None:
        """Speaks fixed text and returns once it has finished playing (or the caller talked over it)."""
        self.say(text)
        gen = self._gen
        assert gen is not None
        deadline = self.clock() + timeout_s * 1000.0
        await asyncio.wait({gen.tasks[-1]}, timeout=timeout_s)         # the drain task: every sentence sent
        while not gen.cancelled and self.tracker.buffered_ms() > 0 and self.clock() < deadline:
            await asyncio.sleep(0.05)

    def seed_history(self, messages: list[dict]) -> None:
        """Restores earlier turns ({"role": "user"|"agent", "text"}) before the call starts, so a call that was
        reconnected after a failure continues the same conversation."""
        for m in messages:
            if m.get("text"):
                self.history.append({"role": "assistant" if m["role"] == "agent" else "user", "content": m["text"]})

    async def inject_text(self, text: str) -> None:
        """A caller turn that arrived as text (keypad digits): interrupts the agent like speech would, then replies."""
        g = self._gen
        if g and g.committed and self._agent_active():
            await self._interrupt(g)
        elif g and not g.committed:
            await self._cancel(g)
            self._gen = None
        self.stats["turns"] += 1
        self.history.append({"role": "user", "content": text})
        base = list(self.history)
        self.history.append({"role": "assistant", "content": ""})
        gen = _Generation(np.zeros(0, np.int16), base, self.clock())
        gen.committed = True
        gen.commit.set()
        gen.tasks = [asyncio.create_task(self._produce(gen)), asyncio.create_task(self._synth(gen)),
                     asyncio.create_task(self._drain(gen))]
        self._gen = gen
        self._emit("user_transcript", text=text, source="text")

    async def close(self) -> None:
        self._cancel_eot()
        if self._gen:
            await self._cancel(self._gen)
        if self._transcript_task:
            await asyncio.gather(self._transcript_task, return_exceptions=True)

    # --------------------------------------------------------------- events
    def _emit(self, type_: str, **data: Any) -> None:
        self.on_event({"type": type_, "t": self.clock(), **data})

    async def _handle(self, ev: TurnEvent) -> None:
        if ev.kind is Ev.SPEECH_START:
            self._utt = list(self._pre)
            self._pre.clear()
            g = self._gen
            if g and g.committed and self._agent_active():
                await self._pause_for_barge(g)
            self._emit("user_speech_start")
        elif ev.kind is Ev.SPECULATE:
            self.stats["speculations"] += 1
            self._start_generation()
            self._emit("speculate_start")
            self._start_eot()
        elif ev.kind is Ev.SPEECH_RESUME:
            self._cancel_eot()
            if self._gen and not self._gen.committed:
                self.stats["speculations_cancelled"] += 1
                await self._cancel(self._gen)
                self._gen = None
                self._emit("speculate_cancelled")
        elif ev.kind is Ev.END_OF_TURN:
            self._cancel_eot()
            self.stats["turns"] += 1
            self._emit("user_end_of_turn")
            if self._gen and not self._gen.committed:
                self._commit(self._gen)
        elif ev.kind is Ev.SPEECH_ABORT:
            self._cancel_eot()
            self._utt = []
            self._emit("speech_abort")
            g = self._gen
            if g and g.holding:
                await self._resume_after_barge(g)

    def _cancel_eot(self) -> None:
        t, self._eot_task = self._eot_task, None
        if t and not t.done() and t is not asyncio.current_task():
            t.cancel()

    def _start_eot(self) -> None:
        """Ask the async scorer whether the caller is done; its verdict can shorten (or lengthen) the pause."""
        if self.eot_async is None:
            return
        audio = np.concatenate(self._utt) if self._utt else np.zeros(0, np.int16)
        history = list(self.history)
        self._cancel_eot()
        self._eot_task = asyncio.create_task(self._eot_verdict(audio, history))

    async def _eot_verdict(self, audio: np.ndarray, history: list[dict]) -> None:
        t0 = self.clock()
        try:
            score = float(await self.eot_async(audio, history))  # type: ignore[misc]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="eot", error=repr(e))
            return
        self._last_eot = score
        self._emit("eot_score", score=round(score, 3), ms=self.clock() - t0)
        for ev in self.turn.apply_eot(score):
            await self._handle(ev)

    def _eot_score(self) -> float:
        audio = np.concatenate(self._utt) if self._utt else np.zeros(0, np.int16)
        return float(self.eot_fn(audio, self.history))  # type: ignore[misc]

    def _agent_active(self) -> bool:
        g = self._gen
        return bool(g and g.committed and not g.cancelled
                    and (not g.produce_done or self.tracker.is_playing()))

    # ----------------------------------------------------------- generation
    def _start_generation(self) -> None:
        audio = np.concatenate(self._utt) if self._utt else np.zeros(0, np.int16)
        gen = _Generation(audio, list(self.history), self._last_speech_clock)
        gen.tasks = [asyncio.create_task(self._produce(gen)), asyncio.create_task(self._synth(gen)),
                     asyncio.create_task(self._drain(gen))]
        self._gen = gen

    def _commit(self, gen: _Generation) -> None:
        gen.committed = True
        self.history.append({"role": "user", "content": "[caller audio]"})
        self.history.append({"role": "assistant", "content": ""})
        user_idx = len(self.history) - 2
        self._transcript_task = asyncio.create_task(self._transcribe(gen.audio, user_idx))
        gen.commit.set()

    async def _transcribe(self, audio: np.ndarray, idx: int) -> None:
        try:
            text = (await self.llm.transcribe(audio)).strip() or "[inaudible]"
        except Exception as e:  # noqa: BLE001
            text = "[inaudible]"
            self._emit("error", where="transcribe", error=repr(e))
        self.history[idx]["content"] = text
        self._emit("user_transcript", text=text)
        if self.on_utterance is not None:
            try:
                self.on_utterance(audio, text, self._last_eot)
            except Exception as e:  # noqa: BLE001  (a dataset write must never break a call)
                self._emit("error", where="utterance_sink", error=repr(e))

    async def _produce(self, gen: _Generation) -> None:
        followups: list[dict] = []
        try:
            if self._transcript_task and not self._transcript_task.done():
                await self._transcript_task  # previous turn's transcript must be in history
                gen.base_history = list(self.history[: len(gen.base_history)])
            for _round in range(self.cfg.max_tool_rounds + 1):
                chunker = SentenceChunker()
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                t_req, first_tok = self.clock(), True
                async for item in self.llm.stream(gen.base_history, gen.audio, followups,
                                                  self.cfg.tools_schema):
                    if isinstance(item, ToolCall):
                        calls.append(item)
                        continue
                    if first_tok:
                        first_tok = False
                        self._emit("llm_first_token", ms=self.clock() - t_req)
                    text_parts.append(item)
                    for piece in chunker.push(item):
                        gen.textq.put_nowait(piece)
                for piece in chunker.flush():
                    gen.textq.put_nowait(piece)
                if not calls:
                    break
                await gen.commit.wait()  # never run side effects for an uncommitted reply
                followups.append({"role": "assistant", "content": "".join(text_parts) or None,
                                  "tool_calls": [{"id": c.id, "type": "function",
                                                  "function": {"name": c.name,
                                                               "arguments": json.dumps(c.arguments)}}
                                                 for c in calls]})
                for c in calls:
                    self._emit("tool_call", name=c.name, arguments=c.arguments)
                    t_tool = self.clock()
                    out = await self._run_tool(c, gen)
                    self._emit("tool_result", name=c.name, ms=self.clock() - t_tool, chars=len(out))
                    followups.append({"role": "tool", "tool_call_id": c.id, "content": out})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="produce", error=repr(e))
        finally:
            gen.textq.put_nowait(None)

    async def _synth(self, gen: _Generation) -> None:
        """Text -> audio, just in time. Only the first sentence is synthesized eagerly (so it is
        ready the instant the turn commits); later sentences wait for the commit and until the
        caller is within `tts_lookahead_ms` of running out of audio. This keeps TTS demand at about
        one sentence per active call instead of a burst of whole replies, and cancelled
        speculative replies never cost more than one sentence."""
        idx = 0
        try:
            while True:
                # A tentative barge-in must keep the sentence already pulled off textq. Queueing
                # more audio while the drain is held lets the cleared tail be prepended in front.
                while gen.holding and not gen.cancelled:
                    await asyncio.sleep(self.cfg.tts_poll_s)
                if gen.cancelled:
                    return
                text = await gen.textq.get()
                if text is None:
                    return
                if idx > 0:
                    await gen.commit.wait()
                    while not gen.cancelled and (
                            gen.holding
                            or self.tracker.buffered_ms() + gen.pending_ms > self.cfg.tts_lookahead_ms):
                        await asyncio.sleep(self.cfg.tts_poll_s)
                if gen.cancelled:
                    return
                await self._speak(gen, text)
                idx += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="tts", error=repr(e))
        finally:
            gen.produce_done = True
            gen.outbox.put_nowait(None)

    async def _run_tool(self, call: ToolCall, gen: Optional[_Generation] = None) -> str:
        fn = self.tools.get(call.name)
        if fn is None:
            return json.dumps({"error": f"unknown tool {call.name}"})
        try:
            r = await fn(call.arguments)
            if isinstance(r, ToolResult):
                if r.after is not None and gen is not None:
                    gen.after = r
                r = r.content
            return json.dumps(r, default=str)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": repr(e)})

    def _spawn_action(self, a: ToolResult) -> None:
        t = asyncio.create_task(self._run_action(a))      # own task: ending the call cancels the generation's tasks
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _run_action(self, a: ToolResult) -> None:
        try:
            await a.after()   # type: ignore[misc]
            self._emit("call_action_done")
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="call_action", error=repr(e))

    async def _finish_action(self, gen: _Generation) -> None:
        deadline = self.clock() + 30000.0
        while self.tracker.buffered_ms() > 0 and self.clock() < deadline:   # the reply must finish playing first
            await asyncio.sleep(0.05)
        a, gen.after = gen.after, None      # taken only now: a barge-in during the wait still sees it
        if a is not None:
            self._spawn_action(a)

    async def _render(self, text: str) -> tuple[np.ndarray, float]:
        t_tts = self.clock()
        parts = [p async for p in self.tts.synthesize(text)]
        self._emit("tts_done", ms=self.clock() - t_tts, chars=len(text))
        pcm = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        pcm = resample(pcm, self.tts.sample_rate, self.cfg.out_rate)
        return pcm, pcm_duration_ms(len(pcm), self.cfg.out_rate)

    async def _speak(self, gen: _Generation, text: str) -> None:
        pcm, dur = await self._render(text)
        gen.pending_ms += dur
        await gen.outbox.put((text, pcm))

    def _requeue_front(self, gen: _Generation, item: Any) -> None:
        rest = []
        while True:
            try:
                rest.append(gen.outbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        gen.outbox.put_nowait(item)
        for queued in rest:
            gen.outbox.put_nowait(queued)

    async def _drain(self, gen: _Generation) -> None:
        await gen.commit.wait()
        while True:
            # A tentative barge-in holds the buffer. A strict hang-up must not run until the
            # interrupt is confirmed (or the hold is released); a cough must not drop the call.
            while gen.holding and not gen.cancelled:
                await asyncio.sleep(self.cfg.tts_poll_s)
            if gen.cancelled:
                return
            item = await gen.outbox.get()
            if gen.holding and not gen.cancelled:
                # Already taken off the queue: put it back so the cleared tail can be prepended.
                self._requeue_front(gen, item)
                continue
            if gen.cancelled:
                return
            if item is None:
                if gen.after is not None and not gen.cancelled:
                    await self._finish_action(gen)
                return
            replay = False
            if isinstance(item, tuple) and len(item) == 3:
                text, pcm, replay = item
            else:
                text, pcm = item
            if not gen.first_audio_sent:
                gen.first_audio_sent = True
                self._emit("agent_first_audio",
                           latency_ms=self.clock() - gen.speech_end_clock)
            gen.sending = True
            try:
                await self.send_audio(pcm, self.cfg.out_rate)
                if self.recorder is not None:
                    self.recorder.agent(pcm, self.cfg.out_rate)
                dur = pcm_duration_ms(len(pcm), self.cfg.out_rate)
                self.tracker.on_sent(text, dur)
                gen.pending_ms -= dur
                # A replay is audio clear() threw away. spoken/history already include it.
                if not replay:
                    gen.spoken.append(text)
                    if self.history and self.history[-1]["role"] == "assistant":
                        self.history[-1]["content"] = " ".join(gen.spoken)
                self._emit("agent_speech", text=text)
            finally:
                gen.sending = False

    # ----------------------------------------------------------- barge-in
    async def _pause_for_barge(self, gen: _Generation) -> None:
        """Stop the far end without cancelling the reply. Speech may still turn out to be noise."""
        if gen.holding or gen.cancelled:
            return
        heard, cut = self.tracker.heard()
        gen.barge_paused = True
        gen.paused_heard = heard
        gen.paused_cut = cut
        gen.paused_spoken = list(gen.spoken)
        gen.spoken_mark = len(gen.spoken)
        gen.holding = True                 # before the first await, so drain cannot send another chunk
        while gen.sending:
            await asyncio.sleep(0)
        await self.clear_audio()
        if self.recorder is not None:
            self.recorder.clear()

    def _outgoing_idle(self, gen: _Generation) -> bool:
        """Produce and synth will not enqueue anything else (the drain may still be in its end wait)."""
        if not gen.tasks:
            return True
        producers = gen.tasks[:-1] if len(gen.tasks) > 1 else gen.tasks
        if any(not t.done() for t in producers):
            return False
        return gen.outbox.empty() and gen.textq.empty()

    def _unheard_sent_text(self, gen: _Generation) -> str:
        """Words already sent (or sent after the pause snapshot) that clear() discarded.

        Does not include text still in the synth/outbox: that pipeline keeps running. Does not
        repeat the last sentence when the tracker says it was fully heard.
        """
        spoken = [s.strip() for s in gen.paused_spoken if s and s.strip()]
        heard = (gen.paused_heard or "").strip()
        joined = " ".join(spoken).strip()
        if not heard:
            rest = joined
        elif joined.startswith(heard):
            rest = joined[len(heard):].strip()
        else:
            h, w = heard.split(), joined.split()
            rest = " ".join(w[len(h):]) if h and w[: len(h)] == h else ""
        extra = [s.strip() for s in gen.spoken[gen.spoken_mark:] if s and s.strip()]
        if extra:
            rest = f"{rest} {' '.join(extra)}".strip() if rest else " ".join(extra)
        return rest

    async def _put_replay(self, gen: _Generation, text: str) -> None:
        pcm, dur = await self._render(text)
        gen.pending_ms += dur
        await gen.outbox.put((text, pcm, True))

    async def _prepend_replay_text(self, gen: _Generation, text: str) -> None:
        chunker = SentenceChunker()
        pieces = [p for p in (*chunker.push(text), *chunker.flush()) if p.strip()]
        built = []
        for piece in pieces:
            if gen.cancelled:
                return
            pcm, dur = await self._render(piece)
            built.append((piece, pcm, dur))
        if gen.cancelled:
            return
        rest = []
        while True:
            try:
                rest.append(gen.outbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        for piece, pcm, dur in built:
            gen.pending_ms += dur
            gen.outbox.put_nowait((piece, pcm, True))
        for item in rest:
            gen.outbox.put_nowait(item)

    def _start_tail_playback(self, gen: _Generation, text: str) -> None:
        """The reply already left the outbox. Speak the cleared tail without a second history turn."""
        gen.produce_done = False
        gen.cancelled = False

        async def produce() -> None:
            try:
                chunker = SentenceChunker()
                for piece in [*chunker.push(text), *chunker.flush()]:
                    if gen.cancelled:
                        return
                    await self._put_replay(gen, piece)
            finally:
                gen.produce_done = True
                if not gen.cancelled:
                    gen.outbox.put_nowait(None)

        gen.tasks = [asyncio.create_task(produce()),
                     asyncio.create_task(self._drain(gen))]

    async def _resume_after_barge(self, gen: _Generation) -> None:
        """A cough: do not trim history. Re-speak whatever clear() cut off, then continue the reply."""
        if gen.cancelled or not gen.holding:
            return
        tail = self._unheard_sent_text(gen)
        if self._outgoing_idle(gen):
            # Drain has finished or is only waiting to run a deferred hang-up. Cancelling that wait
            # keeps a strict hang-up from firing while the tail is still unheard; the new drain
            # reaches _finish_action after the tail.
            drain = gen.tasks[-1] if gen.tasks else None
            if drain is not None and not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
            self.tracker.commit_heard(gen.paused_heard)
            if tail:
                self._start_tail_playback(gen, tail)
            elif gen.after is not None:
                await self._finish_action(gen)
        else:
            self.tracker.commit_heard(gen.paused_heard)
            if tail:
                await self._prepend_replay_text(gen, tail)
        gen.holding = False
        gen.barge_paused = False
        self._emit("barge_aborted", text=tail)

    async def _interrupt(self, gen: _Generation) -> None:
        # After a pause the transport buffer is already gone, so the clock must not advance "heard".
        if gen.barge_paused:
            heard, cut = gen.paused_heard, gen.paused_cut
        else:
            heard, cut = self.tracker.heard()
        await self.clear_audio()          # stop the far end first: that is what the caller hears
        if self.recorder is not None:
            self.recorder.clear()
        await self._cancel(gen)
        gen.holding = False
        a, gen.after = gen.after, None
        if a is not None and a.strict:
            self._spawn_action(a)         # strict: the caller talking over the goodbye does not keep the call alive
        self.history = trim_history(self.history, heard, cut)
        self.tracker.reset()
        self.stats["interruptions"] += 1
        self._emit("user_interrupted", heard=heard, cut=cut)

    async def _cancel(self, gen: _Generation) -> None:
        gen.cancelled = True
        for t in gen.tasks:
            t.cancel()
        await asyncio.gather(*gen.tasks, return_exceptions=True)
