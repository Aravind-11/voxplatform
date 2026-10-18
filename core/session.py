"""Per-call session orchestrator: VAD -> turn manager -> Ultravox -> chunker -> TTS -> transport.

Design points
  * Speculative generation: a reply starts on a short pause (SPECULATE) but its audio is held
    until END_OF_TURN commits it; SPEECH_RESUME cancels it. Tool calls also wait for commit so a
    cancelled speculation can never cause a side effect.
  * Barge-in: caller speech during playback clears the far-end buffer, cancels generation and
    trims history to the text the caller actually heard.
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
from .history import estimate_messages, fit_history
from .turn import Ev, TurnConfig, TurnEvent, TurnManager

FRAME_SAMPLES = 512   # 32 ms at 16 kHz (Silero VAD frame)
IN_RATE = 16000
# Not a string: a model sentence that equals the apology must not count as this attempt.
_FALLBACK = object()


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
    error_phrase: str = "Sorry, could you say that again?"  # spoken once if a committed reply fails
    turn: TurnConfig = field(default_factory=TurnConfig)
    max_context_tokens: int = 6000     # model context budget; the stored transcript is not trimmed to this
    history_tail: int = 6              # recent messages kept verbatim when the budget replaces older turns


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
        self.fallback_used = False     # the spoken apology is claimed at most once per reply
        self.fallback_spoken = False   # synthesis of the fallback sentinel has started
        self.sentences: list[str] = [] # model sentences put on textq, in order
        self.uncommitted_tts_failed = False  # speculative TTS died; later ordinary sentences will not play
        self.after: Optional[ToolResult] = None   # deferred call action requested by a tool this reply


class Session:
    def __init__(self, *, llm: LLM, tts: TTS, vad: VAD, send_audio: SendAudio,
                 clear_audio: ClearAudio, clock: Callable[[], float],
                 cfg: SessionConfig | None = None, tools: dict[str, Tool] | None = None,
                 on_event: OnEvent | None = None,
                 eot_fn: Optional[Callable[[np.ndarray, list[dict]], float]] = None,
                 eot_async: Optional[Callable[[np.ndarray, list[dict]], Awaitable[float]]] = None,
                 recorder: Any = None,
                 on_utterance: Optional[Callable[[np.ndarray, str, Optional[float]], None]] = None,
                 vad_scheduler: Any = None):
        self.llm, self.tts, self.vad = llm, tts, vad
        self.vad_scheduler = vad_scheduler    # None: score inline so unit rigs stay deterministic
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
        self._turn_lock = asyncio.Lock()  # feed holds this across its score so the frame is applied first
        self.stats = {"turns": 0, "speculations": 0, "speculations_cancelled": 0, "interruptions": 0}

    # ------------------------------------------------------------------ input
    async def feed(self, frame: np.ndarray) -> None:
        """Feed one 512-sample 16 kHz int16 frame (32 ms)."""
        # Scoring awaits. Keep the lock until this frame is applied so a verdict, inject_text,
        # or say that becomes runnable during the score cannot commit or barge ahead of it.
        async with self._turn_lock:
            if self.recorder is not None:
                self.recorder.mic(frame)
            if self.vad_scheduler is not None:
                prob = await self.vad_scheduler.prob(self.vad, frame)
            else:
                prob = self.vad.prob(frame)   # no scheduler: stay on this task (Rig tests depend on it)
            if prob >= self.cfg.turn.threshold:
                self._last_speech_clock = self.clock()
            self.turn.set_agent_speaking(self._agent_active())
            (self._utt if self.turn.in_turn else self._pre).append(frame)
            for ev in self.turn.process(prob):
                await self._handle(ev)

    async def start(self) -> None:
        """Called by the transport once the call is connected: speaks the configured greeting, if any."""
        if self.cfg.greeting.strip():
            await self.say(self.cfg.greeting.strip())

    async def say(self, text: str) -> None:
        """Speaks fixed text with no model call. It is a normal committed reply: the caller can talk over it
        (barge-in) and the history keeps only what they heard."""
        async with self._turn_lock:
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
        await self.say(text)   # lock covers only the install; playback below must stay interruptible
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
        async with self._turn_lock:
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
                await self._interrupt(g)
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
        async with self._turn_lock:
            # A speech frame already in feed cancels this task; if it was replaced, do not commit.
            if asyncio.current_task() is not self._eot_task:
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
        # Replaced each round so the short first-clause split stays per round. The name
        # lives outside the loop so a model error can flush the round that failed.
        chunker = SentenceChunker()
        try:
            if self._transcript_task and not self._transcript_task.done():
                await self._transcript_task  # previous turn's transcript must be in history
                gen.base_history = list(self.history[: len(gen.base_history)])
            for _round in range(self.cfg.max_tool_rounds + 1):
                chunker = SentenceChunker()
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                t_req, first_tok = self.clock(), True
                # Fitted view for the model only. self.history (and the store transcript) stay complete.
                # This turn's tool call and result stay whole, so the history view gives up the tokens they take.
                room = self.cfg.max_context_tokens - estimate_messages(followups)
                model_history = fit_history(gen.base_history, room, self.cfg.history_tail)
                async for item in self.llm.stream(model_history, gen.audio, followups,
                                                  self.cfg.tools_schema):
                    if isinstance(item, ToolCall):
                        calls.append(item)
                        continue
                    if first_tok:
                        first_tok = False
                        self._emit("llm_first_token", ms=self.clock() - t_req)
                    text_parts.append(item)
                    for piece in chunker.push(item):
                        gen.sentences.append(piece)
                        gen.textq.put_nowait(piece)
                for piece in chunker.flush():
                    gen.sentences.append(piece)
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
            # push() holds a finished sentence until one more character arrives. A clean
            # finish flushes it; an error must too. A fragment with no sentence ending stays dropped.
            for piece in chunker.flush_finished():
                gen.sentences.append(piece)
                gen.textq.put_nowait(piece)
            # Only a committed reply owes the caller an apology. A speculative reply that is
            # cancelled, or that fails before the turn commits, must stay silent.
            if gen.committed and not gen.cancelled and not gen.fallback_used:
                gen.fallback_used = True
                # The model already closed with this sentence and that audio will still play.
                # Queue the sentinel only when it will not, including after a speculative TTS
                # failure that drops the ordinary sentences.
                if not self._closing_phrase_will_play(gen):
                    gen.textq.put_nowait(_FALLBACK)
        finally:
            gen.textq.put_nowait(None)

    def _closing_phrase_will_play(self, gen: _Generation) -> bool:
        """True when the caller will already hear the apology as the reply's last line.

        A TTS failure before commit drops ordinary sentences, so a matching tail is not heard
        and must not swallow the real fallback."""
        if gen.uncommitted_tts_failed or gen.cancelled:
            return False
        phrase = self.cfg.error_phrase
        if gen.sentences and gen.sentences[-1] == phrase:
            return True
        pending = len(gen.sentences) - len(gen.spoken)
        return bool(gen.spoken) and gen.spoken[-1] == phrase and pending <= 0

    async def _synth(self, gen: _Generation) -> None:
        """Text -> audio, just in time. Only the first sentence is synthesized eagerly (so it is
        ready the instant the turn commits); later sentences wait for the commit and until the
        caller is within `tts_lookahead_ms` of running out of audio. This keeps TTS demand at about
        one sentence per active call instead of a burst of whole replies, and cancelled
        speculative replies never cost more than one sentence."""
        idx = 0
        try:
            while True:
                item = await gen.textq.get()
                if item is None:
                    return
                is_fallback = item is _FALLBACK
                if is_fallback:
                    if gen.fallback_spoken or gen.cancelled:
                        continue
                    await gen.commit.wait()
                    if gen.cancelled or not gen.committed or gen.fallback_spoken:
                        continue
                    gen.fallback_used = True
                    gen.fallback_spoken = True
                    text = self.cfg.error_phrase
                else:
                    text = item
                    if idx > 0:
                        await gen.commit.wait()
                        while self.tracker.buffered_ms() + gen.pending_ms > self.cfg.tts_lookahead_ms:
                            await asyncio.sleep(self.cfg.tts_poll_s)
                try:
                    await self._speak(gen, text)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    await self._on_tts_failure(gen, is_fallback, e)
                    return
                if not is_fallback:
                    idx += 1
        except asyncio.CancelledError:
            raise
        finally:
            gen.produce_done = True
            gen.outbox.put_nowait(None)

    async def _on_tts_failure(self, gen: _Generation, is_fallback: bool, e: Exception) -> None:
        """One apology for a committed TTS failure. The apology's own failure is not retried.

        A failure before commit must not end this task. Drain would take the terminal outbox
        mark, and a model error after commit would queue an apology nobody reads. On cancel,
        stay silent. On commit, speak a queued apology once and drop ordinary sentences: a
        successful reply whose speculative TTS died stays silent, unless a deferred hang-up
        or transfer would otherwise run against that silence. A failure after commit also
        stays until produce closes textq: the tool may still be awaiting, and ending here
        would drop its deferred action and leave that produce task running.
        """
        if is_fallback:
            self._emit("error", where="tts_fallback", error=repr(e))
            return
        if not gen.committed and not gen.cancelled:
            gen.uncommitted_tts_failed = True
        self._emit("error", where="tts", error=repr(e))
        if gen.cancelled:
            return
        if gen.uncommitted_tts_failed:
            await self._play_queued_fallback(gen)
            return
        await self._speak_fallback(gen)
        await self._skip_until_textq_closes(gen)

    async def _skip_until_textq_closes(self, gen: _Generation) -> None:
        """Drop later sentences until produce finishes. outbox closes only after this returns."""
        while not gen.cancelled:
            item = await gen.textq.get()
            if item is None:
                return
            if item is _FALLBACK:
                await self._speak_fallback(gen)

    async def _play_queued_fallback(self, gen: _Generation) -> None:
        await gen.commit.wait()
        if gen.cancelled or not gen.committed:
            return
        while not gen.cancelled:
            item = await gen.textq.get()
            if item is None:
                # textq closes from produce's finally, which is after a tool may have set
                # `after`. With nothing sent, drain would hang up or transfer on dead air.
                if gen.after is not None and not gen.spoken:
                    await self._speak_fallback(gen)
                return
            if item is _FALLBACK:
                await self._speak_fallback(gen)

    async def _speak_fallback(self, gen: _Generation) -> None:
        """Synthesize the apology directly. A second failure is reported and not retried."""
        if gen.fallback_spoken or not gen.committed or gen.cancelled:
            return
        gen.fallback_used = True
        gen.fallback_spoken = True
        try:
            await self._speak(gen, self.cfg.error_phrase)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="tts_fallback", error=repr(e))

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

    async def _speak(self, gen: _Generation, text: str) -> None:
        t_tts = self.clock()
        parts = [p async for p in self.tts.synthesize(text)]
        self._emit("tts_done", ms=self.clock() - t_tts, chars=len(text))
        pcm = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        pcm = resample(pcm, self.tts.sample_rate, self.cfg.out_rate)
        gen.pending_ms += pcm_duration_ms(len(pcm), self.cfg.out_rate)
        await gen.outbox.put((text, pcm))

    async def _drain(self, gen: _Generation) -> None:
        await gen.commit.wait()
        while True:
            item = await gen.outbox.get()
            if item is None:
                if gen.after is not None and not gen.cancelled:
                    await self._finish_action(gen)
                return
            text, pcm = item
            if not gen.first_audio_sent:
                gen.first_audio_sent = True
                self._emit("agent_first_audio",
                           latency_ms=self.clock() - gen.speech_end_clock)
            await self.send_audio(pcm, self.cfg.out_rate)
            if self.recorder is not None:
                self.recorder.agent(pcm, self.cfg.out_rate)
            dur = pcm_duration_ms(len(pcm), self.cfg.out_rate)
            self.tracker.on_sent(text, dur)
            gen.pending_ms -= dur
            gen.spoken.append(text)
            self._emit("agent_speech", text=text)
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["content"] = " ".join(gen.spoken)

    # ----------------------------------------------------------- barge-in
    async def _interrupt(self, gen: _Generation) -> None:
        heard, cut = self.tracker.heard()
        await self.clear_audio()          # stop the far end first: that is what the caller hears
        if self.recorder is not None:
            self.recorder.clear()
        await self._cancel(gen)
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
