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
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol

import numpy as np

from .audio import resample, pcm_duration_ms
from .backchannel import is_backchannel
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


class _WordSlice:
    """Tracker text for one audio chunk. `heard()` calls `split` and stringifies, so filling `.text`
    after `on_sent` still fixes barge-in without holding the first chunk until synthesis finishes.

    `sentence` is the utterance this chunk belongs to. `covered` is set once a transcript marker
    or a prefix commit has already counted the chunk, so cancel does not record it twice.
    """

    def __init__(self, text: str = "", sentence: str = "") -> None:
        self.text = text
        self.sentence = sentence
        self.covered = False

    def split(self, sep=None, maxsplit=-1):
        return self.text.split() if sep is None else self.text.split(sep, maxsplit)

    def __str__(self) -> str:
        return self.text


def _fill_word_slices(slices: list[_WordSlice], lengths: list[int], sentence: str) -> None:
    """Give each chunk the words that fall in its share of the samples. One chunk keeps the sentence verbatim."""
    if len(slices) == 1:
        slices[0].text = sentence
        return
    words = sentence.split()
    total = sum(lengths) or 1
    prev = 0
    acc = 0
    last = len(slices) - 1
    for i, (sl, n) in enumerate(zip(slices, lengths)):
        acc += n
        k = len(words) if i == last else int(len(words) * acc / total)
        k = min(len(words), max(prev, k))
        sl.text = " ".join(words[prev:k])
        prev = k


def _claim_words_so_far(slices: list[_WordSlice], lengths: list[int], sentence: str, out_rate: int,
                        ms_per_word: float = 180.0) -> None:
    """Words a barge-in may claim before synthesis of this sentence has finished.

    Elapsed audio caps the claim at about one word per 180 ms. The chunk in progress does not
    receive the rest of the sentence: that would mark every word heard as soon as the first
    chunk reached the caller.
    """
    words = sentence.split()
    if not slices or not words:
        return
    elapsed = sum(pcm_duration_ms(n, out_rate) for n in lengths)
    n_words = min(len(words), int(elapsed // ms_per_word))
    total = sum(lengths) or 1
    prev = 0
    acc = 0
    last = len(slices) - 1
    for i, (sl, n) in enumerate(zip(slices, lengths)):
        acc += n
        k = n_words if i == last else int(n_words * acc / total)
        k = min(n_words, max(prev, k))
        sl.text = " ".join(words[prev:k])
        prev = k


@dataclass
class SessionConfig:
    system_prompt: str = "You are a helpful voice assistant. Keep replies short and conversational."
    out_rate: int = 8000
    pre_roll_frames: int = 10
    max_tool_rounds: int = 3
    output_latency_ms: float = 0.0
    mark_timeout_ms: float = 1500.0    # give up waiting for a playback mark this long after the clock estimate
    tts_lookahead_ms: float = 1500.0   # synthesize sentence k+1 only when < this much audio is queued
    tts_poll_s: float = 0.02           # how often a waiting sentence re-checks the playback buffer
    tools_schema: list[dict] = field(default_factory=list)
    greeting: str = ""                 # spoken by the agent as soon as the call connects (no model call)
    error_phrase: str = "Sorry, could you say that again?"  # spoken once if a committed reply fails
    filler_text: str = "One moment, let me check that."  # spoken when a tool call outlives filler_delay_s
    filler_delay_s: float = 0.7
    tool_fillers: dict[str, str] = field(default_factory=dict)  # per-tool line; overrides filler_text. Empty disables.
    turn: TurnConfig = field(default_factory=TurnConfig)
    max_context_tokens: int = 6000     # model context budget; the stored transcript is not trimmed to this
    history_tail: int = 6              # recent messages kept verbatim when the budget replaces older turns
    short_first_sentence: bool = True  # runtime asks for a short opener; Kokoro still cannot hear the caller


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
        self.outbox_audio = 0          # audio items still in outbox (not transcript markers)
        self.tasks: list[asyncio.Task] = []
        self.spoken: list[str] = []
        self.sent_slices: list[_WordSlice] = []  # slices whose send_audio has returned
        self.failed_sentences: set[str] = set()  # a later chunk of this sentence did not send
        self.produce_done = False
        self.first_audio_sent = False
        self.cancelled = False
        self.fallback_used = False     # the spoken apology is claimed at most once per reply
        self.fallback_spoken = False   # synthesis of the fallback sentinel has started
        self.sentences: list[str] = [] # model sentences put on textq, in order
        self.uncommitted_tts_failed = False  # speculative TTS died; later ordinary sentences will not play
        self.speak_lock = asyncio.Lock()  # filler audio and sentence audio must not synthesize at the same time
        self.holding_sentence = False     # dequeued by _synth but not spoken yet (off textq during lookahead)
        self.holding = False           # tentative barge-in: drain must not send until this clears
        self.barge_paused = False
        self.paused_heard = ""
        self.paused_cut = False
        self.paused_spoken: list[str] = []
        self.spoken_mark = 0           # len(spoken) when a tentative barge-in began
        self.sending = False           # drain is inside send_audio; pause waits so clear cannot race it
        self.after: Optional[ToolResult] = None   # deferred call action requested by a tool this reply
        # Transcript already running when this generation was created. Not this turn's task:
        # _commit may replace the session pointer before _produce is scheduled.
        self.prior_transcript: Optional[asyncio.Task] = None


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
        self.tracker = PlaybackTracker(clock, self.cfg.output_latency_ms,
                                       mark_timeout_ms=self.cfg.mark_timeout_ms)
        self.history: list[dict] = [{"role": "system", "content": self.cfg.system_prompt}]
        self._pre: deque[np.ndarray] = deque(maxlen=self.cfg.pre_roll_frames)
        self._utt: list[np.ndarray] = []
        self._gen: Optional[_Generation] = None
        self._last_speech_clock = 0.0
        self._transcript_task: Optional[asyncio.Task] = None
        self._transcript_tasks: set[asyncio.Task] = set()
        self._transcripts: list[tuple[int, asyncio.Task]] = []  # (history index, task), oldest first
        self._bg: set[asyncio.Task] = set()
        self._turn_lock = asyncio.Lock()  # feed holds this across its score so the frame is applied first
        self._overlapped = False          # this caller utterance started while the agent was playing
        self._overlap_unscored = 0        # interrupts waiting on a transcript before they count
        self._overlap_audio: list[np.ndarray] = []  # speech while the agent talks, before a turn exists
        self._collecting_overlap = False  # keep collecting after playback stops, until the run ends
        self._fed_speech = False
        self._overlap_token = 0           # id of an overlap check; a newer check does not retire an older one
        self._overlap_tasks: set[asyncio.Task] = set()
        self._overlap_live: dict[int, list[np.ndarray]] = {}
        self._overlap_claimed: dict[int, list[np.ndarray]] = {}
        self._overlap_dropped: set[int] = set()
        self._overlap_commit_lock = asyncio.Lock()  # one real overlap commits at a time
        self.stats = {"turns": 0, "speculations": 0, "speculations_cancelled": 0, "interruptions": 0}
        self._turn_owner = None

    async def _without_turn_lock(self, awaitable):
        """Drop the turn lock across a socket wait so keypad can enter on this task.

        feed holds the lock so a frame is applied before say or inject. Resume awaits the
        drain on that same task; inject_text has to take the lock there, before commit_heard.
        """
        owner = self._turn_owner is asyncio.current_task()
        if owner:
            self._turn_owner = None
            self._turn_lock.release()
        try:
            return await awaitable
        finally:
            if owner:
                await self._turn_lock.acquire()
                self._turn_owner = asyncio.current_task()

    def _record(self, fn: Callable, *args: Any) -> None:
        """Disk and header errors must not end the call or skip playback accounting."""
        try:
            fn(*args)
        except (OSError, struct.error) as e:
            self._emit("error", where="recorder", error=repr(e))

    # ------------------------------------------------------------------ input
    async def feed(self, frame: np.ndarray) -> None:
        """Feed one 512-sample 16 kHz int16 frame (32 ms)."""
        # Scoring awaits. Keep the lock until this frame is applied so a verdict, inject_text,
        # or say that becomes runnable during the score cannot commit or barge ahead of it.
        async with self._turn_lock:
            self._turn_owner = asyncio.current_task()
            try:
                if self.recorder is not None:
                    self._record(self.recorder.mic, frame)
                if self.vad_scheduler is not None:
                    prob = await self.vad_scheduler.prob(self.vad, frame)
                else:
                    prob = self.vad.prob(frame)   # no scheduler: stay on this task (Rig tests depend on it)
                speech = prob >= self.cfg.turn.threshold
                if speech:
                    self._last_speech_clock = self.clock()
                agent = self._agent_active()
                self.turn.set_agent_speaking(agent)
                # Keep the burst ourselves. _pre is only 320 ms, shorter than the backchannel gate,
                # so a "stop" that ends before the gate would already have fallen out of it.
                # Collection continues after playback stops; clearing here would cut the onset.
                if speech and not self.turn.in_turn and (agent or self._collecting_overlap):
                    self._overlap_audio.append(frame)
                    self._collecting_overlap = True
                    if len(self._overlap_audio) > 40:
                        del self._overlap_audio[:-40]
                self._fed_speech = speech
                (self._utt if self.turn.in_turn else self._pre).append(frame)
                events = self.turn.process(prob)
                # A cough under the barge floor never opens a check. Drop it, or the next yeah is
                # transcribed glued to that audio with the silence between them removed.
                # A run still pending (a short hole, or playback that just stopped) keeps its audio.
                if (not speech and not self.turn.in_turn and not self.turn.pending_run
                        and not any(ev.kind is Ev.OVERLAP_CHECK for ev in events)):
                    self._overlap_audio.clear()
                    self._collecting_overlap = False
                for ev in events:
                    await self._handle(ev)
                g = self._gen
                # barge_confirm_ms is from the start of the utterance, not the pause.
                # The default backchannel gate is already past it, so a normal barge confirms
                # on the same frame. A shorter gate stays paused until speech reaches it.
                if g and g.holding and not g.cancelled and self.turn.speech_ms >= self.cfg.turn.barge_confirm_ms:
                    await self._interrupt(g)
            finally:
                if self._turn_owner is asyncio.current_task():
                    self._turn_owner = None

    async def start(self) -> None:
        """Called by the transport once the call is connected: speaks the configured greeting, if any."""
        if self.cfg.greeting.strip():
            await self.say(self.cfg.greeting.strip())

    def _drop_finished_playback(self) -> None:
        """Forget chunks once playback has finished. A later turn must not inherit their acks."""
        if not self.tracker.is_playing():
            self.tracker.reset()

    async def say(self, text: str) -> None:
        """Speaks fixed text with no model call. It is a normal committed reply: the caller can talk over it
        (barge-in) and the history keeps only what they heard."""
        async with self._turn_lock:
            self._drop_finished_playback()
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
        # Until the first ack, buffered_ms follows the local clock and can hit zero while Twilio
        # still has the audio. Voicemail hangs up when this returns. The give-up is re-read each
        # pass: ends_at moves when more audio is armed while this sleeps.
        while not gen.cancelled and self.clock() < deadline:
            if self._playback_blocking():
                await asyncio.sleep(0.05)
                continue
            break

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
            self._drop_finished_playback()
            self.stats["turns"] += 1
            self.history.append({"role": "user", "content": text})
            base = list(self.history)
            self.history.append({"role": "assistant", "content": ""})
            gen = _Generation(np.zeros(0, np.int16), base, self.clock())
            gen.prior_transcript = self._transcript_task
            gen.committed = True
            gen.commit.set()
            gen.tasks = [asyncio.create_task(self._produce(gen)), asyncio.create_task(self._synth(gen)),
                         asyncio.create_task(self._drain(gen))]
            self._gen = gen
            self._emit("user_transcript", text=text, source="text")

    async def close(self) -> None:
        try:
            self._overlap_dropped.update(self._overlap_live)
            self._overlap_dropped.update(self._overlap_claimed)
            self._overlap_live.clear()
            self._overlap_claimed.clear()
            tasks = [t for t in list(self._overlap_tasks) if not t.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._overlap_tasks.clear()
            self._cancel_eot()
            if self._gen:
                await self._cancel(self._gen)
        finally:
            # Hang-up can land while close() is still in _cancel. Finish every in-flight ASR,
            # oldest included, so call.ended sees the lines. Shielded: cancelling close() must
            # not cancel the transcript.
            await self._finish_transcripts()

    async def _finish_transcripts(self) -> None:
        # Every in-flight ASR, not only the latest pointer: a later commit replaces it
        # while the earlier turn is still "[caller audio]". Shielded, and a cancel of
        # close() waits the shield out. A bare gather would cancel the transcript.
        tasks = [t for t in list(self._transcript_tasks) if not t.done()]
        if not tasks:
            return
        try:
            await asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True)
        except asyncio.CancelledError:
            pending = [t for t in tasks if not t.done()]
            if pending:
                await asyncio.gather(*(asyncio.shield(t) for t in pending), return_exceptions=True)
            raise

    # --------------------------------------------------------------- events
    def _emit(self, type_: str, **data: Any) -> None:
        self.on_event({"type": type_, "t": self.clock(), **data})

    async def _handle(self, ev: TurnEvent) -> None:
        if ev.kind is Ev.SPEECH_START:
            # _pre is 320 ms. A held burst, including one whose playback just stopped, lives
            # in the overlap buffer. Audio already handed to a classifier is spliced in front
            # so the model still hears the onset when that result is discarded.
            held: list[np.ndarray] = []
            if self._collecting_overlap and self._overlap_audio:
                held = list(self._overlap_audio)
                if self._fed_speech and self._pre and held[-1] is not self._pre[-1]:
                    held.append(self._pre[-1])
            current = held if held else list(self._pre)
            detached = self._take_inflight_overlap()
            if detached:
                skip = {id(f) for f in detached}
                current = [f for f in current if id(f) not in skip]
            self._utt = detached + current
            self._overlap_audio = []
            self._collecting_overlap = False
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
            overlapped, self._overlapped = self._overlapped, False
            if self._gen and not self._gen.committed:
                self._commit(self._gen, overlapped=overlapped, speech_ms=self.turn.last_speech_ms)
        elif ev.kind is Ev.OVERLAP_CHECK:
            frames = list(self._overlap_audio)
            self._overlap_audio = []
            self._collecting_overlap = False
            self._overlap_token += 1
            token = self._overlap_token
            self._overlap_live[token] = frames
            audio = np.concatenate(frames) if frames else np.zeros(0, np.int16)
            task = asyncio.create_task(self._classify_overlap(audio, float(ev.speech_ms), token))
            self._overlap_tasks.add(task)
            self._bg.add(task)
            task.add_done_callback(self._overlap_tasks.discard)
            task.add_done_callback(self._bg.discard)
        elif ev.kind is Ev.SPEECH_ABORT:
            self._cancel_eot()
            self._utt = []
            self._overlapped = False
            self._emit("speech_abort")
            g = self._gen
            if g and g.holding:
                await self._resume_after_barge(g)

    def _overlap_stale(self, token: int) -> bool:
        return token in self._overlap_dropped

    def _take_inflight_overlap(self) -> list[np.ndarray]:
        """Frames handed to a classifier that a real turn must still play to the model."""
        parts: list[np.ndarray] = []
        tokens = sorted(set(self._overlap_live) | set(self._overlap_claimed))
        for token in tokens:
            frames = self._overlap_live.pop(token, None)
            if frames is None:
                frames = self._overlap_claimed.pop(token, None)
            else:
                self._overlap_claimed.pop(token, None)
            self._overlap_dropped.add(token)
            if frames:
                parts.extend(f for f in frames if len(f))
        return parts

    async def _classify_overlap(self, audio: np.ndarray, speech_ms: float, token: int) -> None:
        """Transcribe a short overlap before clearing playback.

        Empty audio, a failed transcript, and a backchannel leave the reply playing. Real words
        ("stop", "no") interrupt and start a turn on this audio even if a later burst has already
        started. If a real turn starts first, that turn keeps this audio and the result is ignored.
        Once the words are real, a later clear or transcript error still commits the turn.
        """
        raw = ""
        try:
            raw = (await self.llm.transcribe(audio)).strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="overlap_transcribe", error=repr(e))
            raw = ""
        if self._overlap_stale(token):
            return
        if (not raw or raw == "[inaudible]"
                or is_backchannel(raw, speech_ms, max_ms=self.cfg.turn.backchannel_ms)):
            self._overlap_live.pop(token, None)
            if self._overlap_stale(token):
                return
            self._emit("backchannel", text=raw, duration_ms=speech_ms)
            return
        if self._overlap_stale(token):
            return
        # Claim the frames so a turn that opens during the awaits below takes them, and this
        # result does not also commit them.
        frames = self._overlap_live.pop(token, None)
        if frames is not None:
            self._overlap_claimed[token] = frames
        committed = False
        try:
            async with self._overlap_commit_lock:
                if self._overlap_stale(token):
                    return
                committed = await self._commit_real_overlap(audio, speech_ms, token)
        except asyncio.CancelledError:
            raise
        finally:
            if not committed:
                self._release_overlap_token(token)

    async def _commit_real_overlap(self, audio: np.ndarray, speech_ms: float, token: int) -> bool:
        """Interrupt whatever is playing and commit this overlap as its own turn.

        Returns False when a real turn already took the audio. Clear and the previous
        transcript are awaited here; if either fails the utterance is still committed.
        Callers serialize this so a second overlap sees the generation the first one started.
        """
        g = self._gen
        did_interrupt = False
        unscored_before = self._overlap_unscored
        overlapped_before = self._overlapped
        try:
            if g and not g.cancelled:
                if g.committed and self._agent_active():
                    await self._interrupt(g)
                    did_interrupt = True
                else:
                    await self._cancel(g)
                    if self._gen is g:
                        self._gen = None
            if self._overlap_stale(token):
                return False
            # Commit replaces _transcript_task. Wait for the one already in flight first, or the
            # model is sent "[caller audio]" for the turn the caller just finished.
            prior = self._transcript_task
            if prior is not None and not prior.done():
                await prior
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="overlap_commit", error=repr(e))
            if self._overlap_stale(token):
                return False
            did_interrupt = await self._recover_overlap_interrupt(
                g, did_interrupt, unscored_before, overlapped_before)
        if self._overlap_stale(token):
            return False
        self._land_overlap_turn(audio, speech_ms, token, did_interrupt)
        return True

    async def _recover_overlap_interrupt(self, g: Optional[_Generation], did_interrupt: bool,
                                         unscored_before: int, overlapped_before: bool) -> bool:
        """Cancel a reply _interrupt did not finish stopping. The overlap turn still commits."""
        if g is not None and not g.cancelled:
            # _interrupt counts the barge before clear_audio. Roll that back if clear never returned,
            # then stop the reply locally so it cannot keep playing.
            self._overlap_unscored = unscored_before
            self._overlapped = overlapped_before
            try:
                await self._cancel(g)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._emit("error", where="overlap_cancel", error=repr(e))
            if self._gen is g:
                self._gen = None
            heard, cut = self.tracker.heard()
            self.history = trim_history(self.history, heard, cut)
            self.tracker.reset()
            self._overlapped = True
            self._overlap_unscored = unscored_before + 1
            self.stats["interruptions"] += 1
            try:
                self._emit("user_interrupted", heard=heard, cut=cut)
            except Exception as e:  # noqa: BLE001
                self._emit("error", where="overlap_interrupt", error=repr(e))
            return True
        # Interrupt already cancelled playback and counted it, then a later step (the prior
        # transcript) failed. Keep that count so the new turn's transcript can close it.
        if self._overlap_unscored > unscored_before:
            return True
        return did_interrupt

    def _land_overlap_turn(self, audio: np.ndarray, speech_ms: float, token: int,
                           did_interrupt: bool) -> None:
        self._overlap_claimed.pop(token, None)
        self.stats["turns"] += 1
        try:
            self._emit("user_end_of_turn")
        except Exception as e:  # noqa: BLE001 — handler must not drop a turn that already qualified
            self._emit("error", where="overlap_commit", error=repr(e))
        self._utt = [audio] if len(audio) else []
        self._start_generation()
        self._commit(self._gen, overlapped=did_interrupt, speech_ms=speech_ms)
        self._utt = []
        self._overlapped = False

    def _release_overlap_token(self, token: int) -> None:
        """Forget audio this check will not commit, so a later turn cannot prepend it."""
        self._overlap_dropped.add(token)
        self._overlap_live.pop(token, None)
        self._overlap_claimed.pop(token, None)

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
        self._drop_finished_playback()
        audio = np.concatenate(self._utt) if self._utt else np.zeros(0, np.int16)
        gen = _Generation(audio, list(self.history), self._last_speech_clock)
        gen.prior_transcript = self._transcript_task
        gen.tasks = [asyncio.create_task(self._produce(gen)), asyncio.create_task(self._synth(gen)),
                     asyncio.create_task(self._drain(gen))]
        self._gen = gen

    def _commit(self, gen: _Generation, overlapped: bool = False, speech_ms: float = 0.0) -> None:
        gen.committed = True
        self.history.append({"role": "user", "content": "[caller audio]"})
        self.history.append({"role": "assistant", "content": ""})
        user_idx = len(self.history) - 2
        # Snapshot before this task is recorded, so it waits on earlier slots and not on itself.
        # A faster later turn must not emit user_transcript before those earlier slots do.
        prior = [t for _, t in self._transcripts]
        task = asyncio.create_task(self._transcribe(gen.audio, user_idx, prior, overlapped, speech_ms))
        self._transcripts.append((user_idx, task))
        self._transcript_tasks.add(task)

        def _done(done: asyncio.Task) -> None:
            self._transcript_tasks.discard(done)
            self._transcripts = [(i, t) for i, t in self._transcripts if t is not done]

        if task.done():
            _done(task)
        else:
            task.add_done_callback(_done)
        self._transcript_task = task
        gen.transcript_task = task   # so a later turn's task is not the one we await
        gen.user_idx = user_idx
        gen.commit.set()

    def note_overlap_transcript(self, text: str, duration_ms: float) -> None:
        """Transcript of caller audio that overlapped the agent.

        A backchannel emits `backchannel` and does not count as an interruption. Playback was already
        cleared at SPEECH_START; this cannot put that audio back.
        """
        pending = self._overlap_unscored > 0
        if pending:
            self._overlap_unscored -= 1
        if not is_backchannel(text, duration_ms):
            return
        if pending:
            self.stats["interruptions"] = max(0, self.stats["interruptions"] - 1)
        self._emit("backchannel", text=text, duration_ms=duration_ms)

    async def _transcribe(self, audio: np.ndarray, idx: int, prior: list[asyncio.Task],
                          overlapped: bool = False, speech_ms: float = 0.0) -> None:
        failed = False
        raw = ""
        try:
            raw = (await self.llm.transcribe(audio)).strip()
            text = raw or "[inaudible]"
        except Exception as e:  # noqa: BLE001
            failed = True
            text = "[inaudible]"
            self._emit("error", where="transcribe", error=repr(e))
        # A faster later turn must not emit user_transcript before the earlier slot does.
        for t in prior:
            if not t.done():
                await asyncio.shield(t)
        self.history[idx]["content"] = text
        try:
            self._emit("user_transcript", text=text)
        except Exception as e:  # noqa: BLE001  (a store write must not skip the tool round)
            self._emit("error", where="user_transcript", error=repr(e))
        if overlapped and not failed:
            self.note_overlap_transcript(raw, speech_ms)
        elif overlapped and self._overlap_unscored:
            self._overlap_unscored -= 1          # a failed transcript must not un-count a later turn
        if self.on_utterance is not None:
            try:
                self.on_utterance(audio, text, self._last_eot)
            except Exception as e:  # noqa: BLE001  (a dataset write must never break a call)
                self._emit("error", where="utterance_sink", error=repr(e))

    def _tool_round_history(self, gen: _Generation) -> list[dict]:
        """This turn's caller audio, as text, prepended to the history the tool round sees.

        `user_idx` is only set for an audio commit. Text turns (keypad) already have the user
        message in `base_history`, so they are left as they are.
        """
        idx = getattr(gen, "user_idx", None)
        if not isinstance(idx, int) or not (0 <= idx < len(self.history)):
            return list(gen.base_history)
        content = self.history[idx].get("content")
        transcript = "[inaudible]" if content == "[caller audio]" else content
        return [*gen.base_history, {"role": "user", "content": transcript}]

    async def _produce(self, gen: _Generation) -> None:
        followups: list[dict] = []
        # Replaced each round so the short first-clause split stays per round. The name
        # lives outside the loop so a model error can flush the round that failed.
        chunker = SentenceChunker()
        try:
            # Every older slot still in this request, not only the latest pointer. A newer task
            # must not hide an older one that is still "[caller audio]". This turn's own task is
            # past base_history, so cancelling the reply does not cancel it. Shield anyway.
            pending = [t for idx, t in self._transcripts if idx < len(gen.base_history) and not t.done()]
            if pending:
                for t in pending:
                    await asyncio.shield(t)
                gen.base_history = list(self.history[: len(gen.base_history)])
            for round_i in range(self.cfg.max_tool_rounds + 1):
                chunker = SentenceChunker()
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                t_req, first_tok = self.clock(), True
                # One audio clip per request. Round 0 keeps this turn's audio; later rounds send the
                # transcript (text) and must not re-prefill that clip. The user text stays out of followups.
                # The fitted view is for the model only. self.history stays complete, and followup
                # tokens come out of the same budget.
                if round_i == 0:
                    hist, audio = gen.base_history, gen.audio
                else:
                    hist, audio = self._tool_round_history(gen), np.zeros(0, np.int16)
                room = self.cfg.max_context_tokens - estimate_messages(followups)
                model_history = fit_history(hist, room, self.cfg.history_tail)
                async for item in self.llm.stream(model_history, audio, followups,
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
                task = getattr(gen, "transcript_task", None)
                # Await this generation's transcript even when a newer turn has replaced the
                # pointer. Skipping it sends the tool round "[inaudible]" instead of the words.
                # Shielded: a barge-in cancels this reply and must not cancel the transcript.
                if task is not None and not task.done():
                    await asyncio.shield(task)
                followups.append({"role": "assistant", "content": "".join(text_parts) or None,
                                  "tool_calls": [{"id": c.id, "type": "function",
                                                  "function": {"name": c.name,
                                                               "arguments": json.dumps(c.arguments)}}
                                                 for c in calls]})
                for c in calls:
                    self._emit("tool_call", name=c.name, arguments=c.arguments)
                    t_tool = self.clock()
                    done = asyncio.Event()
                    # Not Session.say: that would replace self._gen and drop this reply.
                    filler = asyncio.create_task(self._tool_filler(gen, c.name, done))
                    try:
                        out = await self._run_tool(c, gen)
                    finally:
                        done.set()
                        filler.cancel()
                        await asyncio.gather(filler, return_exceptions=True)
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
                # A tentative barge-in must keep the sentence already pulled off textq.
                while gen.holding and not gen.cancelled:
                    await asyncio.sleep(self.cfg.tts_poll_s)
                if gen.cancelled:
                    return
                item = await gen.textq.get()
                if item is None:
                    return
                is_fallback = item is _FALLBACK
                # Off textq until _speak returns, including the commit and lookahead waits.
                # A tool filler must see this or it takes speak_lock in that gap and plays first.
                gen.holding_sentence = True
                try:
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
                            while not gen.cancelled and (
                                    gen.holding
                                    or self.tracker.buffered_ms() + gen.pending_ms > self.cfg.tts_lookahead_ms):
                                await asyncio.sleep(self.cfg.tts_poll_s)
                    if gen.cancelled:
                        return
                    try:
                        await self._speak(gen, text)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        await self._on_tts_failure(gen, is_fallback, e)
                        return
                    if not is_fallback:
                        idx += 1
                finally:
                    gen.holding_sentence = False
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

    def _playback_blocking(self) -> bool:
        """True while a hang-up or say_and_wait should keep holding.

        Armed marks, and a send that raised (bytes may already be buffered, with no echo coming),
        hold until mark_timeout_ms after the live clock estimate. ends_at moves when more audio
        is armed, so this is read on every pass of the wait.
        """
        now = self.clock()
        if self.tracker.pending_marks or self.tracker._unconfirmed:
            return now < self.tracker.ends_at() + self.cfg.mark_timeout_ms
        return self.tracker.buffered_ms() > 0

    async def _finish_action(self, gen: _Generation) -> None:
        deadline = self.clock() + 30000.0
        # Marks lag the local clock. A tentative barge must not take gen.after while holding:
        # a cough resumes the tail, a confirmed interrupt decides strict vs cancelled.
        while not gen.cancelled and self.clock() < deadline and (
                gen.holding or self._playback_blocking()):
            await asyncio.sleep(0.05)
        if gen.cancelled or gen.holding:
            return
        a, gen.after = gen.after, None      # taken only now: a barge-in during the wait still sees it
        if a is not None:
            self._spawn_action(a)

    async def _tool_filler(self, gen: _Generation, name: str, done: asyncio.Event) -> None:
        """Speak a short line if the tool is still running after filler_delay_s and every sentence
        already queued for this reply has been synthesized. Blank text disables it."""
        text = self.cfg.tool_fillers.get(name, self.cfg.filler_text)
        if not str(text).strip() or gen.cancelled:
            return
        await asyncio.sleep(self.cfg.filler_delay_s)
        # textq is empty while _synth holds the next sentence for lookahead, so the flag is required.
        # Do not take speak_lock here: _speak takes it, and the lock is not reentrant.
        while gen.holding_sentence or not gen.textq.empty():
            if done.is_set() or gen.cancelled:
                return
            await asyncio.sleep(self.cfg.tts_poll_s)
        if done.is_set() or gen.cancelled:
            return
        await self._speak(gen, str(text))

    def _commit_transcript(self, gen: _Generation, text: str) -> None:
        """Record assistant text whose audio was sent, and emit one agent_speech."""
        if not text:
            return
        gen.spoken.append(text)
        self._emit("agent_speech", text=text)
        if self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] = " ".join(gen.spoken)

    def _cover_slices(self, track: Any) -> None:
        if isinstance(track, list):
            for sl in track:
                if isinstance(sl, _WordSlice):
                    sl.covered = True

    def _commit_sent_prefix(self, gen: _Generation) -> None:
        """One transcript for slices send_audio already accepted and not yet recorded.

        Chunks still in the outbox are not included. Barge-in trims history after this, so the
        heard prefix plus `[interrupted]` still replaces the assistant turn.
        """
        groups: list[list[_WordSlice]] = []
        for sl in gen.sent_slices:
            if sl.covered:
                continue
            if groups and groups[-1][0].sentence == sl.sentence:
                groups[-1].append(sl)
            else:
                groups.append([sl])
        parts: list[str] = []
        for group in groups:
            joined = " ".join(sl.text for sl in group if sl.text).strip()
            for sl in group:
                sl.covered = True
            if joined:
                parts.append(joined)
        if parts:
            self._commit_transcript(gen, " ".join(parts))

    def attach_transport_marks(self, transport) -> None:
        """If the transport echoes playback marks, feed them to the tracker."""
        if hasattr(transport, "set_mark_handler"):
            transport.set_mark_handler(self.tracker.note_mark)

    async def _speak(self, gen: _Generation, text: str) -> None:
        """Enqueue each TTS chunk as it arrives. The sentence is recorded once every chunk is sent,
        or only the words on chunks already sent if synthesis or send_audio stops early."""
        # Held by both the synth loop and a tool filler so their audio cannot interleave.
        # Drain does not take the lock, so a chunk can be sent while the next one is still synthesizing.
        async with gen.speak_lock:
            t_tts = self.clock()
            slices: list[_WordSlice] = []
            lengths: list[int] = []
            yielded = False
            try:
                async for part in self.tts.synthesize(text):
                    yielded = True
                    pcm = resample(np.asarray(part), self.tts.sample_rate, self.cfg.out_rate)
                    if len(pcm) == 0:
                        continue
                    # The slice is what barge-in reads. It stays a conservative prefix until synthesis finishes.
                    sl = _WordSlice("", text)
                    slices.append(sl)
                    lengths.append(len(pcm))
                    _claim_words_so_far(slices, lengths, text, self.cfg.out_rate)
                    gen.pending_ms += pcm_duration_ms(len(pcm), self.cfg.out_rate)
                    gen.outbox_audio += 1
                    await gen.outbox.put(("", sl, pcm))
            except Exception:
                # A later chunk failed: keep only the words attached to audio already queued.
                queued = " ".join(sl.text for sl in slices if sl.text).strip()
                if queued:
                    await gen.outbox.put((queued, slices, None))
                raise
            self._emit("tts_done", ms=self.clock() - t_tts, chars=len(text))
            if slices:
                _fill_word_slices(slices, lengths, text)
                await gen.outbox.put((text, slices, None))
            elif yielded:
                # The engine handed back silence on purpose (an empty frame). Still commit the sentence.
                await gen.outbox.put((text, text, np.zeros(0, np.int16)))
            else:
                # A 2xx body with no samples is not speech the caller heard. Raising would make
                # _synth drop every sentence still queued; skip this one and keep going.
                self._emit("error", where="tts", error="empty audio")

    async def _render(self, text: str) -> tuple[np.ndarray, float]:
        """Full PCM for a replay. Streaming sends go through _speak; this stays for the tail
        clear() threw away, which is already in history and must be queued as one replay item."""
        t_tts = self.clock()
        parts = [p async for p in self.tts.synthesize(text)]
        self._emit("tts_done", ms=self.clock() - t_tts, chars=len(text))
        pcm = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        pcm = resample(pcm, self.tts.sample_rate, self.cfg.out_rate)
        return pcm, pcm_duration_ms(len(pcm), self.cfg.out_rate)

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

    def _unpack_outbox(self, item: Any) -> tuple[str, Any, Any, bool]:
        """Return text, track, pcm, replay.

        (text, pcm, True) is a replay of audio clear() discarded (the third element is bool, not samples).
        (text, slices, None) is the sentence marker. ("", slice, pcm) is one streamed chunk.
        (text, text, zeros) is a sentence synthesize produced no samples for. (text, pcm) is a legacy pair.
        """
        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], bool):
            text, pcm, replay = item
            return text, text, pcm, replay
        if isinstance(item, tuple) and len(item) == 3:
            text, track, pcm = item
            return text, track, pcm, False
        text, pcm = item
        return text, text, pcm, False

    async def _drain(self, gen: _Generation) -> None:
        try:
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
                text, track, pcm, replay = self._unpack_outbox(item)
                if pcm is None:
                    # A send failure may already have recorded earlier slices of this sentence.
                    # The marker still names the whole sentence; do not append that prefix twice.
                    slices = [sl for sl in track if isinstance(sl, _WordSlice)] if isinstance(track, list) else []
                    if slices and any(sl.sentence in gen.failed_sentences for sl in slices):
                        self._commit_sent_prefix(gen)
                        self._cover_slices(track)
                        continue
                    already = any(sl.covered for sl in slices)
                    fresh = [sl for sl in slices if not sl.covered and sl.text]
                    self._cover_slices(track)
                    if slices and not already:
                        self._commit_transcript(gen, text)
                    elif fresh:
                        self._commit_transcript(gen, " ".join(sl.text for sl in fresh).strip())
                    elif not slices and text:
                        self._commit_transcript(gen, text)
                    continue
                counted = isinstance(track, _WordSlice) and len(pcm) > 0
                if counted:
                    gen.outbox_audio -= 1
                if not gen.first_audio_sent:
                    gen.first_audio_sent = True
                    self._emit("agent_first_audio",
                               latency_ms=self.clock() - gen.speech_end_clock)
                gen.sending = True
                try:
                    failed = False
                    mark = None
                    try:
                        mark = await self.send_audio(pcm, self.cfg.out_rate)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        if gen.holding:
                            # Tentative barge: this chunk did not land. The next drain replays it.
                            if counted:
                                gen.outbox_audio += 1
                            self._commit_sent_prefix(gen)
                            self._emit("error", where="send_audio", error=repr(e))
                            self._requeue_front(gen, item)
                            return
                        # Media may already be buffered (Twilio writes it, then the mark).
                        # Keep the chunk on the clock so lookahead and hang-up still cover it.
                        self._emit("error", where="send_audio", error=repr(e))
                        mark = None
                        failed = True
                    if failed and isinstance(track, _WordSlice):
                        # The sentence marker must not record words whose send raised.
                        gen.failed_sentences.add(track.sentence)
                    elif isinstance(track, _WordSlice):
                        gen.sent_slices.append(track)
                    if self.recorder is not None and not failed:
                        self._record(self.recorder.agent, pcm, self.cfg.out_rate)
                    dur = pcm_duration_ms(len(pcm), self.cfg.out_rate)
                    self.tracker.on_sent(track if isinstance(track, _WordSlice) else text, dur)
                    if isinstance(mark, str) and mark:
                        self.tracker.arm_mark(mark)
                    if failed:
                        self.tracker._unconfirmed = True
                    gen.pending_ms -= dur
                    if replay:
                        # Already in spoken/history. Still tell listeners the words went out again.
                        self._emit("agent_speech", text=text)
                    elif text:
                        self._commit_transcript(gen, text)
                finally:
                    gen.sending = False
        except asyncio.CancelledError:
            # Barge-in, hang-up, and shutdown cancel drain before it reads the sentence marker.
            self._commit_sent_prefix(gen)
            raise

    # ----------------------------------------------------------- barge-in
    async def _pause_for_barge(self, gen: _Generation) -> None:
        """Stop the far end without cancelling the reply. Speech may still turn out to be noise."""
        if gen.holding or gen.cancelled:
            return
        heard, cut = self.tracker.heard()
        # Audio still in synthesize, in the outbox, or inside send_audio is not in the tracker,
        # so a finished chunk looks like a clean end unless we force the cut. A confirmed
        # interrupt reads this snapshot, not a later tracker.heard().
        if not gen.produce_done or gen.outbox_audio > 0 or gen.sending:
            cut = True
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
            self._record(self.recorder.clear)

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

    async def _prepend_replay_text(self, gen: _Generation, text: str) -> bool:
        """Queue a replay in front of whatever synth still has. False when nothing was queued."""
        chunker = SentenceChunker()
        pieces = [p for p in (*chunker.push(text), *chunker.flush()) if p.strip()]
        built = []
        for piece in pieces:
            if gen.cancelled:
                return False
            pcm, dur = await self._render(piece)
            built.append((piece, pcm, dur))
        if gen.cancelled or not built:
            return False
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
        return True

    def _drain_alive(self, gen: _Generation) -> bool:
        return bool(gen.tasks) and not gen.tasks[-1].done()

    def _replace_drain(self, gen: _Generation) -> None:
        """Previous drain has exited. Keep producers; start a player that reads the outbox."""
        drain = asyncio.create_task(self._drain(gen))
        if gen.tasks:
            gen.tasks = [*gen.tasks[:-1], drain]
        else:
            gen.tasks = [drain]

    def _start_tail_playback(self, gen: _Generation, text: str) -> None:
        """The reply already left the outbox. Speak the cleared tail without a second history turn."""
        # A keypad turn can cancel or replace this generation while the idle drain is shutting down.
        # Clearing cancelled here would revive a player close() no longer owns.
        if gen.cancelled or self._gen is not gen:
            return
        gen.produce_done = False

        async def produce() -> None:
            queued = False
            try:
                chunker = SentenceChunker()
                for piece in [*chunker.push(text), *chunker.flush()]:
                    if gen.cancelled:
                        return
                    if not piece.strip():
                        continue
                    await self._put_replay(gen, piece)
                    queued = True
                # Only once the audio is on a live drain. A failed synth must not look like playout.
                if queued and not gen.cancelled and self._drain_alive(gen):
                    self._emit("barge_aborted", text=text)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._emit("error", where="tts", error=repr(e))
            finally:
                gen.produce_done = True
                if not gen.cancelled and queued:
                    gen.outbox.put_nowait(None)
                elif not gen.cancelled:
                    # Nothing queued: do not hand the drain a finished-reply sentinel. That runs
                    # gen.after and hangs up before the cleared tail was ever spoken.
                    drain = gen.tasks[-1] if gen.tasks else None
                    if drain is not None and not drain.done() and drain is not asyncio.current_task():
                        drain.cancel()
                        drain.add_done_callback(lambda t: t.cancelled() or t.exception())

        gen.tasks = [asyncio.create_task(produce()),
                     asyncio.create_task(self._drain(gen))]

    async def _resume_idle(self, gen: _Generation, tail: str) -> bool:
        """Drain has finished, or is only waiting to run a deferred hang-up.

        A tail is spoken in the background; barge_aborted is emitted there once it is queued.
        """
        drain = gen.tasks[-1] if gen.tasks else None
        if drain is not None and not drain.done():
            drain.cancel()
            await self._without_turn_lock(asyncio.gather(drain, return_exceptions=True))
        # inject_text can run on that await. Do not commit_heard, replay, or hang up on a
        # generation it cancelled or replaced.
        if gen.cancelled or self._gen is not gen:
            return False
        self.tracker.commit_heard(gen.paused_heard)
        if tail:
            self._start_tail_playback(gen, tail)
            return False
        # Nothing left to say. Drop the hold first or _finish_action waits on it forever.
        gen.holding = False
        if gen.after is not None:
            await self._finish_action(gen)
        return True

    async def _resume_dead_drain(self, gen: _Generation, tail: str) -> bool:
        """The player died (send_audio failed). Prepend the tail onto a new drain; keep synth."""
        self.tracker.commit_heard(gen.paused_heard)
        self._replace_drain(gen)
        if tail and not await self._prepend_replay_text(gen, tail):
            return False
        return self._drain_alive(gen)

    async def _resume_after_barge(self, gen: _Generation) -> None:
        """A cough: do not trim history. Re-speak whatever clear() cut off, then continue the reply."""
        if gen.cancelled or not gen.holding:
            return
        tail = self._unheard_sent_text(gen)
        accepted = False
        try:
            if self._outgoing_idle(gen):
                accepted = await self._resume_idle(gen, tail)
            elif not self._drain_alive(gen):
                accepted = await self._resume_dead_drain(gen, tail)
            else:
                self.tracker.commit_heard(gen.paused_heard)
                accepted = await self._prepend_replay_text(gen, tail) if tail else True
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._emit("error", where="tts", error=repr(e))
            accepted = False
        finally:
            # A replay error must not stick the pause or escape the read loop.
            gen.holding = False
            gen.barge_paused = False
        if accepted and not gen.cancelled:
            self._emit("barge_aborted", text=tail)

    async def _interrupt(self, gen: _Generation) -> None:
        self._overlapped = True
        self._overlap_unscored += 1
        # After a pause the transport buffer is already gone, so the clock must not advance "heard".
        paused_barge = gen.barge_paused
        if paused_barge:
            heard, cut = gen.paused_heard, gen.paused_cut
            unfinished = False
        else:
            # _cancel flips produce_done. Read the cut reason first. heard is read after cancel
            # so time spent clearing the socket is not counted as playback.
            unfinished = (not gen.produce_done) or gen.outbox_audio > 0 or gen.sending
        await self.clear_audio()          # stop the far end first: that is what the caller hears
        if gen.cancelled:
            # Another caller cancelled this generation while clear_audio waited on the socket.
            # Keep their trim; a snapshot from this entry would wipe it and count the barge twice.
            self._overlap_unscored = max(0, self._overlap_unscored - 1)
            return
        if self.recorder is not None:
            self._record(self.recorder.clear)
        await self._cancel(gen)           # sets gen.cancelled before it yields, so the other caller loses
        gen.holding = False
        if not paused_barge:
            heard, cut = self.tracker.heard()
            # A dry jitter buffer is not a finished turn while the next chunk is still synthesizing.
            if unfinished:
                cut = True
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
        # Drain's cancel handler records slices it had already sent. Repeat is a no-op once covered,
        # and covers a drain task that finished without seeing the marker.
        self._commit_sent_prefix(gen)
