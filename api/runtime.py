"""Builds a Session for a call from stored config, wiring events to the store/transport."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.asr import AsrLLM, HttpASR
from core.backchannel import BACKCHANNEL_MS
from core.metrics import METRICS
from core.recorder import CallRecorder, write_wav16
from core.redact import redact_text
from core.session import Session, SessionConfig
from core.turn import TurnConfig
from core.vad_exec import VADScheduler
from . import tools as toolmod
from .experiments import VariantLLM, choose_variant
from .store import Store
from .writebehind import MessageLog


# Same wrapper tests import from this module. AsrLLM owns the behavior.
_LlmWithASR = AsrLLM


class LimitedTTS:
    """The semaphore stays (GIL-bound G2P); streaming removes the per-sentence buffer; a separate process is VOX_TTS_URL.

    Caps simultaneous syntheses process-wide: TTS front-ends (G2P) are CPU/GIL-bound, so many
    parallel threads only add contention. The semaphore is shared by every call in the process and
    is held while each chunk is yielded, not released after a private buffer of the whole sentence.
    Barge-in detaches the session task from an in-process worker, but the slot stays held until
    that thread actually exits."""
    _sem: Optional[asyncio.Semaphore] = None
    _bg: set[asyncio.Task] = set()

    def __init__(self, inner, limit: int):
        self.inner, self.limit = inner, limit

    @property
    def sample_rate(self) -> int:
        # HttpTTS replaces this when a WAV header arrives. A copy taken at init would resample
        # that audio at the wrong rate for the rest of the call.
        return self.inner.sample_rate

    async def synthesize(self, text: str):
        if LimitedTTS._sem is None:
            LimitedTTS._sem = asyncio.Semaphore(self.limit)
        sem = LimitedTTS._sem
        await sem.acquire()
        try:
            async for p in self.inner.synthesize(text):
                yield p
        finally:
            pending = _detached_workers(self.inner)
            if not pending:
                sem.release()
            else:
                task = asyncio.create_task(_release_tts_slot(sem, pending))
                LimitedTTS._bg.add(task)
                task.add_done_callback(LimitedTTS._bg.discard)


def _detached_workers(inner) -> list:
    take = getattr(inner, "detached_workers", None)
    if take is None:
        return []
    return [f for f in take() if f is not None and not f.done()]


async def _release_tts_slot(sem: asyncio.Semaphore, workers: list) -> None:
    try:
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        sem.release()


def _count(e: dict, variant: str = "") -> None:
    """Session events -> Prometheus metrics. `variant` labels turns only; other series stay unlabeled."""
    t = e["type"]
    if t == "agent_first_audio":
        METRICS.observe("vox_first_audio_ms", e["latency_ms"])
    elif t == "llm_first_token":
        METRICS.observe("vox_llm_first_token_ms", e["ms"])
    elif t == "tts_done":
        METRICS.observe("vox_tts_ms", e["ms"])
    elif t == "tool_call":
        METRICS.inc("vox_tool_calls_total", tool=e["name"])
    elif t == "tool_result":
        METRICS.observe("vox_tool_ms", e["ms"], tool=e["name"])
    elif t == "user_interrupted":
        METRICS.inc("vox_interruptions_total")
    elif t == "user_end_of_turn":
        # The label is one sample in the shared scrape. A line break would render as a second sample.
        if variant and "\n" not in variant and "\r" not in variant:
            METRICS.inc("vox_turns_total", variant=variant)
        else:
            METRICS.inc("vox_turns_total")
    elif t == "error":
        METRICS.inc("vox_errors_total", where=str(e.get("where", "")))


# History is text because the speech model accepts one audio clip. Ask the agent to repeat slots
# before acting when the operator turned that reminder on.
RESTATE_SLOTS = ("Earlier caller turns are transcripts and may not match the audio. "
                 "Restate any name, date, or number before you act on it.")

# Cascaded TTS cannot start until text exists. This does not let Kokoro hear the caller.
SHORT_FIRST_SENTENCE = "Keep your first sentence under eight words."


# Said when a phone call is reconnected after the media stream or its node was lost. Short phrases per language;
# the non-English ones have not been reviewed by a native speaker.
RESUME_PHRASE = {"en": "Sorry about that, we got cut off for a moment. I'm back.",
                 "es": "Disculpa, se cortó un momento. Ya estoy de vuelta.",
                 "fr": "Désolé, nous avons été coupés un instant. Je suis de retour.",
                 "hi": "माफ़ कीजिए, कनेक्शन कुछ पल के लिए कट गया था। मैं वापस आ गया हूँ।",
                 "it": "Scusa, ci siamo interrotti per un momento. Sono di nuovo qui.",
                 "ja": "失礼しました、一瞬途切れてしまいました。戻りました。",
                 "pt": "Desculpe, a ligação caiu por um instante. Já voltei.",
                 "zh": "不好意思，刚才断线了一下。我回来了。"}

# Spoken once when a committed reply fails in the model or in TTS. Short phrases per language;
# the non-English ones have not been reviewed by a native speaker.
ERROR_PHRASE = {"en": "Sorry, could you say that again?",
                "es": "Perdón, ¿puede repetirlo?",
                "fr": "Désolé, pouvez-vous répéter ?",
                "hi": "माफ़ कीजिए, क्या आप फिर से बोलेंगे?",
                "it": "Scusi, può ripetere?",
                "ja": "すみません、もう一度言っていただけますか？",
                "pt": "Desculpe, pode repetir?",
                "zh": "不好意思，请再说一遍。"}


@dataclass
class Runtime:
    llm: object
    make_tts: Callable[[str], object]       # voice -> TTS
    make_vad: Callable[[], object]          # fresh VAD per call (stateful)
    store: Store
    eot_fn: Optional[Callable] = None
    clock: Callable[[], float] = lambda: time.monotonic() * 1000.0   # ms: Session/PlaybackTracker expect ms
    http: Optional[object] = None          # shared httpx.AsyncClient for tool calls and Twilio transfer
    embedder: Optional[object] = None      # optional embeddings client for hybrid knowledge-base search
    public_url: str = ""                   # set by create_app from Settings; used in Twilio callback URLs
    record_dir: str = ""                   # where recordings / the utterance dataset go; "" -> VOX_RECORD_DIR env
    live: dict = field(default_factory=dict)       # call_id -> (Session, transport): lets webhooks reach a live call
    recorders: dict = field(default_factory=dict)  # call_id -> CallRecorder
    vad_scheduler: VADScheduler = field(default_factory=VADScheduler)  # one scheduler per process (one Runtime)
    bus: object = None                             # CallBus; create_app sets one. None in unit tests that build a Runtime by hand.
    _bus_handlers: dict = field(default_factory=dict)  # call_id -> handler subscribed on bus

    def _drop_bus(self, call_id: str) -> None:
        handler = self._bus_handlers.pop(call_id, None)
        if self.bus is not None and handler is not None:
            self.bus.unsubscribe(call_id, handler)

    def __post_init__(self) -> None:
        self.message_log = MessageLog(self.store)

    def flush_messages(self, call_id: str | None = None) -> None:
        """Block until queued transcripts are in the store. Call via asyncio.to_thread.

        Pass `call_id` so a different call's failed insert is not reported as this call's.
        """
        self.message_log.flush(call_id)

    def wait_messages(self) -> None:
        """Block until queued transcripts are visible. Call via asyncio.to_thread.

        Unlike `flush_messages`, this leaves a failed insert for `finish` to raise.
        """
        self.message_log.wait()

    def peek_message_error(self, call_id: str | None = None) -> BaseException | None:
        """This call's insert failure, still stored so flush_messages can raise it."""
        return self.message_log.peek_error(call_id)

    def finalize(self, call_id: str) -> Optional[str]:
        """Called when a call ends: forgets the live session and writes the recording. Returns the WAV path."""
        self._drop_bus(call_id)
        self.live.pop(call_id, None)
        rec = self.recorders.pop(call_id, None)
        rec_dir = self.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        if rec is None or not rec_dir:
            return None
        path = os.path.join(rec_dir, f"{call_id}.wav")
        rec.save(path)
        return path

    def _earlier_messages(self, call_id: str, keep: int = 40) -> list[dict]:
        """Transcript of the call this one continues (following the chain of reconnects), oldest first, last `keep`.

        Waits until queued rows are written. Must not run on the event loop.
        """
        self.message_log.wait()
        chain: list[str] = []
        cur = call_id
        while cur and len(chain) < 4:
            chain.append(cur)
            c = self.store.get_call_any(cur)
            m = (c or {}).get("metadata") or {}
            cur = (json.loads(m) if isinstance(m, str) else m).get("resumedFrom")
        out: list[dict] = []
        for cid in reversed(chain):
            out += self.store.messages(cid)
        return out[-keep:]

    def _assigned_variant(self, call: dict, experiment: dict) -> dict:
        """Variant for this call.

        A resume or a failed transfer is a new call row. Walk `resumedFrom` (same tenant, at most 4 rows, the same
        cap as `_earlier_messages`) and bucket off that root id so the conversation keeps one variant. Reuse an
        ancestor's recorded variant id only when it still belongs to this experiment. This call's own metadata
        `variant` is not an assignment.
        """
        variants = [v for v in (experiment.get("variants") or []) if isinstance(v, dict)]
        by_id = {v.get("id"): v for v in variants if isinstance(v.get("id"), str)}
        chain: list[dict] = []
        seen: set[str] = set()
        cur_id = str(call.get("id") or "")
        tid = call.get("tenant_id")
        pending: Optional[dict] = call
        while cur_id and cur_id not in seen and len(chain) < 4:
            seen.add(cur_id)
            if pending is not None and str(pending.get("id") or "") == cur_id:
                row = pending
            else:
                row = self.store.get_call_any(cur_id)
            pending = None
            if not row or row.get("tenant_id") != tid:
                break
            chain.append(row)
            raw = row.get("metadata") or {}
            if isinstance(raw, str):
                raw = json.loads(raw or "{}")
            nxt = raw.get("resumedFrom") if isinstance(raw, dict) else None
            if not isinstance(nxt, str) or not nxt:
                break
            cur_id = nxt
        for row in chain[1:]:
            raw = row.get("metadata") or {}
            if isinstance(raw, str):
                raw = json.loads(raw or "{}")
            if not isinstance(raw, dict):
                continue
            vid = raw.get("variant")
            # Own metadata.variant is ignored above. An ancestor's recorded id counts only while this experiment
            # still has that id. Caller creates do not get to write it; build_session does.
            if isinstance(vid, str) and vid in by_id:
                return dict(by_id[vid])
        root_id = str(chain[-1]["id"]) if chain else str(call.get("id") or "")
        return choose_variant(experiment, root_id)

    @staticmethod
    def _eot_mode() -> str:
        return os.environ.get("VOX_EOT", "")

    def _eot_async(self):
        """Model-based end-of-turn verdict (VOX_EOT=model|1|both). Opt-in until accuracy is verified."""
        if self._eot_mode() in ("1", "model", "both") and hasattr(self.llm, "eot_score"):
            return lambda audio, history: self.llm.eot_score(history, audio)
        return None

    def _eot_sync(self):
        """Acoustic end-of-turn cues, no model (VOX_EOT=prosody|both); falls back to an injected eot_fn."""
        if self._eot_mode() in ("prosody", "both"):
            from core.prosody import eot_prosody
            return eot_prosody
        return self.eot_fn

    def _session_llm(self, variant=None):
        """LLM handed to the Session. Transcripts use `VOX_ASR_URL` when that client is configured.

        A variant model wraps first, then ASR, so tool rounds and history transcripts both hit
        HttpASR while `stream` stays on that variant. The wrapper is not stored on `self.llm`:
        `_eot_async` checks `eot_score` on the Ultravox client, and with no URL (or no shared HTTP
        client) `VllmClient.transcribe` stays in place. Earlier audio still cannot be replayed
        into Ultravox (`--limit-mm-per-prompt audio:1`); the separate transcript can diverge
        from what the model heard.
        """
        llm = self.llm
        model_name = str(variant.get("model") or "").strip() if variant else ""
        if model_name and hasattr(self.llm, "model"):
            llm = VariantLLM(self.llm, model_name)
        url = os.environ.get("VOX_ASR_URL", "").strip().rstrip("/")
        if not url or self.http is None:
            return llm
        model = os.environ.get("VOX_ASR_MODEL", "whisper-1").strip() or "whisper-1"
        return AsrLLM(llm, HttpASR(url, self.http, model=model))

    def build_session(self, call: dict, transport, forward_events: bool = False) -> Session:
        cfg = call["config"]
        meta = call.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        variant = None
        experiment = cfg.get("experiment")
        if isinstance(experiment, dict) and experiment.get("variants"):
            variant = self._assigned_variant(call, experiment)
            # The call dict can be older than the row: SIP writes the channel id after this object was loaded.
            stored = (self.store.get_call_any(call["id"]) or {}).get("metadata") or {}
            if isinstance(stored, str):
                stored = json.loads(stored or "{}")
            if not isinstance(stored, dict):
                stored = {}
            meta = {**stored, **meta, "experiment": experiment.get("id"), "variant": variant.get("id")}
            call["metadata"] = meta
            self.store.update_call(call["id"], metadata=meta)
        sid, token = os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")
        # On SIP, transport.call_sid is the AudioSocket UUID. ARI will not redirect or delete that id.
        if str(call.get("medium") or "") == "sip":
            channel_sid = str(meta.get("callSid") or "")
        else:
            channel_sid = getattr(transport, "call_sid", "") or str(meta.get("callSid") or "")
        ctx = toolmod.CallContext(call_id=call["id"], caller=str(meta.get("from", "")), callee=str(meta.get("to", "")),
                                  call_sid=channel_sid,
                                  end=transport.end, twilio={"sid": sid, "token": token} if sid and token else None,
                                  medium=str(call.get("medium") or ""))
        tid = call["tenant_id"]
        ctx.docs = lambda name: self.store.corpus_docs(tid, name)
        ctx.embedder, ctx.base_url = self.embedder, self.public_url
        if self.http is not None:
            from .callcontrol import control_from_env
            ctx.call_control = control_from_env(self.http, self.public_url or "", call.get("medium") or None)
        if str(call.get("medium") or "") == "sip":
            cid = call["id"]

            def record_sip_transfer(number: str) -> bool:
                # The dialplan looks the target up by the AudioSocket UUID after this leg drops.
                cur = self.store.get_call_any(cid)
                if not cur:
                    return False
                audio = str((cur.get("metadata") or {}).get("audioSocketUuid") or "")
                try:
                    audio = str(uuidlib.UUID(audio))
                except ValueError:
                    return False
                merged = self.store.merge_call_metadata(cid, {"transferTo": number, "audioSocketUuid": audio})
                return merged is not None

            ctx.record_sip_transfer = record_sip_transfer
        ctx.send_dtmf = getattr(transport, "send_dtmf", None)
        ctx.client_call = getattr(transport, "client_call", None)
        specs = toolmod.resolve_library(cfg.get("tools", []),
                                        lambda n: (self.store.get_tool(tid, n) or {}).get("spec"))
        # Any confirm tool, or the agent flag, turns the read-back guard on for this call.
        ctx.guard_slots = bool(cfg.get("confirmSlots")) or toolmod.needs_slot_guard(specs)
        schemas, fns = toolmod.build(specs, ctx, self.http)
        fillers = toolmod.filler_texts(specs)
        rec_dir = self.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        recorder = None
        on_utt = None
        redact = bool(cfg.get("redact"))
        if cfg.get("record") and rec_dir:
            recorder = CallRecorder(path=os.path.join(rec_dir, call["id"] + ".part.wav"))
            self.recorders[call["id"]] = recorder
            if os.environ.get("VOX_DATASET") == "1":
                counter = {"n": 0}

                def on_utt(audio, text, eot):          # noqa: F811
                    counter["n"] += 1
                    if redact:
                        text = redact_text(text)
                    base = os.path.join(rec_dir, "utterances", f"{call['id']}-{counter['n']:03d}")
                    write_wav16(base + ".wav", audio)
                    with open(os.path.join(rec_dir, "utterances", "index.jsonl"), "a") as f:
                        f.write(json.dumps({"call": call["id"], "n": counter["n"], "text": text, "eot": eot,
                                            "seconds": round(len(audio) / 16000, 2)}) + "\n")
        from . import variables as varmod
        values = {**{k: str(v) for k, v in (cfg.get("variables") or {}).items()},
                  **{k: str(v) for k, v in (meta.get("variables") or {}).items()},
                  "caller": str(meta.get("from", "")), "callee": str(meta.get("to", "")), "callId": call["id"]}
        prompt_src = cfg.get("systemPrompt") or SessionConfig.system_prompt
        variant_prompt = str(variant.get("systemPrompt") or "").strip() if variant else ""
        configured = cfg.get("systemPrompt") if isinstance(cfg.get("systemPrompt"), str) else ""
        if variant_prompt:
            prompt_src = str(variant["systemPrompt"])   # non-empty variant prompt replaces the agent prompt
        prompt = varmod.render(prompt_src, values)
        # The failed-transfer addendum lives on cfg["systemPrompt"] and metadata promptSuffix. Append that suffix
        # only when a variant prompt replaced the configured prompt and the configured prompt already ends with it.
        # A voice-only variant already has the notes inline; a client suffix that is not that tail is not a note.
        if variant_prompt:
            suffix = meta.get("promptSuffix") if isinstance(meta, dict) else None
            if isinstance(suffix, str) and suffix.strip() and configured.endswith(suffix):
                prompt += varmod.render(suffix, values)
        resumed = bool(meta.get("resumedFrom"))
        from core.tts import LANG_NAME
        if cfg.get("language") in LANG_NAME:
            name = LANG_NAME[cfg["language"]]
            prompt += (f"\n\nAlways speak and reply in {name}, whatever language the caller uses; "
                       f"if they speak another language, politely stay in {name}.")
        lang = cfg.get("language") or "en"
        if ctx.guard_slots:
            prompt += "\n\n" + toolmod.READBACK_INSTRUCTION
        if cfg.get("restateSlots"):
            prompt += "\n\n" + RESTATE_SLOTS
        if getattr(transport, "narrowband", False):
            prompt += ("\n\nThis call is narrowband telephone audio. Spell back any name and read back any number "
                       "before you use it.")
        # Lost-stream reconnects replace the greeting with a "cut off" apology. A failed warm transfer also
        # sets resumedFrom, but its own greeting is the "not available" line and must be kept.
        greeting = varmod.render(str(cfg.get("greeting") or ""), values)
        if resumed and meta.get("resumeReason") != "transfer":
            greeting = RESUME_PHRASE.get(lang, RESUME_PHRASE["en"])
        short_first_sentence = True
        if short_first_sentence:
            prompt += "\n\n" + SHORT_FIRST_SENTENCE
        # Twilio jitter: trust mark acks, and don't assume playback is instant. Other transports
        # keep output_latency_ms at 0 so clock-driven unit tests stay put.
        use_marks = getattr(transport, "out_rate", 0) == 8000 and hasattr(transport, "set_mark_handler")
        scfg = SessionConfig(system_prompt=prompt,
                             short_first_sentence=short_first_sentence,
                             out_rate=transport.out_rate, tools_schema=schemas,
                             output_latency_ms=200.0 if use_marks else 0.0,
                             greeting=greeting,
                             error_phrase=ERROR_PHRASE.get(lang, ERROR_PHRASE["en"]),
                             tool_fillers=fillers,
                             turn=TurnConfig(end_ms=float(os.environ.get("VOX_END_MS", "640")),
                                             eot_threshold=float(os.environ.get("VOX_EOT_THRESHOLD", "0.8")),
                                             patient_end_ms=float(os.environ.get("VOX_PATIENT_MS", "1000")),
                                             backchannel_ms=BACKCHANNEL_MS.get(str(cfg.get("bargeIn") or "normal"),
                                                                                BACKCHANNEL_MS["normal"])))
        cid = call["id"]
        log = self.message_log
        store = self.store
        variant_id = str(variant["id"]) if variant else ""

        def on_event(e: dict) -> None:
            if os.environ.get('VOX_LOG_EVENTS'):
                print('EV', cid, e, flush=True)
            _count(e, variant_id)
            if e["type"] in ("user_transcript", "agent_speech"):
                text = redact_text(e["text"]) if redact else e["text"]
                role = "user" if e["type"] == "user_transcript" else "agent"
                log.add(cid, role, text, time.time())
            # agent_first_audio carries latency_ms; the other three stages carry ms.
            # Speculative llm/tts events fire before END_OF_TURN increments stats["turns"].
            # `sess` is assigned below; events run only after the session exists.
            if e["type"] in ("llm_first_token", "tts_done", "agent_first_audio", "tool_result"):
                ms = e["ms"] if "ms" in e else e.get("latency_ms")
                if ms is not None:
                    try:
                        turn = int(sess.stats["turns"])
                        gen = sess._gen
                        if gen is not None and not gen.committed:
                            turn += 1
                        store.add_timing(cid, turn, e["type"], ms, time.time())
                    except Exception:  # noqa: BLE001  (a timings write must never break the call)
                        METRICS.inc("vox_errors_total", where="timings")
            if forward_events and hasattr(transport, "send_event"):
                import asyncio
                asyncio.get_running_loop().create_task(transport.send_event(e))

        voice = cfg.get("voice", "af_heart")
        if variant and variant.get("voice"):
            voice = variant["voice"]
        tts = self.make_tts(voice)
        if hasattr(tts, "speed"):
            tts.speed = float(cfg.get("speed", 1.0))
        lim = int(os.environ.get("VOX_TTS_CONCURRENCY", "4"))
        if lim > 0:
            tts = LimitedTTS(tts, lim)
        llm = self._session_llm(variant)
        eot_async = self._eot_async()
        if llm is not self.llm and eot_async is not None:
            eot_async = lambda audio, history, bound=llm: bound.eot_score(history, audio)
        sess = Session(llm=llm, tts=tts, vad=self.make_vad(),
                       send_audio=transport.send_audio, clear_audio=transport.clear, clock=self.clock,
                       cfg=scfg, tools=fns, on_event=on_event, eot_fn=self._eot_sync(),
                       eot_async=eot_async, recorder=recorder, on_utterance=on_utt,
                       vad_scheduler=self.vad_scheduler)
        sess.attach_transport_marks(transport)

        async def voicemail(message: str) -> None:
            # Same flag the owning node's webhook sets before it speaks, so a retry published
            # from another node does not play a second message over the first.
            if getattr(sess, "_vm_once", False):
                return
            sess._vm_once = True
            try:
                await sess.say_and_wait(message)
            finally:
                await transport.end()
        ctx.voicemail = voicemail
        # AMD may be delivered on another node, or on the Redis listener thread. Schedule onto
        # the loop that owns this session; there is no loop when a sync test builds a session.
        if self.bus is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                def on_bus(msg: dict, _loop=loop, _vm=voicemail) -> None:
                    if not isinstance(msg, dict) or msg.get("op") != "voicemail":
                        return
                    text = msg.get("text") or ""
                    if not text:
                        return
                    fut = asyncio.run_coroutine_threadsafe(_vm(str(text)), _loop)

                    def _retrieved(done) -> None:
                        if done.cancelled():
                            return
                        try:
                            done.result()
                        except Exception:  # noqa: BLE001  (retrieved so the failure is not logged as abandoned)
                            METRICS.inc("vox_errors_total", where="voicemail")

                    fut.add_done_callback(_retrieved)
                self._drop_bus(call["id"])
                self.bus.subscribe(call["id"], on_bus)
                self._bus_handlers[call["id"]] = on_bus
        if resumed:
            sess.seed_history(self._earlier_messages(meta["resumedFrom"]))
        self.live[call["id"]] = (sess, transport)
        ctx.turn_fn = lambda: sess.stats["turns"]      # lets confirm-before-acting tools tell turns apart

        def last_assistant_text() -> str:
            history = sess.history
            end = len(history)
            gen = sess._gen
            # Drain copies this generation's sentence into history[-1] before the next tool
            # round. That text is the proposal still in progress, not a finished read-back.
            if (gen is not None and gen.committed and not gen.cancelled and end
                    and history[-1].get("role") == "assistant"
                    and any(not t.done() for t in gen.tasks)):
                end -= 1
            for message in reversed(history[:end]):
                if message.get("role") == "assistant" and str(message.get("content") or "").strip():
                    return str(message["content"])
            return ""
        ctx.last_assistant_text = last_assistant_text

        async def last_user_text() -> str:
            # Tools run while the committed turn is still "[caller audio]". Wait for every in-flight
            # transcript: barge-in cancels this generation, and a newer turn replaces `_transcript_task`
            # while an older line is still being written. A newer benign line must not hide an older one.
            # Injection phrases are not a complete defense.
            try:
                while True:
                    pending = [t for t in list(sess._transcript_tasks) if not t.done()]
                    if not pending:
                        break
                    await sess._shield_transcripts(pending)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  (a failed transcript still leaves the earlier lines to check)
                pass
            lines = []
            for message in sess.history:
                if message.get("role") != "user":
                    continue
                text = str(message.get("content") or "").strip()
                if text and text != "[caller audio]":
                    lines.append(text)
            return "\n".join(lines)
        ctx.last_user_text = last_user_text

        def latest_user_line() -> str:
            # Consent is this line only. "[caller audio]" is not a yes, and an older yes
            # does not count once a later line is stored. Injection uses last_user_text,
            # which joins every line, so a newer benign line cannot hide an older phrase.
            for message in reversed(sess.history):
                if message.get("role") == "user":
                    return str(message.get("content") or "")
            return ""
        ctx.latest_user_line = latest_user_line
        ctx.audit = lambda action, detail: self.store.audit(tid, "tool", action, cid, detail)
        return sess
