"""Builds a Session for a call from stored config, wiring events to the store/transport."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.metrics import METRICS
from core.recorder import CallRecorder, write_wav16
from core.session import Session, SessionConfig
from core.turn import TurnConfig
from . import tools as toolmod
from .store import Store


class LimitedTTS:
    """Caps simultaneous syntheses process-wide: TTS front-ends (G2P) are CPU/GIL-bound, so many
    parallel threads only add contention. The semaphore is shared by every call in the process."""
    _sem: Optional[asyncio.Semaphore] = None

    def __init__(self, inner, limit: int):
        self.inner, self.limit = inner, limit

    @property
    def sample_rate(self) -> int:
        # HttpTTS learns the rate from the WAV header during synthesize. A snapshot taken here
        # would resample that audio at the default rate.
        return self.inner.sample_rate

    async def synthesize(self, text: str):
        if LimitedTTS._sem is None:
            LimitedTTS._sem = asyncio.Semaphore(self.limit)
        async with LimitedTTS._sem:
            parts = [p async for p in self.inner.synthesize(text)]
        for p in parts:
            yield p


def _count(e: dict) -> None:
    """Session events -> Prometheus metrics."""
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
        METRICS.inc("vox_turns_total")
    elif t == "error":
        METRICS.inc("vox_errors_total", where=str(e.get("where", "")))


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

    def finalize(self, call_id: str) -> Optional[str]:
        """Called when a call ends: forgets the live session and writes the recording. Returns the WAV path."""
        self.live.pop(call_id, None)
        rec = self.recorders.pop(call_id, None)
        rec_dir = self.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        if rec is None or not rec_dir:
            return None
        path = os.path.join(rec_dir, f"{call_id}.wav")
        rec.save(path)
        return path

    def _earlier_messages(self, call_id: str, keep: int = 40) -> list[dict]:
        """Transcript of the call this one continues (following the chain of reconnects), oldest first, last `keep`."""
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

    def build_session(self, call: dict, transport, forward_events: bool = False) -> Session:
        cfg = call["config"]
        meta = call.get("metadata") or {}
        if isinstance(meta, str):
            import json
            meta = json.loads(meta or "{}")
        sid, token = os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")
        ctx = toolmod.CallContext(call_id=call["id"], caller=str(meta.get("from", "")), callee=str(meta.get("to", "")),
                                  call_sid=getattr(transport, "call_sid", "") or str(meta.get("callSid", "")),
                                  end=transport.end, twilio={"sid": sid, "token": token} if sid and token else None)
        tid = call["tenant_id"]
        ctx.docs = lambda name: self.store.corpus_docs(tid, name)
        ctx.embedder, ctx.base_url = self.embedder, self.public_url
        ctx.send_dtmf = getattr(transport, "send_dtmf", None)
        ctx.client_call = getattr(transport, "client_call", None)
        specs = toolmod.resolve_library(cfg.get("tools", []),
                                        lambda n: (self.store.get_tool(tid, n) or {}).get("spec"))
        schemas, fns = toolmod.build(specs, ctx, self.http)
        rec_dir = self.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        recorder = None
        on_utt = None
        if cfg.get("record") and rec_dir:
            recorder = CallRecorder()
            self.recorders[call["id"]] = recorder
            if os.environ.get("VOX_DATASET") == "1":
                counter = {"n": 0}

                def on_utt(audio, text, eot):          # noqa: F811
                    counter["n"] += 1
                    base = os.path.join(rec_dir, "utterances", f"{call['id']}-{counter['n']:03d}")
                    write_wav16(base + ".wav", audio)
                    with open(os.path.join(rec_dir, "utterances", "index.jsonl"), "a") as f:
                        f.write(json.dumps({"call": call["id"], "n": counter["n"], "text": text, "eot": eot,
                                            "seconds": round(len(audio) / 16000, 2)}) + "\n")
        from . import variables as varmod
        values = {**{k: str(v) for k, v in (cfg.get("variables") or {}).items()},
                  **{k: str(v) for k, v in (meta.get("variables") or {}).items()},
                  "caller": str(meta.get("from", "")), "callee": str(meta.get("to", "")), "callId": call["id"]}
        prompt = varmod.render(cfg.get("systemPrompt") or SessionConfig.system_prompt, values)
        resumed = bool(meta.get("resumedFrom"))
        from core.tts import LANG_NAME
        if cfg.get("language") in LANG_NAME:
            name = LANG_NAME[cfg["language"]]
            prompt += (f"\n\nAlways speak and reply in {name}, whatever language the caller uses; "
                       f"if they speak another language, politely stay in {name}.")
        scfg = SessionConfig(system_prompt=prompt,
                             out_rate=transport.out_rate, tools_schema=schemas,
                             greeting=(RESUME_PHRASE.get(cfg.get("language") or "en", RESUME_PHRASE["en"]) if resumed
                                       else varmod.render(str(cfg.get("greeting") or ""), values)),
                             turn=TurnConfig(end_ms=float(os.environ.get("VOX_END_MS", "640")),
                                             eot_threshold=float(os.environ.get("VOX_EOT_THRESHOLD", "0.8")),
                                             patient_end_ms=float(os.environ.get("VOX_PATIENT_MS", "1000"))))
        store, cid = self.store, call["id"]

        def on_event(e: dict) -> None:
            if os.environ.get('VOX_LOG_EVENTS'):
                print('EV', cid, e, flush=True)
            _count(e)
            if e["type"] == "user_transcript":
                store.add_message(cid, "user", e["text"], time.time())
            elif e["type"] == "agent_speech":
                store.add_message(cid, "agent", e["text"], time.time())
            if forward_events and hasattr(transport, "send_event"):
                import asyncio
                asyncio.get_running_loop().create_task(transport.send_event(e))

        tts = self.make_tts(cfg.get("voice", "af_heart"))
        if hasattr(tts, "speed"):
            tts.speed = float(cfg.get("speed", 1.0))
        lim = int(os.environ.get("VOX_TTS_CONCURRENCY", "4"))
        if lim > 0:
            tts = LimitedTTS(tts, lim)
        sess = Session(llm=self.llm, tts=tts, vad=self.make_vad(),
                       send_audio=transport.send_audio, clear_audio=transport.clear, clock=self.clock,
                       cfg=scfg, tools=fns, on_event=on_event, eot_fn=self._eot_sync(),
                       eot_async=self._eot_async(), recorder=recorder, on_utterance=on_utt)

        async def voicemail(message: str) -> None:
            await sess.say_and_wait(message)
            await transport.end()
        ctx.voicemail = voicemail
        if resumed:
            sess.seed_history(self._earlier_messages(meta["resumedFrom"]))
        self.live[call["id"]] = (sess, transport)
        ctx.turn_fn = lambda: sess.stats["turns"]      # lets confirm-before-acting tools tell turns apart
        return sess
