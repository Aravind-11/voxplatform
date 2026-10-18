"""Production wiring (GPU box). Environment variables:

  VOX_ADMIN_KEY            admin key for POST /admin/tenants and /metrics (required to create tenants)
  VOX_PUBLIC_URL           public https URL Twilio and browsers reach (wss is derived from it)
  VOX_DB                   SQLite path (default vox.db) or postgresql://user:pass@host/db
  VOX_REDIS_URL            redis://host:6379/0 -> concurrency cap shared by all API nodes, and the call bus
                           (AMD voicemail when the webhook hits a different node). Default: in-process.
  VOX_VLLM_URL             Ultravox server, default http://127.0.0.1:8000/v1
  VOX_ASR_URL              optional OpenAI-compatible transcriber (include /v1). Caller transcripts come from it; Ultravox still hears only the current turn
  VOX_ASR_MODEL            transcription model for VOX_ASR_URL, default whisper-1
  VOX_AUDIO_STYLE          input_audio | audio_url
  VOX_DEFAULT_CONCURRENCY  per-tenant active-call cap, default 10
  VOX_NODE_MAX_CALLS       active calls this process admits (default 32). deploy.sh sets this to the
                           already-split per-process share; this process does not divide by VOX_PROCS again.
                           0 admits nobody. Unset leaves the limiter without a process ceiling and Settings
                           uses 32. A negative or non-integer value fails startup. A positive node cap smaller
                           than VOX_PROCS is rejected by deploy.sh, not rounded up to 1 per process. At the cap,
                           new calls are refused and /healthz is 503 with nodeFull so the load balancer stops
                           sending more.
  VOX_OVERFLOW_URL         optional http(s) URL. A full or draining node answers Twilio with <Redirect> to it
                           instead of the spoken busy line. Any other scheme is rejected.
  TWILIO_AUTH_TOKEN        verifies inbound Twilio webhooks and places outbound calls
  TWILIO_ACCOUNT_SID       outbound calls and the transfer tool
  VOX_TWILIO_NUMBER        default caller id for outbound calls
  VOX_OUTBOUND_PER_DAY     per-tenant outbound call cap per 24 h (default 50)
  VOX_OUTBOUND_PREFIXES    comma list, e.g. +1 : restrict dialing to these country prefixes
  VOX_METRICS_TOKEN        bearer token for GET /metrics
  VOX_RECORD_DIR           where recordings (and the utterance dataset) are written
  VOX_DATASET=1            also save every recorded caller utterance with transcript + end-of-turn score
  VOX_EMBED_URL            OpenAI-compatible embeddings server (e.g. http://127.0.0.1:8001/v1) -> hybrid knowledge-base search
  VOX_EMBED_MODEL          default BAAI/bge-small-en-v1.5;  VOX_EMBED_MIN_SIM  cosine floor for dense matches (default 0)
  VOX_RETENTION_DAYS       erase ended calls older than this for tenants without their own limit (default: keep)
  VOX_NO_ESPEAK=1          run Kokoro without the GPL espeak-ng fallback (English only; Spanish/French/Italian/Portuguese/Hindi need espeak)
  VOX_PRELOAD_LANGS        Kokoro language codes to load at start-up, default a (a,b English; e,f,h,i,p,j,z others; see docs/LANGUAGES.md)
  VOX_TTS_URL              if set, OpenAI-compatible {url}/audio/speech is used instead of Kokoro
  VOX_TTS_MODEL            model field for VOX_TTS_URL (default tts-1);  VOX_TTS_API_KEY  bearer token if that endpoint requires one
  VOX_TTS_RATE             raw s16le rate when VOX_TTS_URL does not return a WAV (default 24000; a WAV header wins)
  VOX_AUDIOSOCKET_PORT / VOX_AUDIOSOCKET_AGENT / VOX_AUDIOSOCKET_HOST   SIP path: TCP port Asterisk streams call audio to, and the agent id that answers when no number is mapped (docs/SIP.md)
  VOX_AUDIOSOCKET_RATE     AudioSocket PCM rate (8000 ulaw, or 16000 with g722/slin16). Default 8000.
  VOX_SIP_TOKEN            shared secret for POST /internal/sip/setup and POST /sip/recover; empty = both endpoints off
  VOX_PROCS                API processes on one host (default 1). Unset or 1: a single uvicorn with --workers 1.
                           Above 1, scripts/deploy.sh starts one process per port 8080..8080+N-1, not uvicorn's
                           shared --workers model: a WebSocket stays on the process that accepted it, and live
                           calls / VAD are per process. Inside one container, do not raise --workers.
                           VOX_PROCS>1 requires VOX_DB to be postgres:// or postgresql://. deploy.sh refuses
                           a SQLite file first: several processes writing it raise "database is locked".
  VOX_ALLOW_UNSIGNED_TWILIO=1   development only
  VOX_EOT, VOX_EOT_THRESHOLD, VOX_END_MS, VOX_PATIENT_MS, VOX_TTS_CONCURRENCY, VOX_LOG_EVENTS: see docs/TUNING.md
Run: uvicorn api.main:app --host 0.0.0.0 --port 8080   (put TLS in front; Twilio needs wss)."""
from __future__ import annotations

import os

import httpx

from api import security
from api.app import Settings, create_app
from api.runtime import Runtime
from api.store import Store
from core.llm import VllmClient
from core.tts import KokoroTTS
from core.vad import SileroVAD


def build():
    env = os.environ
    settings = Settings(admin_key=env.get("VOX_ADMIN_KEY", ""),
                        public_base_url=env.get("VOX_PUBLIC_URL", "http://localhost:8080"),
                        twilio_auth_token=env.get("TWILIO_AUTH_TOKEN", ""),
                        twilio_account_sid=env.get("TWILIO_ACCOUNT_SID", ""),
                        twilio_number=env.get("VOX_TWILIO_NUMBER", ""),
                        metrics_token=env.get("VOX_METRICS_TOKEN", ""),
                        record_dir=env.get("VOX_RECORD_DIR", ""),
                        outbound_per_day=int(env.get("VOX_OUTBOUND_PER_DAY", "50")),
                        outbound_prefixes=tuple(p.strip() for p in env.get("VOX_OUTBOUND_PREFIXES", "").split(",") if p.strip()),
                        allow_unsigned_twilio=env.get("VOX_ALLOW_UNSIGNED_TWILIO") == "1",
                        retention_default_days=int(env.get("VOX_RETENTION_DAYS", "0")),
                        default_concurrency=int(env.get("VOX_DEFAULT_CONCURRENCY", "10")),
                        node_max_calls=int(env.get("VOX_NODE_MAX_CALLS", "32")),
                        overflow_url=env.get("VOX_OVERFLOW_URL", ""),
                        audiosocket_port=int(env.get("VOX_AUDIOSOCKET_PORT", "0")),
                        audiosocket_agent=env.get("VOX_AUDIOSOCKET_AGENT", ""),
                        audiosocket_host=env.get("VOX_AUDIOSOCKET_HOST", "127.0.0.1"),
                        sip_token=env.get("VOX_SIP_TOKEN", ""),
                        audiosocket_rate=int(env.get("VOX_AUDIOSOCKET_RATE", "8000")))
    store = Store(env.get("VOX_DB", "vox.db"))
    llm = VllmClient(base_url=env.get("VOX_VLLM_URL", "http://127.0.0.1:8000/v1"),
                     audio_style=env.get("VOX_AUDIO_STYLE", "input_audio"))
    http = httpx.AsyncClient()
    embedder = None
    if env.get("VOX_EMBED_URL"):
        from core.embed import OpenAIEmbedder
        embedder = OpenAIEmbedder(env["VOX_EMBED_URL"], env.get("VOX_EMBED_MODEL", "BAAI/bge-small-en-v1.5"), http,
                                  min_sim=float(env.get("VOX_EMBED_MIN_SIM", "0")))
    if env.get("VOX_TTS_URL"):
        from core.tts import HttpTTS
        tts_url = env["VOX_TTS_URL"].strip()
        tts_model = env.get("VOX_TTS_MODEL", "tts-1")
        tts_key = env.get("VOX_TTS_API_KEY", "")
        tts_rate = int(env.get("VOX_TTS_RATE", "24000"))

        def make_tts(voice, url=tts_url, model=tts_model, key=tts_key, rate=tts_rate, client=http):
            return HttpTTS(url, client, model=model, voice=voice, api_key=key, sample_rate=rate)
    else:
        from core.tts import PipelineSet
        pipelines = PipelineSet(preload=env.get("VOX_PRELOAD_LANGS", "a"))   # one front end per language, shared by all calls

        def make_tts(voice, _pipelines=pipelines):
            return KokoroTTS(voice=voice, pipeline=_pipelines.for_voice(voice))
    rt = Runtime(llm, make_tts, SileroVAD, store,
                 http=http, record_dir=settings.record_dir, embedder=embedder)   # one pooled client for tools and Twilio
    limiter = security.make_limiter(env, settings.default_concurrency)
    return create_app(store, rt, limiter, security.WebhookDispatcher(httpx.AsyncClient()), settings)


app = build()
