"""Production wiring (GPU box). Env: VOX_ADMIN_KEY, VOX_PUBLIC_URL, VOX_DB, TWILIO_AUTH_TOKEN,
VOX_VLLM_URL, VOX_AUDIO_STYLE, VOX_DEFAULT_CONCURRENCY, VOX_ALLOW_UNSIGNED_TWILIO,
TWILIO_ACCOUNT_SID (only for the transfer tool).
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
                        allow_unsigned_twilio=env.get("VOX_ALLOW_UNSIGNED_TWILIO") == "1",
                        default_concurrency=int(env.get("VOX_DEFAULT_CONCURRENCY", "10")))
    store = Store(env.get("VOX_DB", "vox.db"))
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")               # loaded once, shared by all calls
    llm = VllmClient(base_url=env.get("VOX_VLLM_URL", "http://127.0.0.1:8000/v1"),
                     audio_style=env.get("VOX_AUDIO_STYLE", "input_audio"))
    rt = Runtime(llm, lambda voice: KokoroTTS(voice=voice, pipeline=pipeline), SileroVAD, store,
                 http=httpx.AsyncClient())            # one pooled client for tool calls and Twilio transfer
    limiter = security.ConcurrencyLimiter(settings.default_concurrency)
    return create_app(store, rt, limiter, security.WebhookDispatcher(httpx.AsyncClient()), settings)


app = build()
