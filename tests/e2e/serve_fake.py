"""Starts the real API app with fake model/TTS/VAD so browser and HTTP end-to-end tests need no GPU.
Usage: python tests/e2e/serve_fake.py PORT   (honours VOX_ADMIN_KEY, VOX_PUBLIC_URL, VOX_METRICS_TOKEN)"""
import os
import sys
import tempfile

import numpy as np
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api import security
from api.app import Settings, create_app
from api.runtime import Runtime
from api.store import Store


class FakeVAD:
    def prob(self, frame):
        return 0.95 if np.abs(frame).max() > 1500 else 0.02


class FakeLLM:
    n = 0

    async def stream(self, history, audio, followups, tools):
        self.n += 1
        yield "Nice to meet you. "
        yield f"That was turn {self.n}."

    async def transcribe(self, audio):
        return "hello agent"


class FakeTTS:
    sample_rate = 24000

    async def synthesize(self, text):
        t = np.arange(int(24000 * 0.05 * len(text.split()))) / 24000
        yield (3000 * np.sin(2 * np.pi * 330 * t)).astype(np.int16)


def main():
    port = int(sys.argv[1])
    rec = tempfile.mkdtemp()
    os.environ["VOX_RECORD_DIR"] = rec
    store = Store(os.environ.get("VOX_DB", ":memory:"))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    s = Settings(admin_key=os.environ.get("VOX_ADMIN_KEY", "adm"),
                 public_base_url=os.environ.get("VOX_PUBLIC_URL", f"http://127.0.0.1:{port}"),
                 metrics_token=os.environ.get("VOX_METRICS_TOKEN", "mt"), record_dir=rec,
                 allow_unsigned_twilio=os.environ.get("VOX_ALLOW_UNSIGNED_TWILIO") == "1")
    redis_url = os.environ.get("VOX_REDIS_URL", "")
    limiter = security.RedisLimiter(redis_url, 5, ttl_s=float(os.environ.get("VOX_REDIS_TTL", "90"))) if redis_url else security.ConcurrencyLimiter(5)
    uvicorn.run(create_app(store, rt, limiter, None, s), host=os.environ.get("VOX_HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
