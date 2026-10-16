"""Runs the real FastAPI app on a socket with fake model components and drives it with the synthetic caller."""
import asyncio
import threading
import time

import numpy as np
import pytest
import uvicorn

from api import security
from api.app import Settings, create_app
from api.runtime import Runtime
from api.store import Store
from harness.caller import run_call, run_suite


class VAD:
    def prob(self, f):
        return 1.0 if np.abs(f).max() > 1000 else 0.0


class SlowLLM:
    def __init__(self, delay): self.delay = delay

    async def stream(self, h, a, f, t):
        await asyncio.sleep(self.delay)
        yield "This is a fairly long reply that keeps going. "
        yield "And another sentence follows it. "
        yield "And a third one for good measure. "

    async def transcribe(self, a): return "x"


class TTS:
    sample_rate = 24000

    async def synthesize(self, text):
        for _ in range(4):
            await asyncio.sleep(0.01)
            yield np.full(12000, 800, dtype=np.int16)      # 0.5 s each


@pytest.fixture
def server():
    store = Store()
    rt = Runtime(SlowLLM(0.05), lambda v: TTS(), VAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(50), None,
                     Settings(admin_key="a", public_base_url="http://127.0.0.1:18080"))
    cfg = uvicorn.Config(app, host="127.0.0.1", port=18080, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started: break
        time.sleep(0.05)
    tid = store.create_tenant("t", 50)
    key = security.generate_key()
    store.add_key(tid, security.hash_key(key))
    yield "http://127.0.0.1:18080", key
    srv.should_exit = True
    th.join(5)


SPEECH = np.full(16000, 5000, dtype=np.int16)           # 1 s of "speech"


async def test_caller_measures_response_latency(server):
    base, key = server
    out = await run_suite(base, key, SPEECH, n=3, barge=False)
    assert out["ok"] == 3, out
    # end-of-turn silence (640 ms) + LLM 50 ms + tts ~ 10 ms, plus pacing slop
    assert 500 < out["p50"] < 1500, out


async def test_caller_measures_barge_in_silence_time(server):
    base, key = server
    out = await run_suite(base, key, SPEECH, n=2, barge=True)
    assert out["ok"] == 2, out
    assert out["p95"] < 600, out
