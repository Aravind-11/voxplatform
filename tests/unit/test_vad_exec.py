"""VAD scoring stays off the event loop and batches frames that share a loop beat."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from api.runtime import Runtime
from api.store import Store
from core.metrics import Metrics
from core.vad_exec import EXECUTOR, VADScheduler, current_batch, watch_loop_lag

FRAME = np.zeros(512, np.int16)


class _SpyPool(ThreadPoolExecutor):
    def __init__(self):
        super().__init__(max_workers=2, thread_name_prefix="vox-vad-test")
        self.jobs = 0
        self._token = 0

    def submit(self, fn, *args, **kwargs):
        self.jobs += 1
        token = self.jobs

        def wrapped():
            self._token = token
            return fn(*args, **kwargs)

        return super().submit(wrapped)


async def test_same_beat_probs_run_inside_one_executor_call():
    pool = _SpyPool()
    try:
        sched = VADScheduler(pool)
        seen: list[int] = []

        class V:
            def prob(self, frame):
                seen.append(pool._token)
                assert current_batch() is not None
                return 0.42

        v1, v2 = V(), V()

        async def same_beat():
            # Both coroutines are scheduled before either is awaited past the flush point.
            a = asyncio.create_task(sched.prob(v1, FRAME))
            b = asyncio.create_task(sched.prob(v2, FRAME))
            await asyncio.sleep(0)
            return await asyncio.gather(a, b)

        assert await same_beat() == [0.42, 0.42]
        assert pool.jobs == 1
        assert seen == [1, 1]
    finally:
        pool.shutdown(wait=True)


async def test_slow_prob_does_not_stall_the_event_loop():
    sched = VADScheduler()
    order: list[str] = []

    class Slow:
        def prob(self, frame):
            time.sleep(0.3)
            return 0.2

    async def tick():
        await asyncio.sleep(0.05)
        order.append("tick")

    async def scored():
        await sched.prob(Slow(), FRAME)
        order.append("vad")

    await asyncio.gather(scored(), tick())
    assert order == ["tick", "vad"]


async def test_watch_loop_lag_gauge_rises_when_the_loop_is_blocked():
    metrics = Metrics()
    task = asyncio.create_task(watch_loop_lag(metrics))
    try:
        await asyncio.sleep(0.12)
        assert metrics.gauges["vox_loop_lag_ms"]() >= 0
        assert "late" in metrics.help["vox_loop_lag_ms"]
        time.sleep(0.25)                 # blocks the loop; the in-flight tick is now late
        lag = 0.0
        for _ in range(5):              # the overdue tick runs on a later loop turn than sleep(0)
            await asyncio.sleep(0.02)
            lag = max(lag, metrics.gauges["vox_loop_lag_ms"]())
        assert lag >= 50
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_executor_is_the_named_vox_vad_pool():
    assert EXECUTOR._max_workers == 2
    assert EXECUTOR._thread_name_prefix == "vox-vad"


def test_runtime_session_uses_the_process_scheduler_and_rigs_do_not():
    class TTS:
        sample_rate = 8000

    store = Store()
    rt = Runtime(object(), lambda v: TTS(), lambda: None, store)
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "webSocket", {"systemPrompt": "hi"})

    class T:
        out_rate = 8000

        async def send_audio(self, pcm, rate): ...

        async def clear(self): ...

        async def end(self): ...

    sess = rt.build_session(call, T())
    assert sess.vad_scheduler is rt.vad_scheduler

    from core.session import Session, SessionConfig

    class Inline:
        def prob(self, frame):
            return 0.0

    bare = Session(llm=object(), tts=object(), vad=Inline(), send_audio=T.send_audio, clear_audio=T.clear,
                   clock=lambda: 0.0, cfg=SessionConfig())
    assert bare.vad_scheduler is None
