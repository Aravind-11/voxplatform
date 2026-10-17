"""Run Silero off the event loop.

One process-wide pool (`vox-vad`, two workers). Frames that arrive before the next loop turn
share a single executor job: each caller still awaits its own future, but `vad.prob` for that
batch runs back-to-back on one worker so a burst of calls does not enqueue one job per frame.
"""
from __future__ import annotations

import asyncio
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Named pool. Threads show up as vox-vad_0 / vox-vad_1. Shared by every VADScheduler.
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vox-vad")

_batch_ids = itertools.count(1)
_local = threading.local()


def current_batch() -> int | None:
    """Id of the executor job currently calling `vad.prob`, else None.

    Probs that shared a job see the same id. Outside a job this is None.
    """
    return getattr(_local, "batch_id", None)


def _score(items: list[tuple[Any, Any]]) -> list[tuple[bool, Any]]:
    _local.batch_id = next(_batch_ids)
    out: list[tuple[bool, Any]] = []
    for vad, frame in items:
        try:
            out.append((True, float(vad.prob(frame))))
        except Exception as exc:  # noqa: BLE001  (hand the failure to that caller only)
            out.append((False, exc))
    return out


class VADScheduler:
    """Micro-batches `vad.prob` onto `EXECUTOR`."""

    def __init__(self, executor: ThreadPoolExecutor | None = None) -> None:
        self._executor = executor or EXECUTOR
        self._queue: list[tuple[Any, Any, asyncio.Future]] = []
        self._flush_scheduled = False

    async def prob(self, vad: Any, frame: Any) -> float:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.append((vad, frame, fut))
        if not self._flush_scheduled:
            # Next loop turn, so every prob() scheduled in this beat joins the same job.
            self._flush_scheduled = True
            loop.call_soon(self._flush)
        result = await fut
        return float(result)

    def _flush(self) -> None:
        self._flush_scheduled = False
        batch = [item for item in self._queue if not item[2].done()]
        self._queue = []
        if not batch:
            return
        loop = asyncio.get_running_loop()
        futs = [fut for _, _, fut in batch]
        items = [(vad, frame) for vad, frame, _ in batch]

        async def finish() -> None:
            try:
                vals = await loop.run_in_executor(self._executor, _score, items)
            except Exception as exc:  # noqa: BLE001
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(exc)
                return
            for fut, (ok, val) in zip(futs, vals):
                if fut.done():
                    continue
                if ok:
                    fut.set_result(val)
                else:
                    fut.set_exception(val)

        asyncio.create_task(finish())


async def watch_loop_lag(metrics: Any) -> None:
    """Every 100 ms, record how late this coroutine woke (`max(0, drift_ms)`) as `vox_loop_lag_ms`."""
    cell = [0.0]
    metrics.describe("vox_loop_lag_ms", "Event-loop lag: milliseconds the 100 ms tick woke up late")
    metrics.gauge("vox_loop_lag_ms", lambda: cell[0])
    loop = asyncio.get_running_loop()
    period = 0.1
    while True:
        start = loop.time()
        await asyncio.sleep(period)
        cell[0] = max(0.0, (loop.time() - start - period) * 1000.0)
