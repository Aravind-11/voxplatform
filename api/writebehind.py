"""Background transcript log so a slow database write cannot stall the event loop.

`flush()` blocks until the queued rows are written (`Queue.join`). Call it from a
coroutine with `asyncio.to_thread` — a direct call on the loop freezes every call.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Callable

from core.metrics import METRICS


class MessageLog:
    def __init__(self, store: Any):
        self.store = store
        self._q: queue.Queue = queue.Queue()
        self._stop = object()
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="vox-messages", daemon=True)
        self._thread.start()

    def add(self, call_id: str, role: str, text: str, t: float) -> None:
        """Enqueue one transcript row and return. The daemon thread does the insert."""
        self._q.put((call_id, role, text, t))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._stop:
                    return
                call_id, role, text, t = item
                try:
                    self.store.add_message(call_id, role, text, t)
                except Exception:  # noqa: BLE001  (one bad insert must not kill the writer)
                    METRICS.inc("vox_errors_total", where="message_log")
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """Block until every `add` so far has been written. Never call this on the event loop."""
        self._q.join()

    def close(self) -> None:
        """Stop the writer after it drains the queue."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._q.put(self._stop)
        self._thread.join(timeout=5)


async def to_thread(fn: Callable, /, *args: Any, **kwargs: Any) -> Any:
    """`asyncio.to_thread` that still finishes when this task is already cancelled.

    Starlette cancels the websocket handler as the client disconnects, before `finally`
    runs. A plain `await asyncio.to_thread(...)` then raises immediately and the transcript
    flush never happens. Clearing a pending cancel around the worker await lets the write
    finish; `task.cancel()` puts the cancel back so the caller's next await still ends.
    """
    task = asyncio.current_task()
    job = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    saw = False
    try:
        while True:
            if task is not None:
                while task.cancelling():
                    saw = True
                    task.uncancel()
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                saw = True
                if job.done():
                    return job.result()
    finally:
        if saw and task is not None and task.cancelling() == 0:
            task.cancel()
