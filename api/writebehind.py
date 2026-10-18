"""Background transcript log so a slow database write cannot stall the event loop.

`flush()` blocks until the queued rows are written (`Queue.join`) and raises if an
insert failed. Call it from a coroutine with `to_thread` — a direct call on the loop
freezes every call.
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
        self._cv = threading.Condition()
        self._closed = False
        self._drained = False
        self._late: list[tuple] = []
        self._errors: list[tuple[str, BaseException]] = []
        self._thread = threading.Thread(target=self._run, name="vox-messages", daemon=True)
        self._thread.start()

    def add(self, call_id: str, role: str, text: str, t: float) -> None:
        """Enqueue one transcript row and return. The daemon thread does the insert."""
        item = (call_id, role, text, t)
        with self._cv:
            if not self._closed:
                self._q.put(item)
                return
            # The stop sentinel is already queued. A put behind it is never written,
            # and flush() would wait forever for a task_done that never comes.
            if not self._drained:
                self._late.append(item)
                self._cv.notify_all()
                return
        self._insert(item)

    def _insert(self, item: tuple) -> None:
        call_id, role, text, t = item
        try:
            self.store.add_message(call_id, role, text, t)
        except Exception as exc:  # noqa: BLE001  (one bad insert must not kill the writer)
            METRICS.inc("vox_errors_total", where="message_log")
            with self._cv:
                self._errors.append((call_id, exc))

    def _take_error(self, call_id: str | None = None) -> BaseException | None:
        """Caller holds self._cv. Pop this call's failure, or the oldest if `call_id` is omitted.

        Another call's flush must not consume this one: that would let the failed call end cleanly.
        """
        if call_id is None:
            if not self._errors:
                return None
            return self._errors.pop(0)[1]
        for i, (cid, exc) in enumerate(self._errors):
            if cid == call_id:
                del self._errors[i]
                return exc
        return None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._stop:
                    return
                self._insert(item)
            finally:
                self._q.task_done()

    def _take_late(self) -> list[tuple]:
        """Caller holds self._cv."""
        batch = list(self._late)
        self._late.clear()
        return batch

    def _until_idle(self) -> None:
        """Block until every `add` so far has been written. Caller must not hold `self._cv`."""
        while True:
            self._q.join()
            with self._cv:
                if self._q.unfinished_tasks:
                    continue
                if self._late or (self._closed and not self._drained):
                    self._cv.wait()
                    continue
                return

    def wait(self) -> None:
        """Block until queued rows are in the store. Never call this on the event loop.

        Does not consume insert errors. A reader that called `flush` would take the failure,
        and `finish` would then report the lost transcript as a clean hang-up.
        """
        self._until_idle()

    def flush(self, call_id: str | None = None) -> None:
        """Block until every `add` so far has been written. Never call this on the event loop.

        Raises this call's insert error (or the oldest, if `call_id` is omitted). The writer keeps running.
        """
        while True:
            self._q.join()
            with self._cv:
                if self._q.unfinished_tasks:
                    continue
                if self._late or (self._closed and not self._drained):
                    self._cv.wait()
                    continue
                err = self._take_error(call_id)
                break
        if err is not None:
            raise err

    def close(self) -> None:
        """Stop the writer after it drains the queue. A failed insert is raised once."""
        with self._cv:
            leader = not self._closed and not self._drained
            if leader:
                self._closed = True
            elif not self._drained:
                while not self._drained:
                    self._cv.wait()
        if not leader:
            with self._cv:
                err = self._take_error()
            if err is not None:
                raise err
            return
        err: BaseException | None = None
        try:
            while True:
                self._q.join()
                with self._cv:
                    if self._q.unfinished_tasks:
                        continue
                    batch = self._take_late()
                if not batch:
                    break
                for item in batch:
                    self._insert(item)
            self._q.put(self._stop)
            self._thread.join()  # no timeout: a slow final insert must not be abandoned
            while True:
                with self._cv:
                    batch = self._take_late()
                    if not batch:
                        self._drained = True
                        err = self._take_error()
                        self._cv.notify_all()
                        break
                for item in batch:
                    self._insert(item)
        except BaseException:
            with self._cv:
                if not self._drained:
                    self._drained = True
                    self._cv.notify_all()
            raise
        if err is not None:
            raise err


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
