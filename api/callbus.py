"""Cross-node messages for one live call.

AMD (answering-machine detection) is an HTTP webhook. The load balancer may send it to a
node that does not hold the WebSocket. That node publishes; the node that owns the call is
subscribed and speaks the voicemail. In-process subscribers are a LocalCallBus. With
VOX_REDIS_URL, every node also listens on Redis channel ``vox:call:{call_id}``.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Callable, Optional

from core.metrics import METRICS

Handler = Callable[[dict], None]


class CallBus:
    def publish(self, call_id: str, msg: dict) -> bool:
        """Deliver ``msg`` to subscribers of ``call_id``. True when at least one subscriber exists."""
        raise NotImplementedError

    def subscribe(self, call_id: str, handler: Handler) -> None:
        raise NotImplementedError

    def unsubscribe(self, call_id: str, handler: Optional[Handler] = None) -> None:
        """Drop ``handler``, or every handler for the call when ``handler`` is omitted."""
        raise NotImplementedError


class LocalCallBus(CallBus):
    """In-process delivery. Handlers run on the publisher's thread."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._lock = threading.Lock()

    def subscribe(self, call_id: str, handler: Handler) -> None:
        with self._lock:
            self._handlers.setdefault(call_id, []).append(handler)

    def unsubscribe(self, call_id: str, handler: Optional[Handler] = None) -> None:
        with self._lock:
            if handler is None:
                self._handlers.pop(call_id, None)
                return
            left = [h for h in self._handlers.get(call_id, ()) if h is not handler]
            if left:
                self._handlers[call_id] = left
            else:
                self._handlers.pop(call_id, None)

    def has(self, call_id: str) -> bool:
        with self._lock:
            return bool(self._handlers.get(call_id))

    def call_ids(self) -> list[str]:
        with self._lock:
            return list(self._handlers)

    def publish(self, call_id: str, msg: dict) -> bool:
        with self._lock:
            handlers = list(self._handlers.get(call_id, ()))
        for h in handlers:
            try:
                h(dict(msg))
            except Exception:  # noqa: BLE001  (one bad handler must not stop the others)
                METRICS.inc("vox_errors_total", where="callbus")
        return bool(handlers)


class RedisCallBus(CallBus):
    """Redis pub/sub on ``vox:call:{call_id}``, plus in-process handlers for this node.

    ``publish`` returns True when this process has a handler or Redis reports a subscriber
    (the owning node's listener). Messages this process publishes are stamped with ``origin``
    so the listener does not deliver them again — local handlers already ran inside ``publish``.
    """

    def __init__(self, url: str = "", client=None, start_thread: bool = True):
        if client is None:
            import redis
            client = redis.Redis.from_url(url, decode_responses=True)
        self.r = client
        self.origin = uuid.uuid4().hex
        self._local = LocalCallBus()
        self._commands: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="vox-callbus", daemon=True)
            self._thread.start()

    @staticmethod
    def _channel(call_id: str) -> str:
        return f"vox:call:{call_id}"

    def subscribe(self, call_id: str, handler: Handler) -> None:
        first = not self._local.has(call_id)
        self._local.subscribe(call_id, handler)
        if first:
            self._commands.put(("sub", self._channel(call_id)))

    def unsubscribe(self, call_id: str, handler: Optional[Handler] = None) -> None:
        self._local.unsubscribe(call_id, handler)
        if not self._local.has(call_id):
            self._commands.put(("unsub", self._channel(call_id)))

    def publish(self, call_id: str, msg: dict) -> bool:
        body = dict(msg)
        local = self._local.publish(call_id, body)
        n = 0
        try:
            n = int(self.r.publish(self._channel(call_id),
                                   json.dumps({"origin": self.origin, "msg": body})) or 0)
        except Exception:  # noqa: BLE001  (Redis blip: the local delivery result still stands)
            METRICS.inc("vox_errors_total", where="callbus")
        return bool(local or n > 0)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _on_wire(self, channel, raw) -> None:
        if isinstance(channel, bytes):
            channel = channel.decode()
        if isinstance(raw, bytes):
            raw = raw.decode()
        prefix = "vox:call:"
        if not isinstance(channel, str) or not channel.startswith(prefix):
            return
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return
        elif isinstance(raw, dict):
            data = raw
        else:
            return
        if not isinstance(data, dict) or data.get("origin") == self.origin:
            return
        msg = data.get("msg")
        if isinstance(msg, dict):
            self._local.publish(channel[len(prefix):], msg)

    def _pull_desired(self) -> set[str]:
        """Channels to subscribe after a (re)connect, including commands already queued."""
        pending = []
        while True:
            try:
                pending.append(self._commands.get_nowait())
            except queue.Empty:
                break
        channels = {self._channel(cid) for cid in self._local.call_ids()}
        for op, channel in pending:
            if op == "sub":
                channels.add(channel)
            else:
                channels.discard(channel)
        return channels

    def _apply(self, ps, channels: set[str]) -> None:
        while True:
            try:
                op, channel = self._commands.get_nowait()
            except queue.Empty:
                return
            if op == "sub":
                if channel not in channels:
                    ps.subscribe(channel)
                    channels.add(channel)
            elif channel in channels:
                ps.unsubscribe(channel)
                channels.discard(channel)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ps = self.r.pubsub(ignore_subscribe_messages=True)
                channels = self._pull_desired()
                for ch in sorted(channels):
                    ps.subscribe(ch)
                while not self._stop.is_set():
                    self._apply(ps, channels)
                    if not channels:
                        if self._stop.wait(0.2):
                            return
                        continue
                    msg = ps.get_message(timeout=0.2)
                    if msg and msg.get("type") == "message":
                        self._on_wire(msg.get("channel", ""), msg.get("data", ""))
            except Exception:  # noqa: BLE001  (reconnect; a dead Redis must not kill the process)
                METRICS.inc("vox_errors_total", where="callbus")
                if self._stop.wait(1.0):
                    return


def make_bus(redis_url: str = "") -> CallBus:
    """Local bus when ``redis_url`` is empty, otherwise Redis pub/sub."""
    if not redis_url:
        return LocalCallBus()
    return RedisCallBus(url=redis_url)
