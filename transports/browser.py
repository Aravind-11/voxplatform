"""Browser WebSocket transport (demo client + SDK-style access).

Protocol
  client -> server : binary frames, int16 little-endian mono PCM at 16 kHz (any frame size)
  client -> server : text frames (JSON): {"type":"client_tool_result","id":...,"result":...} answering a
                     {"type":"client_tool_invocation","id","name","args"} the server sent (see client tools)
  server -> client : binary frames, int16 mono PCM at `out_rate` (default 24 kHz, no resampling)
                     text frames, JSON events: {"type":"clear"} (flush your playback queue),
                     plus session events forwarded by the API layer (transcripts, state).
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import numpy as np

from core.session import FRAME_SAMPLES, Session

SessionFactory = Callable[["BrowserTransport"], Awaitable[Session]]


async def _as_messages(ws):
    async for data in ws.iter_bytes():
        yield "bytes", data


class BrowserTransport:
    out_rate = 24000
    narrowband = False         # browser audio is 16 kHz in and 24 kHz out, not a telephone codec

    def __init__(self, ws, session_factory: SessionFactory):
        self.ws = ws                      # needs: iter_bytes(), send_bytes(b), send_text(s)
        self.session_factory = session_factory
        self.session: Session | None = None
        self._buf = np.zeros(0, dtype=np.int16)
        self._rem = b""                   # odd trailing byte carried to the next message
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self._seq = 0

    async def client_call(self, name: str, args: dict, timeout_s: float = 10.0):
        """Runs a tool in the caller's browser: sends an invocation and waits for the matching result."""
        self._seq += 1
        cid = f"ct{self._seq}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = fut
        try:
            await self.send_event({"type": "client_tool_invocation", "id": cid, "name": name, "args": args})
            return await asyncio.wait_for(fut, timeout_s)
        except asyncio.TimeoutError:
            return {"error": f"client tool {name} did not answer within {timeout_s:g}s"}
        finally:
            self._pending.pop(cid, None)

    def _on_text(self, raw: str) -> None:
        try:
            m = json.loads(raw)
        except ValueError:
            return
        if isinstance(m, dict) and m.get("type") == "client_tool_result":
            fut = self._pending.get(str(m.get("id")))
            if fut and not fut.done():
                fut.set_result({"error": str(m["error"])[:500]} if m.get("error") else m.get("result"))

    async def send_audio(self, pcm: np.ndarray, rate: int) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(pcm.astype("<i2").tobytes())

    async def clear(self) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps({"type": "clear"}))

    async def end(self) -> None:
        close = getattr(self.ws, "close", None)
        if close is not None:
            await close()

    async def send_event(self, event: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(event))

    async def run(self) -> None:
        self.session = await self.session_factory(self)
        await self.session.start()
        try:
            source = self.ws.iter_messages() if hasattr(self.ws, "iter_messages") else _as_messages(self.ws)
            async for kind, data in source:
                if kind == "text":
                    self._on_text(data)
                    continue
                data = self._rem + data
                self._rem = data[len(data) // 2 * 2 :]
                chunk = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
                self._buf = np.concatenate([self._buf, chunk])
                while len(self._buf) >= FRAME_SAMPLES:
                    frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
                    await self.session.feed(frame)
        finally:
            for f in self._pending.values():
                if not f.done():
                    f.cancel()
            await self.session.close()
