"""Browser WebSocket transport (demo client + SDK-style access).

Protocol
  client -> server : binary frames, int16 little-endian mono PCM at 16 kHz (any frame size)
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


class BrowserTransport:
    out_rate = 24000

    def __init__(self, ws, session_factory: SessionFactory):
        self.ws = ws                      # needs: iter_bytes(), send_bytes(b), send_text(s)
        self.session_factory = session_factory
        self.session: Session | None = None
        self._buf = np.zeros(0, dtype=np.int16)
        self._rem = b""                   # odd trailing byte carried to the next message
        self._send_lock = asyncio.Lock()

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
            async for data in self.ws.iter_bytes():
                data = self._rem + data
                self._rem = data[len(data) // 2 * 2 :]
                chunk = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
                self._buf = np.concatenate([self._buf, chunk])
                while len(self._buf) >= FRAME_SAMPLES:
                    frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
                    await self.session.feed(frame)
        finally:
            await self.session.close()
