"""Twilio Media Streams transport (bidirectional <Connect><Stream>).

Wire format (verified against Twilio's WebSocket message docs):
  in : {"event":"connected"|"start"|"media"|"mark"|"dtmf"|"stop", ...}
       media.payload = base64 audio/x-mulaw, 8 kHz, mono, no file header
  out: {"event":"media","streamSid":..., "media":{"payload": b64}}
       {"event":"mark","streamSid":..., "mark":{"name":...}}   (echoed back after playback)
       {"event":"clear","streamSid":...}                        (flush buffered audio)
After `clear`, Twilio echoes a mark for every unplayed chunk, so marks are NOT proof of playback
once an interruption has happened; this transport only counts them.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Awaitable, Callable

import numpy as np

from core import dtmf
from core.audio import StreamResampler, mulaw_to_pcm16, pcm16_to_mulaw
from core.session import FRAME_SAMPLES, Session

SessionFactory = Callable[["TwilioTransport", dict], Awaitable[Session]]
MAX_PAYLOAD_SAMPLES = 8000  # 1 s per outbound media message
DTMF_IDLE_S = 2.0           # keypad digits are handed to the agent as one turn after this much quiet, or on '#'


class TwilioTransport:
    out_rate = 8000
    narrowband = True          # mulaw is 8 kHz; nothing above 4 kHz is on the wire

    def __init__(self, ws, session_factory: SessionFactory):
        self.ws = ws                      # needs: async receive_text() / async send_text(str)
        self.session_factory = session_factory
        self.session: Session | None = None
        self.stream_sid = ""
        self.call_sid = ""
        self.custom: dict = {}
        self.marks_sent = 0
        self.marks_acked = 0
        self._send_lock = asyncio.Lock()
        self._rs = StreamResampler(8000, 16000)
        self._buf = np.zeros(0, dtype=np.int16)
        self._digits = ""
        self._dtmf_task: asyncio.Task | None = None
        self.ended_by_us = False          # True once the agent (not the caller or the network) ended the call
        self.before_end = None            # optional hook run before the stream is closed (records why, for call recovery)

    async def _send(self, obj: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj))

    # --- called by Session ---
    async def send_audio(self, pcm: np.ndarray, rate: int) -> None:
        assert rate == 8000, "Twilio transport expects 8 kHz output"
        for i in range(0, len(pcm), MAX_PAYLOAD_SAMPLES):
            payload = base64.b64encode(pcm16_to_mulaw(pcm[i : i + MAX_PAYLOAD_SAMPLES])).decode()
            await self._send({"event": "media", "streamSid": self.stream_sid,
                              "media": {"payload": payload}})
        self.marks_sent += 1
        await self._send({"event": "mark", "streamSid": self.stream_sid,
                          "mark": {"name": f"m{self.marks_sent}"}})

    async def clear(self) -> None:
        await self._send({"event": "clear", "streamSid": self.stream_sid})

    async def send_dtmf(self, digits: str) -> None:
        """Press keypad digits on the call by sending DTMF tones as audio."""
        await self.send_audio(dtmf.sequence(digits, 8000), 8000)

    def _on_dtmf(self, digit: str) -> None:
        """Collects keypad presses; '#' or a pause ends the entry and it reaches the agent as a text turn."""
        if not digit:
            return
        if self._dtmf_task:
            self._dtmf_task.cancel()
        if digit == "#":
            self._dtmf_task = asyncio.create_task(self._flush_digits(0.0))
            return
        self._digits += digit
        self._dtmf_task = asyncio.create_task(self._flush_digits(DTMF_IDLE_S))

    async def _flush_digits(self, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        digits, self._digits = self._digits, ""
        if digits and self.session is not None:
            await self.session.inject_text(f"[The caller pressed these keypad digits: {digits}]")

    async def end(self) -> None:
        """Hang up. The reason is recorded first, because Twilio asks the server what to do next the moment the stream
        closes (see /twilio/resume): the server must already know this close is deliberate."""
        self.ended_by_us = True
        if self.before_end is not None:
            r = self.before_end()
            if asyncio.iscoroutine(r):
                await r
        close = getattr(self.ws, "close", None)
        if close is not None:
            await close()

    # --- receive loop ---
    async def run(self) -> None:
        try:
            while True:
                msg = json.loads(await self.ws.receive_text())
                ev = msg.get("event")
                if ev == "start":
                    st = msg["start"]
                    self.stream_sid = st["streamSid"]
                    self.call_sid = st.get("callSid", "")
                    self.custom = st.get("customParameters", {}) or {}
                    self.session = await self.session_factory(self, self.custom)
                    await self.session.start()
                elif ev == "media" and self.session is not None:
                    m = msg["media"]
                    if m.get("track", "inbound") == "inbound":
                        await self._on_media(m["payload"])
                elif ev == "dtmf":
                    self._on_dtmf(str((msg.get("dtmf") or {}).get("digit", "")))
                elif ev == "mark":
                    self.marks_acked += 1
                elif ev == "stop":
                    break
        except EOFError:          # socket adapter signals a peer disconnect
            pass
        finally:
            if self._dtmf_task:
                self._dtmf_task.cancel()
            if self.session is not None:
                await self.session.close()

    async def _on_media(self, payload_b64: str) -> None:
        pcm8 = mulaw_to_pcm16(base64.b64decode(payload_b64))
        self._buf = np.concatenate([self._buf, self._rs.push(pcm8)])
        while len(self._buf) >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            await self.session.feed(frame)  # type: ignore[union-attr]
