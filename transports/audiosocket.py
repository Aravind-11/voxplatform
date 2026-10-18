"""Asterisk AudioSocket transport: the SIP path (VoxSun and any other SIP provider or PBX).

SIP/RTP is handled by Asterisk (a separate process, see deploy/sip/); Asterisk streams each call's audio to this server
over one plain TCP connection using the AudioSocket protocol (Asterisk docs: AudioSocket, `AudioSocket(uuid,host:port)`).
Wire format: every message is  kind (1 byte) | length (2 bytes, big endian) | payload.
  0x01 UUID (16 bytes, first message)   0x10 audio (signed 16-bit little-endian PCM, mono, 8 kHz, 20 ms = 320 bytes)
  0x12 audio (the same PCM at 16 kHz, 20 ms = 640 bytes; Asterisk plays 0x10 as 8 kHz)
  0x03 DTMF (1 ASCII char)              0x00 hangup (either direction)      0xff error (1 byte code)
Asterisk plays audio at the rate it arrives, so this transport PACES outbound frames at real time (20 ms each) from its
own queue; that queue is what `clear()` empties on an interruption.
"""
from __future__ import annotations

import asyncio
import struct
import uuid as uuidlib
from typing import Awaitable, Callable

import numpy as np

from core import dtmf
from core.audio import StreamResampler
from core.session import FRAME_SAMPLES, Session

KIND_HANGUP, KIND_UUID, KIND_DTMF, KIND_AUDIO, KIND_ERROR = 0x00, 0x01, 0x03, 0x10, 0xFF
KIND_SLIN16 = 0x12         # 16 kHz signed-linear; 0x10 is 8 kHz on every Asterisk that speaks AudioSocket
FRAME_BYTES = 320          # 20 ms of 8 kHz 16-bit mono; 16 kHz uses sample_rate // 50 * 2
FRAME_S = 0.02
DTMF_IDLE_S = 2.0
SessionFactory = Callable[["AudioSocketTransport", dict], Awaitable[Session]]


def pack(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


async def read_message(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """One message; raises EOFError when the peer closed."""
    try:
        head = await reader.readexactly(3)
        kind, n = struct.unpack(">BH", head)
        return kind, (await reader.readexactly(n) if n else b"")
    except (asyncio.IncompleteReadError, ConnectionError) as e:
        raise EOFError from e


class AudioSocketTransport:
    out_rate = 8000

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, session_factory: SessionFactory,
                 sample_rate: int = 8000):
        self.reader, self.writer = reader, writer
        self.session_factory = session_factory
        self.session: Session | None = None
        self.call_sid = ""                 # the AudioSocket UUID
        self.custom: dict = {}
        self.ended_by_us = False
        self.before_end = None
        self.out_rate = sample_rate        # 8000 ulaw/alaw, or 16000 when the dialplan is slin16 / g722
        self._frame_bytes = sample_rate // 50 * 2    # 20 ms of 16-bit mono
        self._rs = StreamResampler(sample_rate, 16000)   # equal rates are a no-op (no 8 kHz upsample)
        self._buf = np.zeros(0, dtype=np.int16)
        self._out: asyncio.Queue[bytes] = asyncio.Queue()
        self._pending = b""
        self._pacer: asyncio.Task | None = None
        self._digits = ""
        self._dtmf_task: asyncio.Task | None = None
        self._closed = False

    # --- called by Session ---
    async def send_audio(self, pcm: np.ndarray, rate: int) -> None:
        assert rate == self.out_rate, f"AudioSocket transport expects {self.out_rate} Hz output"
        data = self._pending + np.asarray(pcm, dtype="<i2").tobytes()
        frame_bytes = self._frame_bytes
        n = len(data) - len(data) % frame_bytes
        for i in range(0, n, frame_bytes):
            self._out.put_nowait(data[i:i + frame_bytes])
        self._pending = data[n:]
        if self._pending and self._out.empty():            # flush the tail of a reply as a padded last frame
            self._out.put_nowait(self._pending.ljust(frame_bytes, b"\x00"))
            self._pending = b""

    async def clear(self) -> None:
        self._pending = b""
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def send_dtmf(self, digits: str) -> None:
        await self.send_audio(dtmf.sequence(digits, self.out_rate), self.out_rate)

    async def _pace(self) -> None:
        """Sends one queued frame every 20 ms (silence is not sent: Asterisk fills gaps itself)."""
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            frame = await self._out.get()
            now = loop.time()
            if nxt < now - 0.2:
                nxt = now                                   # we were idle: restart the clock instead of bursting
            await asyncio.sleep(max(0.0, nxt - now))
            nxt += FRAME_S
            if self._closed:
                return
            self.writer.write(pack(self._audio_kind(), frame))
            await self.writer.drain()

    def _on_dtmf(self, digit: str) -> None:
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
        self.ended_by_us = True
        if self.before_end is not None:
            r = self.before_end()
            if asyncio.iscoroutine(r):
                await r
        await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.writer.write(pack(KIND_HANGUP))
            await self.writer.drain()
            self.writer.close()
        except Exception:  # noqa: BLE001  (peer already gone)
            pass

    def _abort(self) -> None:
        """Drop the socket without draining. A peer that never reads would block _close() forever."""
        self._closed = True
        try:
            transport = self.writer.transport
            if transport is not None:
                transport.abort()
        except Exception:  # noqa: BLE001
            pass

    # --- receive loop ---
    async def run(self, *, uuid_timeout: float | None = None) -> None:
        try:
            try:
                first = read_message(self.reader)
                if uuid_timeout is not None:
                    kind, payload = await asyncio.wait_for(first, uuid_timeout)
                else:
                    kind, payload = await first
            except TimeoutError:
                # The UUID is the first frame Asterisk sends. A stall here is not a live call.
                self._abort()
                return
            if kind != KIND_UUID or len(payload) != 16:
                raise EOFError("first AudioSocket message must carry the call UUID")
            self.call_sid = str(uuidlib.UUID(bytes=payload))
            self.session = await self.session_factory(self, {"uuid": self.call_sid})
            self._pacer = asyncio.create_task(self._pace())
            await self.session.start()
            while True:
                kind, payload = await read_message(self.reader)
                if kind in (KIND_AUDIO, KIND_SLIN16):
                    await self._on_audio(payload, kind)
                elif kind == KIND_DTMF:
                    self._on_dtmf(payload.decode("ascii", "ignore")[:1])
                elif kind == KIND_HANGUP:
                    break
                elif kind == KIND_ERROR:
                    break
        except EOFError:
            pass
        finally:
            if self._pacer:
                self._pacer.cancel()
            if self._dtmf_task:
                self._dtmf_task.cancel()
            if self.session is not None:
                await self.session.close()
            await self._close()

    def _audio_kind(self) -> int:
        # 16 kHz PCM tagged 0x10 is played at half speed. 8 kHz stays 0x10 / 320 bytes.
        return KIND_SLIN16 if self.out_rate == 16000 else KIND_AUDIO

    async def _on_audio(self, payload: bytes, kind: int = KIND_AUDIO) -> None:
        pcm = np.frombuffer(payload[: len(payload) - len(payload) % 2], dtype="<i2")
        # 0x12 is already 16 kHz, which is what the session consumes. Running it through the
        # configured-rate resampler would treat those samples as 8 kHz when out_rate is 8000.
        if kind == KIND_SLIN16:
            samples = pcm
        else:
            samples = self._rs.push(pcm)
        self._buf = np.concatenate([self._buf, samples])
        while len(self._buf) >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            await self.session.feed(frame)  # type: ignore[union-attr]
