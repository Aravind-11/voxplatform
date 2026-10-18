"""Kokoro-82M TTS adapter (Apache-2.0 weights; needs `pip install kokoro` + apt `espeak-ng`).

KPipeline is a blocking generator, so synthesis runs in a worker thread and PCM pieces are
handed back through an asyncio queue. Output: int16 mono at 24 kHz.
"""
from __future__ import annotations

import asyncio
import io
import wave
from typing import AsyncIterator

import numpy as np


# Kokoro-82M voices by language code (Kokoro's published voice list). The first letter of a voice name is its
# language code, the second is the gender (f/m).
VOICES_BY_LANG = {
    "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_alloy", "af_aoede", "af_jessica", "af_kore",
          "af_nova", "af_river", "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx", "am_puck"],
    "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
}
VOICES = [v for vs in VOICES_BY_LANG.values() for v in vs]
LANG_ISO = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh"}
LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
             "pt": "Portuguese", "zh": "Mandarin Chinese"}
DEFAULT_VOICE = {"en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "hi": "hf_alpha", "it": "if_sara",
                 "ja": "jf_alpha", "pt": "pf_dora", "zh": "zf_xiaoxiao"}
# Languages whose grapheme-to-phoneme stage is espeak-ng (GPL-3.0). With VOX_NO_ESPEAK=1 they are unavailable.
NEEDS_ESPEAK = {"e", "f", "h", "i", "p"}
# Optional Python packages the other two languages need (misaki extras).
NEEDS_MODULE = {"j": "pyopenjtalk", "z": "jieba"}


class LanguageUnavailable(RuntimeError):
    pass


def lang_code_of(voice: str) -> str:
    return voice[:1] if voice[:1] in VOICES_BY_LANG else "a"


def language_of(voice: str) -> str:
    """ISO 639-1 code a voice speaks, e.g. 'ef_dora' -> 'es'."""
    return LANG_ISO[lang_code_of(voice)]


def why_unavailable(lang_code: str) -> str:
    """Empty string if this language can run on this server, else the reason. Cheap: no models are loaded."""
    import importlib.util
    import os
    if lang_code in NEEDS_ESPEAK and os.environ.get("VOX_NO_ESPEAK") == "1":
        return ("needs espeak-ng (GPL-3.0), which VOX_NO_ESPEAK=1 turns off; unset it only after "
                "reading docs/LICENSES.md")
    mod = NEEDS_MODULE.get(lang_code)
    if mod and importlib.util.find_spec(mod) is None:
        return f"needs the '{mod}' Python package (pip install 'misaki[{LANG_ISO[lang_code]}]')"
    return ""


def make_pipeline(lang_code: str = "a"):
    """Loads Kokoro for one language. VOX_NO_ESPEAK=1 replaces the English grapheme-to-phoneme stage with misaki's
    dictionary-only G2P (no espeak-ng fallback), so no GPL-3.0 code is used at run time; out-of-dictionary words
    (rare names, codes) are then pronounced less well. Raises LanguageUnavailable if the language cannot run here."""
    import os
    reason = why_unavailable(lang_code)
    if reason:
        raise LanguageUnavailable(f"language {LANG_ISO[lang_code]!r}: {reason}")
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code=lang_code)
    if os.environ.get("VOX_NO_ESPEAK") == "1" and lang_code in ("a", "b"):
        from misaki import en
        pipeline.g2p = en.G2P(trf=False, british=(lang_code == "b"), fallback=None)
    return pipeline


class PipelineSet:
    """One Kokoro pipeline per language, loaded the first time a voice of that language is used (the 82M weights are
    shared by Kokoro; only the text front end differs). `preload` loads languages at start-up so no caller waits."""

    def __init__(self, factory=make_pipeline, preload: str = "a"):
        import threading
        self.factory, self._p, self._lock = factory, {}, threading.Lock()
        for c in [x.strip() for x in preload.split(",") if x.strip()]:
            self.get(c)

    def get(self, lang_code: str):
        with self._lock:
            if lang_code not in self._p:
                self._p[lang_code] = self.factory(lang_code)
            return self._p[lang_code]

    def for_voice(self, voice: str):
        return self.get(lang_code_of(voice))


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, lang_code: str = "a",
                 pipeline=None):
        if pipeline is None:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code=lang_code)
        self.pipeline = pipeline
        self.voice = voice
        self.speed = speed

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        voice, speed = self.voice, self.speed

        def work() -> None:
            try:
                for _gs, _ps, audio in self.pipeline(text, voice=voice, speed=speed):
                    arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                    loop.call_soon_threadsafe(q.put_nowait, arr)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, work)
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield np.clip(np.rint(item * 32767.0), -32768, 32767).astype(np.int16)
        finally:
            await asyncio.shield(fut)


_WS = b" \t\r\n"
_ERR_OPEN = b"{[<"


def _text_media(resp) -> bool:
    """A JSON or text Content-Type is an error document, not a raw PCM body."""
    headers = getattr(resp, "headers", None)
    getter = getattr(headers, "get", None)
    if getter is None:
        return False
    ctype = str(getter("content-type") or "").split(";", 1)[0].strip().lower()
    return ctype.startswith("text/") or ctype == "application/json" or ctype.endswith("+json")


def _is_error_document(raw: bytes) -> bool:
    """JSON or HTML, including a leading BOM or whitespace. Those bytes are not s16le audio."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    i = 0
    while i < len(raw) and raw[i] in _WS:
        i += 1
    return i < len(raw) and raw[i:i + 1] in _ERR_OPEN


def _pcm_from_tts(raw, default_rate: int) -> tuple[np.ndarray, int]:
    """WAV bytes -> int16 PCM and the header rate. Anything else is raw s16le at `default_rate`."""
    if isinstance(raw, np.ndarray):
        pcm = np.asarray(raw)
        if pcm.dtype != np.int16:
            pcm = np.clip(np.rint(pcm.astype(np.float64)), -32768, 32767).astype(np.int16)
        else:
            pcm = pcm.copy()
        return pcm, default_rate
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw)
    else:
        raw = bytes(raw)
    if len(raw) >= 12 and raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as w:
            rate = w.getframerate() or default_rate
            channels = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"HttpTTS expected 16-bit audio, got {width * 8}-bit")
        pcm = np.frombuffer(frames, dtype="<i2").copy()
        if channels > 1:
            pcm = pcm.reshape(-1, channels)[:, 0].copy()
        return pcm, rate
    # An error document (JSON or HTML) is even-length often enough to decode as loud PCM.
    if _is_error_document(raw):
        raise ValueError("HttpTTS response was not audio")
    if len(raw) % 2:
        raw = raw[: len(raw) - 1]
    return np.frombuffer(raw, dtype="<i2").copy(), default_rate


class _BytePCM:
    """Join streamed body reads into int16 chunks.

    A read is not a whole sample, and a WAV header usually arrives before its data. Decoding
    each read on its own plays the header at the default rate or drops the odd trailing byte.
    """

    def __init__(self, default_rate: int) -> None:
        self.default_rate = default_rate
        self.rate = default_rate
        self._buf = bytearray()
        self._i = 0
        self._mode = "unk"  # unk until the leading bytes say wav or raw
        self._channels = 1
        self._data_left: int | None = None  # bytes still owed by the wav data chunk; -1 if the size is unknown

    def feed(self, piece) -> list[np.ndarray]:
        if isinstance(piece, np.ndarray):
            ready = self._pull(final=False)
            pcm, _rate = _pcm_from_tts(piece, self.rate)
            if len(pcm):
                ready.append(pcm)
            return ready
        self._buf += bytes(piece)
        return self._pull(final=False)

    def finish(self) -> list[np.ndarray]:
        return self._pull(final=True)

    def _pull(self, final: bool) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        while True:
            if self._mode == "unk" and not self._classify(final):
                break
            if self._mode == "raw":
                out.extend(self._take_raw(final))
                break
            if self._data_left is None and not self._take_header():
                if final and self._i < len(self._buf):
                    raise ValueError("HttpTTS response was not audio")
                break
            out.extend(self._take_wav(final))
            if self._data_left != 0:
                break
            self._mode = "unk"
            self._data_left = None
            if self._i >= len(self._buf):
                break
        self._compact()
        return out

    def _classify(self, final: bool) -> bool:
        view = bytes(self._buf[self._i:])
        if not view:
            return False
        if _is_error_document(view):
            raise ValueError("HttpTTS response was not audio")
        if _pending_error_prefix(view) and not final:
            return False
        if final and view.strip(b" \t\r\n") == b"" and view[:1] not in (b"R", b"r"):
            raise ValueError("HttpTTS response was not audio")
        if view.startswith(b"RIFF"):
            if len(view) < 12 and not final:
                return False
            if len(view) >= 12 and view[8:12] != b"WAVE":
                raise ValueError("HttpTTS response was not audio")
            self._mode = "wav"
            return True
        if not final and len(view) < 4 and view.startswith(b"R"):
            return False
        self._mode = "raw"
        return True

    def _take_header(self) -> bool:
        view = self._buf[self._i:]
        if len(view) < 12:
            return False
        pos = 12
        rate = channels = width = None
        while pos + 8 <= len(view):
            cid = bytes(view[pos:pos + 4])
            size = int.from_bytes(view[pos + 4:pos + 8], "little")
            start = pos + 8
            if cid == b"data":
                if width != 2:
                    if width is None:
                        raise ValueError("HttpTTS response was not audio")
                    raise ValueError(f"HttpTTS expected 16-bit audio, got {width * 8}-bit")
                if rate is None or not channels:
                    raise ValueError("HttpTTS response was not audio")
                self.rate = rate or self.default_rate
                self._channels = channels
                self._i += start
                self._data_left = -1 if size >= 0x7FFFFFFF else size
                return True
            if size < 0 or start + size > len(view):
                return False
            if cid == b"fmt ":
                if size < 16:
                    raise ValueError("HttpTTS response was not audio")
                channels = int.from_bytes(view[start + 2:start + 4], "little")
                rate = int.from_bytes(view[start + 4:start + 8], "little")
                width = int.from_bytes(view[start + 14:start + 16], "little") // 8
            pos = start + size + (size & 1)
        return False

    def _take_wav(self, final: bool) -> list[np.ndarray]:
        frame = self._channels * 2
        if self._data_left is not None and self._data_left < 0 and final:
            self._data_left = len(self._buf) - self._i
        avail = len(self._buf) - self._i
        if self._data_left is not None and self._data_left >= 0:
            avail = min(avail, self._data_left)
        n = avail - (avail % frame)
        if n:
            raw = bytes(self._buf[self._i:self._i + n])
            self._i += n
            if self._data_left is not None and self._data_left >= 0:
                self._data_left -= n
            pcm = np.frombuffer(raw, dtype="<i2").copy()
            if self._channels > 1:
                pcm = pcm.reshape(-1, self._channels)[:, 0].copy()
        else:
            pcm = np.zeros(0, dtype=np.int16)
        if final and self._data_left is not None and self._data_left > 0:
            drop = min(len(self._buf) - self._i, self._data_left)
            self._i += drop
            self._data_left -= drop
        return [pcm] if len(pcm) else []

    def _take_raw(self, final: bool) -> list[np.ndarray]:
        avail = len(self._buf) - self._i
        n = avail - (avail % 2)
        if n <= 0:
            if final:
                self._i = len(self._buf)
            return []
        raw = bytes(self._buf[self._i:self._i + n])
        self._i += n
        if final:
            self._i = len(self._buf)
        return [np.frombuffer(raw, dtype="<i2").copy()]

    def _compact(self) -> None:
        if self._i and (self._i == len(self._buf) or self._i > 4096):
            del self._buf[:self._i]
            self._i = 0


def _pending_error_prefix(view: bytes) -> bool:
    """True while the leading bytes could still become a JSON/HTML error and not audio."""
    if view.startswith(b"\xef") and not view.startswith(b"\xef\xbb\xbf"):
        return len(view) < 3
    if view.startswith(b"\xef\xbb\xbf"):
        view = view[3:]
    i = 0
    while i < len(view) and view[i] in _WS:
        i += 1
    return i == len(view)


async def _response_bytes(resp) -> bytes:
    if isinstance(resp, (bytes, bytearray, memoryview)):
        return bytes(resp)
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray, memoryview)):
        return bytes(content)
    if hasattr(resp, "aread"):
        data = resp.aread()
        if asyncio.iscoroutine(data):
            data = await data
        return bytes(data)
    if hasattr(resp, "aiter_bytes"):
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf.extend(chunk)
        return bytes(buf)
    raise TypeError("HttpTTS response has no audio body")


class HttpTTS:
    """OpenAI-compatible `POST {base_url}/audio/speech`.

    `sample_rate` starts at 24000 and is replaced by the rate in a WAV response. If `client.post`
    returns an async iterator, each yielded piece is one chunk; one WAV body is one chunk.
    """

    def __init__(self, base_url: str, client, *, model: str = "tts-1", voice: str = "alloy",
                 api_key: str = "", sample_rate: int = 24000) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.sample_rate = sample_rate

    async def _iter_pieces(self, pieces) -> AsyncIterator[np.ndarray]:
        dec = _BytePCM(self.sample_rate)
        async for piece in pieces:
            for pcm in dec.feed(piece):
                self.sample_rate = dec.rate
                if len(pcm):
                    yield pcm
            self.sample_rate = dec.rate
        for pcm in dec.finish():
            self.sample_rate = dec.rate
            if len(pcm):
                yield pcm

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        payload = {"model": self.model, "voice": self.voice, "input": text}
        kwargs: dict = {"json": payload}
        if self.api_key:
            kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
        posted = self.client.post(f"{self.base_url}/audio/speech", **kwargs)
        if hasattr(posted, "__aiter__"):
            async for pcm in self._iter_pieces(posted):
                yield pcm
            return
        resp = await posted
        status = getattr(resp, "status_code", None)
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"HttpTTS HTTP {status}")
        if _text_media(resp):
            raise ValueError("HttpTTS response was not audio")
        if hasattr(resp, "__aiter__") and not isinstance(resp, (bytes, bytearray)):
            async for pcm in self._iter_pieces(resp):
                yield pcm
            return
        pcm, rate = _pcm_from_tts(await _response_bytes(resp), self.sample_rate)
        self.sample_rate = rate
        if len(pcm):
            yield pcm
