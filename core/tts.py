"""Kokoro-82M TTS adapter (Apache-2.0 weights; needs `pip install kokoro` + apt `espeak-ng`).

KPipeline is a blocking generator, so synthesis runs in a worker thread and PCM pieces are
handed back through an asyncio queue. Output: int16 mono at 24 kHz.

`HttpTTS` is the swap for a commercially licensed engine (`VOX_TTS_URL`). Kokoro's voice list
does not grow when that is set.
"""
from __future__ import annotations

import asyncio
import io
import json
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


class HttpTTS:
    """OpenAI-compatible `POST {url}/audio/speech`.

    The body is `{model, voice, input, response_format: "pcm"}`. OpenAI defines `pcm` as raw
    24 kHz little-endian s16le, which is what a headerless body is decoded as. A WAV response
    sets `sample_rate` from the header. JSON, text, and compressed audio (mpeg, opus, ogg,
    flac, aac, mp4, or an ID3 tag) are refused. Anything else is raw s16le at `rate`
    (`VOX_TTS_RATE`, default 24000). One response is one chunk.
    """

    def __init__(self, voice: str, url: str, client, model: str = "tts-1", api_key: str = "",
                 rate: int = 24000):
        self.voice = voice
        self.url = url.rstrip("/")
        self.client = client
        self.model = model or "tts-1"
        self.api_key = api_key
        self._default_rate = rate
        self.sample_rate = rate

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        # The shared AsyncClient defaults to 5s. Override it for this POST: a sentence can take
        # longer than that to start, but a dead peer must still hit a finite read deadline.
        response = await self.client.post(
            f"{self.url}/audio/speech",
            json={"model": self.model, "voice": self.voice, "input": text, "response_format": "pcm"},
            headers=headers,
            timeout=30)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data = response.content
        pcm, rate = _pcm_from_speech(data, self._default_rate, _content_type(response))
        self.sample_rate = rate
        if len(pcm):
            yield pcm


def make_tts_factory(env, http):
    """Kokoro unless `VOX_TTS_URL` is set, in which case every voice uses that HTTP engine.

    The HTTP branch does not construct a Kokoro pipeline. The fixed voice list is unchanged either way.
    """
    url = env.get("VOX_TTS_URL", "").strip().rstrip("/")
    if url:
        model = env.get("VOX_TTS_MODEL", "tts-1").strip() or "tts-1"
        key = env.get("VOX_TTS_API_KEY", "")
        rate = int(env.get("VOX_TTS_RATE", "24000") or "24000")
        return lambda voice, url=url, model=model, key=key, rate=rate: HttpTTS(
            voice, url, http, model=model, api_key=key, rate=rate)
    pipelines = PipelineSet(preload=env.get("VOX_PRELOAD_LANGS", "a"))   # one front end per language, shared by all calls
    return lambda voice: KokoroTTS(voice=voice, pipeline=pipelines.for_voice(voice))


def _content_type(response) -> str:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return ""
    value = headers.get("content-type")
    if not value and isinstance(headers, dict):
        for key, item in headers.items():
            if str(key).lower() == "content-type":
                value = item
                break
    return str(value or "")


# OpenAI speech defaults to mp3. These are not s16le; a bare 0xFFFB frame sync is left alone
# because those two bytes are common in PCM.
_COMPRESSED_MEDIA = frozenset({
    "audio/mpeg", "audio/mp3", "audio/opus", "audio/ogg", "audio/flac", "audio/aac", "audio/mp4",
})


def _json_document(data: bytes) -> bool:
    """True when the whole body is a JSON object or array. A lone '{' byte is a PCM sample, not a document."""
    try:
        text = data.decode("utf-8-sig").strip()  # a leading UTF-8 BOM is not whitespace
    except UnicodeDecodeError:
        return False
    if len(text) < 2 or text[0] not in "{[":
        return False
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (dict, list))


def _pcm_from_speech(data: bytes, default_rate: int, content_type: str = "") -> tuple[np.ndarray, int]:
    media = content_type.split(";")[0].strip().lower()
    if media.startswith("text/") or "json" in media:
        raise ValueError(f"TTS response is {media}, not audio")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"TTS WAV must be 16-bit, got {width * 8}-bit")
        pcm = np.frombuffer(raw, dtype=np.int16).copy()
        if channels > 1:
            pcm = pcm.reshape(-1, channels)[:, 0].copy()
        return pcm, rate
    if _json_document(data):
        raise ValueError("TTS response is JSON, not audio")
    if media in _COMPRESSED_MEDIA:
        raise ValueError(f"TTS response is {media}, not PCM")
    if data.startswith(b"ID3"):
        raise ValueError("TTS response is an ID3 audio tag, not PCM")
    even = data[: len(data) - (len(data) % 2)]
    return np.frombuffer(even, dtype=np.int16).copy(), default_rate
