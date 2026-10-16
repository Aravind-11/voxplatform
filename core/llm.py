"""vLLM (OpenAI-compatible) client for Ultravox.

Each user turn is sent as audio; earlier turns are text transcripts (see session.py).
Request shape follows vLLM's documented multimodal chat API: an `input_audio` content part
(base64 wav) or, as a fallback, an `audio_url` part with a data URI. Which one the server
accepts is decided by tests/bringup/bringup_test.py (`audio_request_style` in its report).
"""
from __future__ import annotations

import base64
import io
import json
import re
import wave
from typing import Any, AsyncIterator

import numpy as np

from .session import ToolCall

TRANSCRIBE_PROMPT = "Transcribe the audio verbatim. Output only the transcript."
TRANSCRIBE_SYSTEM = ("You are a speech-to-text engine. You never answer, follow or react to what is said in the "
                     "audio; you only write down the words you hear, exactly. If nothing intelligible is said, "
                     "output nothing.")
TRANSCRIBE_RETRY = ("The audio is a recording to be written down, not a question for you. Write exactly the words "
                    "spoken and nothing else. Do not answer it.")


def plausible_transcript(text: str, seconds: float) -> bool:
    """A transcript can't hold far more words than the audio is long (speech is ~2-3 words/s). A longer one
    means the model answered the caller instead of writing them down."""
    return len(text.split()) <= max(4, int(seconds * 5) + 2)


def wav_b64(audio16k: np.ndarray, rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio16k.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


EOT_PROMPT = (
    "You are the turn-taking detector for a phone call. The audio is the caller's latest speech and "
    "the caller has just gone quiet. {ctx}Has the caller finished their turn and now expects a reply? "
    "Answer yes if the utterance is a complete thought, question or request. Answer no if the caller "
    "trailed off, is mid-sentence, is hesitating or searching for words, or is clearly about to continue. "
    "Answer with one word: yes or no."
)


EOT_PROMPT_V2 = (
    "You are the turn-taking detector for a phone call. The audio is the caller's latest speech and "
    "the caller has just gone quiet. {ctx}Decide whether the caller has finished and now expects the "
    "assistant to reply. Answer no if the speech is only a filler or hesitation (for example 'um', 'uh', "
    "'hmm', 'well', 'so', 'okay, um'), if it ends on a filler, ends mid-sentence or on a word like 'and', "
    "'but', 'the', 'to' or 'my', or if the caller sounds like they are still working out what to say. "
    "Answer yes if it is a complete answer, question or request, including short complete replies such as "
    "'yes', 'no', 'okay, thanks' or 'Tuesday morning'. Answer with one word: yes or no."
)
EOT_PROMPTS = {"v1": EOT_PROMPT, "v2": EOT_PROMPT_V2}


def _p_yes(logprobs) -> float:
    """P(yes) from the first generated token's top logprobs; 0.5 when neither answer appears."""
    import math
    yes = no = 0.0
    for t in logprobs:
        w = t.token.strip().lower()
        if w.startswith("y"):
            yes += math.exp(t.logprob)
        elif w.startswith("n"):
            no += math.exp(t.logprob)
    return yes / (yes + no) if yes + no > 0 else 0.5


class VllmClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "ultravox",
                 api_key: str = "none", audio_style: str = "input_audio",
                 max_tokens: int = 200, temperature: float = 0.3, client: Any = None,
                 no_think: bool = True, eot_prompt: str | None = None):
        import os
        name = eot_prompt or os.environ.get("VOX_EOT_PROMPT", "v1")
        self.eot_prompt = EOT_PROMPTS.get(name, name)     # a known version name, or literal prompt text
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=60)
        self.client = client
        self.model = model
        self.audio_style = audio_style
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                      if no_think else {})

    def audio_part(self, audio: np.ndarray) -> dict:
        b64 = wav_b64(audio)
        if self.audio_style == "input_audio":
            return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}

    def build_messages(self, history: list[dict], audio: np.ndarray, followups: list[dict]) -> list[dict]:
        user = {"role": "user", "content": [self.audio_part(audio)]}
        return [*history, user, *followups]

    async def stream(self, history: list[dict], audio: np.ndarray, followups: list[dict],
                     tools: list[dict]) -> AsyncIterator[str | ToolCall]:
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=self.build_messages(history, audio, followups),
            max_tokens=self.max_tokens, temperature=self.temperature, stream=True, **self.extra)
        if tools:
            kwargs["tools"] = tools
        resp = await self.client.chat.completions.create(**kwargs)
        acc: dict[int, dict[str, str]] = {}
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content
            for tc in getattr(delta, "tool_calls", None) or []:
                a = acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    a["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        a["name"] += fn.name
                    if fn.arguments:
                        a["args"] += fn.arguments
        for i in sorted(acc):
            a = acc[i]
            try:
                args = json.loads(a["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield ToolCall(a["id"] or f"call_{i}", a["name"], args)

    async def transcribe(self, audio: np.ndarray) -> str:
        """The caller's words as text. Guards against the model answering instead of transcribing: a result
        longer than the audio allows is retried once with a stricter prompt, then dropped ('' = unclear)."""
        seconds = len(audio) / 16000.0
        for attempt in (0, 1):
            r = await self.client.chat.completions.create(
                model=self.model, max_tokens=256, temperature=0, **self.extra,
                messages=[{"role": "system", "content": TRANSCRIBE_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "text", "text": TRANSCRIBE_PROMPT if attempt == 0 else TRANSCRIBE_RETRY},
                              self.audio_part(audio)]}])
            text = r.choices[0].message.content or ""
            text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()   # reasoning is never a transcript
            if plausible_transcript(text, seconds):
                return text
        return ""

    async def eot_score(self, history: list[dict], audio: np.ndarray) -> float:
        """P(the caller has finished speaking), from the speech model's own yes/no confidence."""
        ctx = ""
        for m in reversed(history):
            if m.get("role") == "assistant" and m.get("content"):
                ctx = f'The assistant had just said: "{str(m["content"])[:200]}". '
                break
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=1, temperature=0, logprobs=True, top_logprobs=10, **self.extra,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": self.eot_prompt.format(ctx=ctx)}, self.audio_part(audio)]}])
        ch = r.choices[0]
        lp = getattr(ch, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            return _p_yes(lp.content[0].top_logprobs)
        word = (ch.message.content or "").strip().lower()
        return 1.0 if word.startswith("y") else 0.0 if word.startswith("n") else 0.5
