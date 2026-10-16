"""Does a phone line hurt what Ultravox understands?  Same sentences, four channels, word error rate.

Needs the vLLM server (port 8000) and Kokoro on this box.
  python tests/bringup/phone_wer.py
Channels: clean 16 kHz | phone (8 kHz mu-law, no noise) | phone + 20 dB noise | phone + 10 dB noise.
The transcript is a proxy for what the model hears: the agent answers from the audio, not from this text.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from core.audio import resample  # noqa: E402
from core.llm import VllmClient  # noqa: E402

SENTENCES = [
    "What time do you open on Monday?", "I'd like to book a cleaning for Tuesday at eleven.",
    "How much does a teeth whitening cost?", "Can I move my appointment to next week?",
    "My name is Sarah Johnson and my phone number is five five five, one two three four.",
    "Do you take Delta Dental insurance?", "I need to cancel my appointment tomorrow.",
    "I have a bad toothache and need to be seen today.", "Wednesday afternoon would be perfect.",
    "Can you send me a reminder by text?", "What do I need to bring with me?",
    "I'd like to speak to someone about a bill.", "Is there parking near the clinic?",
    "Please book me for the first available slot on Thursday morning.",
    "Where exactly are you located?", "How long does the appointment take?",
    "I think I'd rather come in on Friday at two thirty.", "Do you see children?",
    "My insurance changed last month, so I need to update it.", "Great, see you then.",
]


def norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


async def main() -> None:
    from kokoro import KPipeline
    from eot_eval import degrade          # reuse the exact phone-line model from the other eval
    pipe = KPipeline(lang_code="a")
    rng = np.random.default_rng(11)
    llm = VllmClient()
    voices = ["af_bella", "am_michael"]
    clips = []
    for v in voices:
        for t in SENTENCES:
            parts = [np.asarray(x) for _, _, x in pipe(t, voice=v)]
            pcm = resample((np.concatenate(parts) * 32767).astype(np.int16), 24000, 16000)
            clips.append((t, np.concatenate([np.zeros(3200, np.int16), pcm, np.zeros(4000, np.int16)])))
    chans = [("clean 16 kHz", lambda x: x), ("phone (8k mu-law)", lambda x: degrade(x, None, rng)),
             ("phone + 20 dB noise", lambda x: degrade(x, 20, rng)),
             ("phone + 10 dB noise", lambda x: degrade(x, 10, rng))]
    await llm.transcribe(clips[0][1])     # warm-up
    print(f"\n{len(clips)} utterances ({len(voices)} synthetic voices x {len(SENTENCES)} sentences)")
    print(f"{'channel':<22}{'WER':>8}{'sentences with any error':>28}")
    worst = []
    for name, f in chans:
        err = tot = bad = 0
        for t, pcm in clips:
            hyp = await llm.transcribe(f(pcm))
            e, n = wer(norm(t), norm(hyp))
            err += e; tot += n; bad += e > 0
            if e and name != "clean 16 kHz":
                worst.append((e, name, t, hyp))
        print(f"{name:<22}{err / tot:>8.1%}{bad / len(clips):>28.0%}")
    print("\nExamples of mishearing (most errors first):")
    for e, name, t, hyp in sorted(worst, reverse=True)[:8]:
        print(f"  [{name}] said: {t}\n      heard: {hyp}")


if __name__ == "__main__":
    asyncio.run(main())
