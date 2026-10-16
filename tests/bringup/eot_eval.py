"""Does the model's yes/no 'is the caller done?' judgement separate finished from unfinished speech?

Needs the vLLM server (port 8000) and Kokoro on this box. Synthesizes callers with Kokoro, optionally
degrades them like a phone line (8 kHz, mu-law, noise), asks VllmClient.eot_score for each, and reports
score separation, AUC, latency, and the threshold to use.
  python tests/bringup/eot_eval.py --phone --snr 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.audio import mulaw_to_pcm16, pcm16_to_mulaw, resample  # noqa: E402
from core.llm import VllmClient  # noqa: E402
from core.prosody import eot_prosody  # noqa: E402

COMPLETE = [
    "What time do you open on Monday?", "I'd like to book a cleaning for Tuesday at eleven.",
    "How much does a teeth whitening cost?", "Yes, that works for me.", "No, thank you.",
    "Can I move my appointment to next week?", "My name is Sarah Johnson.", "Do you take Delta Dental insurance?",
    "I need to cancel my appointment tomorrow.", "Are you open on Sundays?", "That sounds good, thanks.",
    "Where exactly are you located?", "I have a bad toothache and need to be seen today.",
    "Can you send me a reminder by text?", "Great, see you then.", "What do I need to bring with me?",
    "My phone number is five five five, one two three four.", "How long does the appointment take?",
    "I'd like to speak to someone about a bill.", "Thank you, that's all I needed.",
    "Is there parking nearby?", "Wednesday afternoon would be perfect.", "I think I'll pass on that.",
    "Do you see children?", "Please book me for the first available slot.",
    # short complete replies: these must still end fast
    "Yes.", "No.", "Okay, thanks.", "Tuesday morning.", "That's right.",
]
INCOMPLETE = [
    "I was calling because I wanted to, um,", "So I'd like to book an appointment for", "What time do you",
    "My name is", "Well, I was thinking that maybe,", "Can you tell me how much the", "I need to cancel my",
    "Yeah so the thing is, I've been having this pain in my", "Do you take insurance, because my", "Uh, let me think,",
    "I'd like to move my appointment from Tuesday to", "And then also I wanted to ask about the,",
    "So my phone number is five five five,", "Is it possible that I could", "I guess what I'm trying to say is",
    "Hmm, so on Wednesday I'm free in the morning but then", "Okay and the address is one two three",
    "I was wondering if you could maybe, uh,", "The reason I'm calling is that last week I", "Can I ask you something about the",
    "It started a few days ago and", "So if I come in on Monday then", "Right, and what about the cost for the",
    "I'm looking for an appointment either Thursday or", "Um, one second, let me find my",
    # filler-only fragments: the real call showed the verdict scoring these 0.98 ("finished")
    "Um,", "Uh,", "Okay, um,", "Hmm,", "Well,", "So,",
]


def degrade(pcm16k: np.ndarray, snr_db: float | None, rng) -> np.ndarray:
    """Phone line: 16k -> 8k, mu-law round trip, line noise at snr_db, back to 16k."""
    x = resample(pcm16k, 16000, 8000)
    x = mulaw_to_pcm16(pcm16_to_mulaw(x))
    if snr_db is not None:
        p = float(np.mean(x.astype(np.float64) ** 2)) + 1e-9
        noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))
        x = np.clip(x + noise, -32768, 32767).astype(np.int16)
    return resample(x, 8000, 16000)


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def report(name: str, sc: list[float], labels: list[int], clips) -> None:
    pos = [s for s, y in zip(sc, labels) if y]
    neg = [s for s, y in zip(sc, labels) if not y]
    print(f"\n=== {name}: AUC {auc(pos, neg):.3f}  (0.5 = coin flip, 1.0 = perfect)")
    print(f"    finished   median {np.median(pos):.2f} | unfinished median {np.median(neg):.2f}")
    print("    threshold  fast-end coverage   false cut-off rate")
    best = None
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        cov = np.mean([s >= thr for s in pos]); fp = np.mean([s >= thr for s in neg])
        print(f"      {thr:.2f}        {cov:6.1%}            {fp:6.1%}")
        if fp <= 0.05 and (best is None or cov > best[1]):
            best = (thr, cov, fp)
    print("    'not done' (<=0.25) catches", f"{np.mean([s <= 0.25 for s in neg]):.1%}", "of unfinished utterances")
    print("    RECOMMENDED threshold:", f"{best[0]} (coverage {best[1]:.0%}, false cut-off {best[2]:.0%})" if best
          else "none reaches <=5% false cut-offs -> do not enable")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--voice", default="af_bella")
    ap.add_argument("--prompt", default="v1", help="end-of-turn prompt version: v1 (original) or v2 (filler-aware)")
    ap.add_argument("--out", default="/lambda/nfs/arvi/results/eot_eval.json")
    a = ap.parse_args()
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    rng = np.random.default_rng(7)

    def synth(text: str, trail_silence_s: float = 0.25) -> np.ndarray:
        parts = [np.asarray(x) for _, _, x in pipe(text, voice=a.voice)]
        pcm = (np.concatenate(parts) * 32767).astype(np.int16)
        pcm = resample(pcm, 24000, 16000)
        pcm = np.concatenate([np.zeros(3200, np.int16), pcm, np.zeros(int(16000 * trail_silence_s), np.int16)])
        return degrade(pcm, a.snr, rng) if a.phone else pcm

    llm = VllmClient(eot_prompt=a.prompt)
    items = [(t, 1) for t in COMPLETE] + [(t, 0) for t in INCOMPLETE]
    clips = [(t, y, synth(t)) for t, y in items]
    await llm.eot_score([], clips[0][2])                       # warm-up
    scores, lat = [], []
    for t, y, pcm in clips:
        t0 = time.perf_counter()
        s = await llm.eot_score([], pcm)
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(s)
    labels = [y for _, y, _ in clips]
    pros = [eot_prosody(pcm) for _, _, pcm in clips]
    comb = [(m + p_) / 2 for m, p_ in zip(scores, pros)]
    print(f"\nchannel: {'phone-degraded' if a.phone else 'clean'} snr={a.snr}  prompt={a.prompt}")
    print(f"model verdict latency per call: p50 {np.median(lat):.0f} ms p95 {np.percentile(lat, 95):.0f} ms")
    for name, sc in (("MODEL (Ultravox yes/no)", scores), ("PROSODY (pitch/energy/hum, no model)", pros),
                     ("AVERAGE of both", comb)):
        report(name, sc, labels, clips)
    scores_for_json = {"model": scores, "prosody": pros, "average": comb}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"phone": a.phone, "snr": a.snr, "latency_ms": lat,
               "items": [{"text": t, "complete": bool(y), **{k: v[i] for k, v in scores_for_json.items()}}
                         for i, (t, y, _) in enumerate(clips)]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
