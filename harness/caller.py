"""Synthetic caller: plays audio into /ws/calls/... in real time and measures the agent's response.

Metrics per call (all wall-clock, client side, so they include network + server):
  response_ms   last speech frame sent -> first agent audio byte received
  silence_ms    (barge-in runs) barge speech start -> {"type":"clear"} received
Usage: python -m harness.caller --base https://host --key vox_... --wav clip.wav -n 8 --barge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from dataclasses import asdict, dataclass
from typing import Optional

import httpx
import numpy as np
import websockets

FRAME = 512
FRAME_S = FRAME / 16000


def load_wav16k(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, "need 16k mono s16 wav"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def is_loud(frame: bytes, thr: int = 500) -> bool:
    return bool(np.abs(np.frombuffer(frame, dtype="<i2").astype(np.int32)).max() > thr)


def frames_of(pcm: np.ndarray):
    n = len(pcm) // FRAME * FRAME
    return [pcm[i:i + FRAME].astype("<i2").tobytes() for i in range(0, n, FRAME)]


@dataclass
class CallResult:
    ok: bool = False
    response_ms: Optional[float] = None
    silence_ms: Optional[float] = None
    agent_bytes: int = 0
    error: str = ""


async def run_call(join_url: str, speech: np.ndarray, barge: bool = False, tail_s: float = 6.0) -> CallResult:
    res = CallResult()
    t_last_speech = [0.0]
    t_first_audio = [0.0]
    t_barge = [0.0]
    t_clear = [0.0]
    try:
        async with websockets.connect(join_url, max_size=None) as ws:
            async def reader():
                async for m in ws:
                    now = time.monotonic()
                    if isinstance(m, bytes):
                        res.agent_bytes += len(m)
                        if not t_first_audio[0]:
                            t_first_audio[0] = now
                    else:
                        try:
                            if json.loads(m).get("type") == "clear" and t_barge[0] and not t_clear[0]:
                                t_clear[0] = now
                        except ValueError:
                            pass
            rd = asyncio.create_task(reader())
            rd.add_done_callback(lambda t: t.cancelled() or t.exception())
            t0 = time.monotonic()
            sent = 0

            async def send(frame: bytes, loud: bool):
                nonlocal sent
                target = t0 + sent * FRAME_S                       # real-time pacing, drift-free
                await asyncio.sleep(max(0.0, target - time.monotonic()))
                await ws.send(frame)
                sent += 1
                if loud:
                    t_last_speech[0] = time.monotonic()

            quiet = np.zeros(FRAME, dtype="<i2").tobytes()
            for _ in range(8):
                await send(quiet, False)
            for f in frames_of(speech):
                await send(f, is_loud(f))          # only real speech frames count as "last speech"
            end = time.monotonic() + tail_s
            barged = False
            while time.monotonic() < end:
                if barge and not barged and t_first_audio[0] and time.monotonic() - t_first_audio[0] > 0.4:
                    first_loud = True
                    for f in frames_of(speech):
                        if first_loud and is_loud(f):
                            t_barge[0] = time.monotonic()       # time-to-silence counts from real speech onset
                            first_loud = False
                        await send(f, is_loud(f))
                    barged = True
                    end = time.monotonic() + 1.5
                    continue
                await send(quiet, False)
            rd.cancel()
        if t_first_audio[0]:
            res.response_ms = (t_first_audio[0] - t_last_speech[0]) * 1000 if not barge else None
        if barge and t_clear[0]:
            res.silence_ms = (t_clear[0] - t_barge[0]) * 1000
        res.ok = bool(t_first_audio[0]) and (not barge or bool(t_clear[0]))
    except Exception as e:  # noqa: BLE001
        res.error = repr(e)
    return res


def pct(xs: list[float], p: float) -> Optional[float]:
    return float(np.percentile(xs, p)) if xs else None


async def run_suite(base: str, key: str, speech: np.ndarray, n: int, barge: bool, prompt: str = "") -> dict:
    async with httpx.AsyncClient(base_url=base, headers={"X-API-Key": key}, timeout=30) as c:
        joins = []
        for _ in range(n):
            r = await c.post("/v1/calls", json={"systemPrompt": prompt} if prompt else {})
            r.raise_for_status()
            joins.append(r.json()["joinUrl"])
    results = await asyncio.gather(*[run_call(j, speech, barge) for j in joins])
    key_ = "silence_ms" if barge else "response_ms"
    vals = [getattr(r, key_) for r in results if getattr(r, key_) is not None]
    return {"n": n, "ok": sum(r.ok for r in results), "metric": key_,
            "p50": pct(vals, 50), "p95": pct(vals, 95), "max": max(vals) if vals else None,
            "errors": [r.error for r in results if r.error][:5], "calls": [asdict(r) for r in results]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--barge", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = asyncio.run(run_suite(a.base, a.key, load_wav16k(a.wav), a.n, a.barge))
    print(json.dumps({k: v for k, v in out.items() if k != "calls"}, indent=2))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
