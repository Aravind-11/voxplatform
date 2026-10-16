#!/usr/bin/env python3
"""Bring-up test for Ultravox served by vLLM (run on the GPU box once the server is healthy).

Checks, in order:
  1. server reachable, model listed
  2. one audio request works (tries `input_audio` wav, falls back to `audio_url` data URI)
  3. streaming time-to-first-token (TTFT) over several runs
  4. concurrency sweep: TTFT p50/p95 and aggregate tokens/s at N simultaneous requests

Writes a JSON report to --out. Nothing here claims real-call latency: it measures the
model server only (audio in -> first text token), not VAD, TTS or telephony.
"""
import argparse
import asyncio
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI, BadRequestError

ASSET_BASE = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset"
ASSETS = {
    "mary_had_lamb": "Transcribe this into English.",
    "winning_call": "What is happening in this audio clip?",
}


def fetch_wav(name: str, cache: Path) -> bytes:
    """Download a public .ogg test clip once, return it as 16-bit WAV bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    ogg = cache / f"{name}.ogg"
    if not ogg.exists():
        urllib.request.urlretrieve(f"{ASSET_BASE}/{name}.ogg", ogg)
    data, sr = sf.read(ogg, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_part(wav: bytes, style: str) -> dict:
    b64 = base64.b64encode(wav).decode()
    if style == "input_audio":
        return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}


def messages(prompt: str, wav: bytes, style: str) -> list:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, audio_part(wav, style)]}]


async def wait_ready(client: AsyncOpenAI, timeout_s: int) -> str:
    t0 = time.time()
    while True:
        try:
            models = await client.models.list()
            return models.data[0].id
        except Exception as e:  # noqa: BLE001 - server still starting
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"server not ready after {timeout_s}s: {e}")
            await asyncio.sleep(3)


async def stream_once(client, model, msgs, max_tokens=96) -> dict:
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    text = []
    stream = await client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=0,
        stream=True, stream_options={"include_usage": True},
    )
    out_tokens = None
    async for chunk in stream:
        if chunk.usage is not None:
            out_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            n_chunks += 1
            text.append(chunk.choices[0].delta.content)
    total = (time.perf_counter() - t0) * 1000
    return {"ttft_ms": ttft, "total_ms": total,
            "tokens": out_tokens if out_tokens is not None else n_chunks,
            "text": "".join(text)}


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--reps", type=int, default=3, help="rounds per concurrency level")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for server")
    ap.add_argument("--cache", default="/lambda/nfs/arvi/assets")
    ap.add_argument("--out", default="/lambda/nfs/arvi/results")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="none", timeout=300)
    model = args.model or await wait_ready(client, args.wait)
    print(f"[1] server ready, model id: {model}")

    clips = {k: fetch_wav(k, Path(args.cache)) for k in ASSETS}
    report = {"model": model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "clips": {}}

    # [2] find a request style that the server accepts
    style = None
    for candidate in ("input_audio", "audio_url"):
        try:
            r = await client.chat.completions.create(
                model=model, max_tokens=64, temperature=0,
                messages=messages(ASSETS["mary_had_lamb"], clips["mary_had_lamb"], candidate))
            print(f"[2] audio request OK via '{candidate}':\n    {r.choices[0].message.content!r}")
            style = candidate
            break
        except BadRequestError as e:
            print(f"[2] '{candidate}' rejected: {e}")
    if style is None:
        raise SystemExit("no audio request style worked; see vLLM server log")
    report["audio_request_style"] = style

    for name, prompt in ASSETS.items():
        r = await client.chat.completions.create(
            model=model, max_tokens=96, temperature=0, messages=messages(prompt, clips[name], style))
        report["clips"][name] = r.choices[0].message.content
        print(f"    [{name}] -> {r.choices[0].message.content!r}")

    # [3] sequential streaming TTFT
    msgs = messages(ASSETS["mary_had_lamb"], clips["mary_had_lamb"], style)
    await stream_once(client, model, msgs)  # warm-up (CUDA graphs, caches)
    seq = [await stream_once(client, model, msgs) for _ in range(8)]
    ttfts = [s["ttft_ms"] for s in seq if s["ttft_ms"] is not None]
    report["sequential"] = {"ttft_ms_p50": pct(ttfts, 50), "ttft_ms_p95": pct(ttfts, 95),
                            "runs": len(ttfts)}
    print(f"[3] sequential TTFT ms  p50={pct(ttfts,50):.0f}  p95={pct(ttfts,95):.0f}  (n={len(ttfts)})")

    # [4] concurrency sweep
    report["concurrency"] = []
    print("[4] concurrency sweep (N simultaneous streaming requests)")
    print("    N   TTFT p50   TTFT p95   agg tok/s   errors")
    for n in [int(x) for x in args.levels.split(",")]:
        ttft_all, tokens, errors = [], 0, 0
        t0 = time.perf_counter()
        for _ in range(args.reps):
            results = await asyncio.gather(
                *[stream_once(client, model, msgs) for _ in range(n)], return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) or r["ttft_ms"] is None:
                    errors += 1
                else:
                    ttft_all.append(r["ttft_ms"])
                    tokens += r["tokens"]
        wall = time.perf_counter() - t0
        row = {"n": n, "ttft_ms_p50": pct(ttft_all, 50), "ttft_ms_p95": pct(ttft_all, 95),
               "agg_tokens_per_s": tokens / wall, "errors": errors}
        report["concurrency"].append(row)
        print(f"  {n:3d}   {row['ttft_ms_p50']:7.0f}    {row['ttft_ms_p95']:7.0f}    "
              f"{row['agg_tokens_per_s']:8.0f}    {errors}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bringup-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
