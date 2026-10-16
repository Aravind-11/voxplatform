"""Does Ultravox actually call tools correctly when it HEARS the request?  (needs vLLM started with TOOLS=1)

Synthesizes callers with Kokoro, sends the audio with tool schemas, and checks which tool the model calls
(or that it correctly calls none). Runs on a clean channel and on a phone-degraded one. Then does one full
round trip: tool result fed back, model answers in words.
  python tests/bringup/tools_check.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from api import tools as toolmod  # noqa: E402
from core.audio import resample  # noqa: E402
from core.llm import VllmClient  # noqa: E402
from core.session import ToolCall  # noqa: E402

SYSTEM = ("You are a friendly receptionist for a dental clinic. Keep answers to one or two short sentences. "
          "Use the tools to check or book appointments. Use hangUp only when the caller says goodbye.")
# (what the caller says, expected tool or None, text the arguments must contain)
CASES = [
    ("Do you have anything free on Tuesday?", "check_availability", "tue"),
    ("What slots are open on Monday?", "check_availability", "mon"),
    ("Please book me for Monday at two in the afternoon.", "book_slot", "mon"),
    ("Is Wednesday open at all?", "check_availability", "wed"),
    ("Can you book Tuesday at eleven for me?", "book_slot", "tue"),
    ("How are you doing today?", None, ""),
    ("What is the name of the clinic?", None, ""),
    ("Do you take children as patients?", None, ""),
    ("Thanks, that's all I needed. Goodbye!", "hangUp", ""),
    ("No that's everything, bye bye.", "hangUp", ""),
]


async def run_one(llm, schemas, audio):
    hist = [{"role": "system", "content": SYSTEM}]
    text, calls = "", []
    async for it in llm.stream(hist, audio, [], schemas):
        if isinstance(it, ToolCall):
            calls.append(it)
        else:
            text += it
    return text.strip(), calls


async def main() -> None:
    from kokoro import KPipeline
    from eot_eval import degrade
    pipe = KPipeline(lang_code="a")
    rng = np.random.default_rng(5)
    schemas, _ = toolmod.build(["check_availability", "book_slot", "hangUp"])
    llm = VllmClient()

    def synth(t, phone):
        parts = [np.asarray(x) for _, _, x in pipe(t, voice="af_bella")]
        pcm = resample((np.concatenate(parts) * 32767).astype(np.int16), 24000, 16000)
        pcm = np.concatenate([np.zeros(3200, np.int16), pcm, np.zeros(4000, np.int16)])
        return degrade(pcm, 20, rng) if phone else pcm

    for phone in (False, True):
        ok = 0
        print(f"\n=== channel: {'phone-degraded (8k mu-law, 20 dB)' if phone else 'clean'}")
        for text, want, must in CASES:
            text_out, calls = await run_one(llm, schemas, synth(text, phone))
            got = calls[0].name if calls else None
            blob = json.dumps(calls[0].arguments).lower() if calls else ""
            good = got == want and must in blob
            ok += good
            print(f"  {'PASS' if good else 'FAIL'}  said: {text!r}\n        expected {want} | got {got} {blob if calls else repr(text_out[:60])}")
        print(f"  -> {ok}/{len(CASES)} correct")

    print("\n=== full round trip (clean): ask, tool result back, model answers in words")
    audio = synth("Do you have anything free on Tuesday?", False)
    _, calls = await run_one(llm, schemas, audio)
    if not calls:
        print("  model did not call a tool; cannot run the round trip")
        return
    c = calls[0]
    _, fns = toolmod.build(["check_availability", "book_slot", "hangUp"])
    result = json.dumps(await fns[c.name](c.arguments))
    follow = [{"role": "assistant", "content": None, "tool_calls": [{"id": c.id, "type": "function",
               "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}]},
              {"role": "tool", "tool_call_id": c.id, "content": result}]
    out = ""
    async for it in llm.stream([{"role": "system", "content": SYSTEM}], audio, follow, schemas):
        out += it if isinstance(it, str) else ""
    print(f"  tool: {c.name}({c.arguments}) -> {result}\n  agent says: {out.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
