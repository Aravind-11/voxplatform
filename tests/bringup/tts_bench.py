"""Where does Kokoro's time go, and does running it in parallel threads help?
python tests/bringup/tts_bench.py     (run with the vLLM server idle)
"""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from kokoro import KPipeline

p = KPipeline(lang_code="a")
print("device:", next(p.model.parameters()).device, "| torch threads:", torch.get_num_threads(), "| cuda:", torch.cuda.is_available())
S = ["I don't have access to specific opening hours for locations.",
     "You might want to check the official website or contact the business directly.",
     "Let me know if you need anything else!", "Hi!"]


def synth(t):
    t0 = time.perf_counter()
    n = sum(len(np.asarray(a)) for _, _, a in p(t, voice="af_bella"))
    return (time.perf_counter() - t0) * 1000, n / 24000


for t in S * 2:
    synth(t)
r = [synth(t) for t in S * 6]
ms, sec = st.median(x[0] for x in r), st.median(x[1] for x in r)
print(f"1 thread: {ms:.0f} ms per sentence for {sec:.1f} s of audio  (real-time factor {sec / (ms / 1000):.0f}x)")
try:                                                            # text -> phonemes only (CPU/Python)
    g = []
    for t in S * 6:
        t0 = time.perf_counter(); p.g2p(t); g.append((time.perf_counter() - t0) * 1000)
    print(f"  of which text->phonemes (CPU): {st.median(g):.0f} ms -> {st.median(g) / ms:.0%} of the total")
except Exception as e:                                          # noqa: BLE001
    print("  (g2p timing unavailable:", repr(e)[:80], ")")
for k in (2, 4, 8, 16):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(k) as ex:
        out = list(ex.map(synth, (S * 40)[: k * 4]))
    dt = time.perf_counter() - t0
    print(f"{k:2d} threads: {len(out) / dt:5.1f} sentences/s | per-sentence p50 {st.median(x[0] for x in out):5.0f} ms")
print("\nReading: if 1-thread time is mostly text->phonemes, the limit is CPU/Python, so more PROCESSES help and GPU batching won't.")
print("If it is mostly the model, throughput should rise with threads/batching; flat throughput = serialized.")
