"""Does running Kokoro in several PROCESSES (each handling one sentence at a time) scale?
Threads in one process collapsed past 4 (tts_bench.py); one thread alone did ~11 sentences/s.
  python tests/bringup/tts_procs.py          (vLLM may stay up but should be idle)
"""
import multiprocessing as mp
import os
import statistics as st
import time

S = ["I don't have access to specific opening hours for locations.",
     "You might want to check the official website or contact the business directly.",
     "Let me know if you need anything else!", "Hi!",
     "Your appointment is scheduled for Tuesday morning at nine.", "Is there a name I should use for the booking?"]
PER_PROC = 24


def worker(i, start, q, torch_threads):
    import numpy as np
    import torch
    torch.set_num_threads(torch_threads)
    from kokoro import KPipeline
    p = KPipeline(lang_code="a")
    for t in S * 2:
        list(p(t, voice="af_bella"))                      # warm-up
    q.put(("ready", i))
    start.wait()
    t0, lat = time.perf_counter(), []
    for t in (S * 20)[:PER_PROC]:
        a = time.perf_counter()
        sum(len(np.asarray(x)) for _, _, x in p(t, voice="af_bella"))
        lat.append((time.perf_counter() - a) * 1000)
    q.put(("done", i, time.perf_counter() - t0, lat))


def run(n, torch_threads):
    ctx = mp.get_context("spawn")
    start, q = ctx.Event(), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, start, q, torch_threads)) for i in range(n)]
    for p in ps:
        p.start()
    for _ in range(n):
        assert q.get()[0] == "ready"
    start.set()
    res = [q.get() for _ in range(n)]
    for p in ps:
        p.join()
    wall = max(r[2] for r in res)
    lat = sorted(x for r in res for x in r[3])
    return n * PER_PROC / wall, st.median(lat), lat[int(len(lat) * 0.95) - 1]


if __name__ == "__main__":
    print(f"cores: {os.cpu_count()} | each process runs {PER_PROC} sentences back to back")
    for tt in (1, 2):
        print(f"\ntorch threads per process: {tt}")
        print("procs  sentences/s   per-sentence p50 / p95 (ms)")
        for n in (1, 2, 4, 6, 8):
            tput, p50, p95 = run(n, tt)
            print(f"{n:5d}  {tput:11.1f}   {p50:6.0f} / {p95:.0f}", flush=True)
