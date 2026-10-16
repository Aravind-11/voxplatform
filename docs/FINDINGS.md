# Findings: self-hosted Ultravox platform vs the paid Ultravox Realtime service

Written 2026-10-15. Everything here is labelled by how it was established. **verified** = run, with the numbers shown.
**written** = code plus unit tests only. **designed** = a document, no code. Where I could not measure something, it says so.

## Verdict (read this first)
- **It is a working self-hosted platform**: real inbound phone call, barge-in, tools, concurrent calls, accounts, recordings,
  dashboard, Postgres/Redis scale-out. The open stack (Ultravox v0.6, Kokoro, Silero) is commercially licensable (docs/LICENSES.md).
- **It is not shown to be better than the paid service, and I am not claiming it is.** I never measured the paid service
  (no baseline). The paid product runs a newer model and better voices; this one runs Ultravox v0.6 and Kokoro-82M.
- **Where it wins on evidence**: cost per minute at sustained load, and control (your servers, your data, your tools, no vendor rate limits).
- **Where it loses or is unproven**: voice quality, model currency, reliability over time, and understanding of real human speech
  (3 mishearings on live calls; the synthetic test that scored 0.3% word error does not represent real callers).

## 1. Cost per minute
Inputs (cited, not measured by me): Lambda on-demand H100 **$2.86/h (PCIe), $3.78/h (SXM)** ([deploybase, Feb 2026](https://deploybase.ai/articles/h100-on-lambda-labs-pricing-specs-and-how-to-rent));
Twilio US local inbound **$0.0085/min** ([Quiq summary of Twilio pricing, Aug 2026](https://quiq.com/blog/twilio-voice-pricing/); third-party page, check Twilio's own);
paid Ultravox **$0.05/min** (from the task brief; I do not know what it includes, and I assume it excludes your phone line).
One GPU minute costs $0.048 to $0.063. Cost per call-minute = GPU minute cost / **average** concurrent calls (not peak: you pay for the idle card).

| Average concurrent calls | GPU $ per call-minute | + Twilio inbound | vs $0.05 paid |
|---|---|---|---|
| 1  | 0.048 to 0.063 | 0.057 to 0.072 | **costs more** |
| 4  | 0.012 to 0.016 | 0.021 to 0.024 | cheaper |
| 16 | 0.003 to 0.004 | 0.011 to 0.012 | ~4x cheaper |
| 24 | 0.002 to 0.003 | 0.011 | ~4.5x cheaper (floor is the phone line) |

**Break-even is about 1 to 1.5 average concurrent calls** (roughly 60 to 90 call-minutes every hour, around the clock).
Below that the paid service is cheaper. Not included: engineering and on-call time, storage, bandwidth, Twilio Media Streams
fees (not verified; check your Twilio bill), and the cost of the quality gap.

## 2. Latency and capacity (verified, 1x H100 80GB, fp8, Kokoro, Silero)
Real inbound phone call, server-side first audio (caller stops talking to first agent audio), excluding the Twilio and tunnel legs:
**648 to 1017 ms, median about 870 ms** over the 10 replies of one call (the first, cold, took 1642 ms and is excluded from the range). Barge-in fired 3 of 3.
About 0.64 s of that is the end-of-turn silence wait. `VOX_EOT=model` can shorten it when the model is confident the caller is done.

Synthetic load (every caller speaks at the same instant, the worst case), p50 / p95 of the same measure, the '4 procs' setting from STATUS.md:

| Calls | p50 / p95 | Completed |
|---|---|---|
| 16 | 1.02 / 1.49 s | 16/16 |
| 24 | 1.42 / 1.70 s | 24/24 |
| 32 | 1.77 / 2.25 s | 32/32 |
| 48 | 3.69 / 5.93 s | 48/48 |

Reading: comfortable to about 24 calls per GPU, usable to 32, degraded past that; the limit is Kokoro text-to-speech throughput
(one synthesis at a time per process). Real callers do not all speak at once, so real headroom is probably higher; unmeasured.
`tests/bringup/tts_procs.py` (process-based TTS scaling) has not been run yet, so whether more TTS processes raise the ceiling is open.
**Not measured:** latency of the paid service; end-to-end latency including Twilio; load with the end-of-turn model check switched on;
anything longer than a few minutes (no soak test, so no uptime claim).

## 3. What was built, and how far each part is proven
Full per-component table: docs/STATUS.md. Summary:
- **verified on the real stack**: real-time conversation, barge-in, a real inbound call, tool calling with audio input (18/20 correct tool and arguments; both misses were an un-called hangUp), transcription guard, load test.
- **verified on CPU with real servers**: the entire unit suite (195 tests) also passes against a **real PostgreSQL 16**; the Redis cap against a **real Redis 7** (shared across two simulated nodes, crash expiry, 40 racing acquires never exceed the cap); dashboard and demo page in **real headless Chromium** with a fake microphone; the deploy script run for real against stand-ins.
- **written, unit-tested with mocks, not run for real**: outbound calls and voicemail detection (Twilio mocked, no real dial), DTMF in and out over a live line (tones verified by an independent decoder; not heard on a call), client tools in a real browser, recording alignment against real Twilio timing, Prometheus scraping, `VOX_NO_ESPEAK`.
- **written, not run anywhere**: Dockerfile and compose (no Docker in the sandbox), Kubernetes manifests (no cluster).
- **not built**: SIP trunking, warm transfer (cold transfer to configured numbers exists), voice cloning, SSO/roles, automatic data-retention schedules, invoicing beyond the usage CSV, an embedding-based knowledge base (BM25 keyword search only).

## 4. Open problems, in order of importance
1. **Real-speech accuracy.** Live calls produced mishearings ("I like to cooking", "Yogen", a Thursday request read as "phone", after which the agent booked Tuesday).
   Mitigation shipped: confirm-before-acting tools. Not yet tried live. The real fix needs recorded human speech:
   enable `record: true` plus `VOX_DATASET=1` and every caller turn is saved with its transcript and end-of-turn score for evaluation.
2. **End-of-turn on fillers.** A human "Um," scored 0.98 "finished". Silence-based ending is the safe default; the model verdict is opt-in.
3. **No paid baseline** and **no soak test**: no claim about parity or uptime.
4. **TTS scaling** past about 24 calls per GPU: run `tts_procs.py`, then decide processes vs batching.

## 5. Reproduce
`python -m pytest tests/unit` (CPU, about 20 s); `VOX_TEST_DB=postgresql://... python -m pytest tests/unit` (Postgres);
`python tests/e2e/browser_test.py` (Chromium); `bash tests/e2e/deploy_dryrun.sh`; on the GPU box `bash scripts/deploy.sh` then `scripts/05_verify.sh`.
