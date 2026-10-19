# Status — what is actually verified

Legend: **verified** = executed with the stated result; **written** = code + unit tests on CPU with fakes only;
**designed** = described, no code.  GPU results below were measured on 1x H100 80GB (Lambda), Ultravox v0.6 Qwen3-32B fp8 on vLLM, Kokoro-82M TTS, Silero VAD, 2026-10-15.

| Capability | State | Evidence / gap |
|---|---|---|
| G.711 µ-law codec, resampling (8k<->16k<->24k) | verified | exhaustive check vs independent reference; streaming == one-shot within 1 LSB |
| Sentence chunker (first-chunk early split, `<think>` strip) | written | unit tests |
| Turn manager (speculate / commit / resume / abort) | written | unit tests; thresholds are untuned guesses |
| Barge-in: clear, cancel, history trimmed to what was heard | written | unit tests with fake clock; real playback timing unmeasured |
| Session orchestrator + tool calling gated on commit | written | unit tests w/ fakes, mutation-checked |
| vLLM client (audio parts, streaming tool-call assembly) | written | fake-client tests; **real server message shape unverified** |
| Silero VAD, Kokoro TTS adapters | written | Kokoro adapter tested with a fake pipeline; real libs never run |
| Twilio Media Streams transport + webhook + signature | written | message formats checked against Twilio docs; signature equals Twilio's own `RequestValidator`; no real call placed |
| Browser WebSocket transport | written | unit + local e2e |
| Multi-tenant API, API keys (hashed), tenant isolation, per-tenant concurrency cap, signed webhooks w/ retry | written | 74 unit tests incl. WebSocket/TwiML paths; SQLite single node only |
| Synthetic caller harness | written | runs against the real app with **fake models** (below) |
| Ultravox model on vLLM (fp8, 32B on 1xH100) | verified | loads; TTFT p50 33 ms @1, 291 ms @32 concurrent, 0 errors |
| End-to-end latency with real models | verified | speech end -> first agent audio 0.71 s @1 call (0.64 s of it is the end-of-turn wait) |
| Concurrent calls per GPU | verified (synthetic, synchronized callers) | see table below |
| Barge-in on real stack | verified | agent silenced 225 ms after caller speech onset; history trimmed to heard text |
| Speech-to-text of caller turns (same model) | verified | exact transcript of test utterance |
| End-of-turn: model verdict (Ultravox yes/no logprob) | verified (synthetic speech only) | AUC 0.965 clean / 0.922 phone-degraded (50 Kokoro utterances, 25 finished + 25 unfinished); verdict ~145 ms. Clean: threshold 0.8 ends 80% of finished turns early at 0% false cut-off; phone: 0.9 -> 64% / 4%. 25 per class: upper 95% bound on "0% false cut-off" is ~11%. On the first real phone call only 1 of 6 verdicts reached 0.9 (tiny sample). Opt-in `VOX_EOT=model` |
| End-of-turn prompt v2 (filler-aware) | **rejected** | worse than v1 on the 61-utterance set: AUC 0.888 vs 0.939 clean, 0.870 vs 0.947 phone; fast-end coverage at 0.9 falls 73% -> 50% clean, 60% -> 43% phone. v1 stays default. Neither explains the real-call filler misfires ('Um,' scored 0.98 on a human; synthetic 'Um,' does not): the eval needs recorded human speech |
| End-of-turn: acoustic cues (pitch fall, energy decay, filler hum; no model) | **failed, do not use** | AUC 0.46 clean / 0.43 phone = at or below chance on the same set (synthetic voices have unnatural pitch, so this does not prove it fails on humans, but it is not validated). Averaging with the model is worse than the model alone. Off by default |
| Caller-turn transcription guard | verified | before: model answered "I am an AI language model..." to a question instead of transcribing (word error 11.8% clean, polluted history). After: strict prompt + length check + one retry. WER 0.3% clean / 0.6% phone / 1.6% @20 dB / 4.3% @10 dB (40 synthetic clips, 2 voices; real speakers will be worse) |
| Real phone call over Twilio (+ Cloudflare quick tunnel) | verified once, informally | caller -> Twilio -> API -> Ultravox -> Kokoro -> caller worked; barge-in fired 3 of 3; server-side first audio 713-980 ms (mean ~850 ms), excludes Twilio/tunnel legs; one call only |
| Tools: HTTP tools, hangUp, transfer (Twilio redirect), call actions after speech | written | unit tests (135+) incl. SSRF guard, static-param precedence, secret redaction, hang-up after goodbye, strict vs non-strict; real model with audio input (vLLM hermes parser, synthetic speech): correct tool and sensible arguments 9/10 clean and 9/10 phone-degraded; the miss both times was not calling hangUp on a 'Thanks, that's all I needed. Goodbye!'; full round trip (tool result -> spoken answer) works; no tool called on small talk. Not yet tried on a live phone call |
| Agent greets first (`greeting` on the agent or per call; fixed text, no model call, interruptible) | written | unit tests incl. barge-in over the greeting and history trimming; not yet heard on a real call |
| Real-call finding: model misheard a spoken request and acted on it | **open problem, mitigated** | live call: caller asked about Thursday; transcript read 'Or better yet, phone.'; the model then booked Tuesday 11:00 unprompted. Earlier calls: 'I like to cooking', 'Yogen'. The synthetic test (0.3% WER) does not represent this speaker/line. Mitigation: `confirm` tools (read back + yes in a later turn), unit-tested, not yet tried live. State-changing tools can require a read-back of names, dates, times, numbers, and amounts (`confirmSlots`, or any `confirm` tool) before they run. Recognition stays model-bound. Real fix needs measurement on recorded human speech |
| UltraVAD | not used | page states no license; own reported accuracy 77.5% / precision 69.6% |
| Postgres store | **verified** (CPU, real server) | the full unit suite (205 tests) passes against PostgreSQL 16; SQLite databases from the first release migrate in place (tested) |
| Redis call cap shared by API nodes | **verified** (CPU, real Redis 7) | two limiter instances share one cap; a crashed node's calls expire (TTL), live nodes refresh; 40 racing acquires never exceed the cap. Not run behind a real multi-node load balancer |
| Drain mode (`/admin/drain`, readiness 503, preStop hook) | written | unit-tested, including the hook's wait loop; not run in a cluster |
| Account features: API key create/revoke/rotate (last-key protection), audit log, per-tenant isolation | written | unit-tested incl. cross-tenant attempts; on SQLite and Postgres |
| Shared tool library (`{"library":"name"}`, resolved at call time, secrets write-only) | written | unit-tested |
| Knowledge-base search (`queryCorpus` tool, BM25 over passages) | written | unit-tested; keyword search only, retrieval quality on real documents unmeasured |
| Usage by day/agent, rate and cost, CSV export, voices list, speaking speed | written | unit-tested; speed not heard on real audio |
| Outbound calls via Twilio REST (caps per day, prefix allow-list), call status callbacks | written | Twilio mocked; no real outbound call placed |
| Voicemail: answering-machine detection callback + `leaveVoicemail` tool | written | Twilio mocked; not tried against a real answering machine |
| DTMF keypad in (digits -> text turn) and out (`playDtmf` tones) | written | in-bound follows Twilio's documented `dtmf` event; tones decode correctly with an independent Goertzel detector; not tested on a live call |
| Client tools (functions run in the caller's browser) | written | roundtrip, error and timeout unit-tested incl. through a real websocket; only a demo handler exists in the page |
| Call recording (stereo WAV, opt-in per agent), download and delete endpoints, per-utterance dataset | written | unit tests + fake-model end-to-end; alignment against real Twilio playback timing unmeasured; you need callers' consent |
| Prometheus `/metrics` (protected) | written | text format unit-tested; no Prometheus has scraped it |
| Browser demo page (`/demo`) and admin dashboard (`/dashboard`) | **verified** (fake model, real browser) | headless Chromium with fake microphone: greeting, speech heard, agent answers, call + transcript + recording appear in the dashboard; screenshots inspected. Not tried with a real mic or the real model |
| `scripts/deploy.sh` (tunnel, API, tenant, agent, optional Twilio webhook via REST) | **verified** (stand-ins) | run for real against stubbed vLLM/cloudflared/model; the output and created agent were checked. Not yet run on the GPU box |
| Docker image, docker-compose stack, Kubernetes manifests | written, not run | no Docker or cluster available when written; YAML parses |
| License audit | done | docs/LICENSES.md; espeak-ng (GPL-3.0) flagged, `VOX_NO_ESPEAK=1` written, not yet run on the GPU |
| Roles on API keys (admin / developer / viewer), last-admin protection | written | unit-tested, every mutating endpoint checked for viewer and developer; keys from before roles become admins (migration tested) |
| Data retention (per-tenant days or server default, hourly sweep, `POST /admin/purge`) | written | unit-tested incl. files, live calls untouched; the hourly background task itself has not run for hours |
| Warm transfer (summary whispered to whoever answers; caller returned to the agent if nobody does) | written | Twilio mocked; TwiML and callbacks unit-tested incl. signature and markup stripping; not tried on a real call. The returned agent keeps the earlier transcript and opens with the not-available apology rather than a fresh greeting |
| Embedding (hybrid) knowledge-base search | written | fused with BM25, falls back to keywords if the embedding server is down; tested with a toy embedder only. `scripts/01b_serve_embed.sh` is written, not run (vLLM flag differs by version); no real model, no retrieval-quality measurement |
| SIP (inbound, via Asterisk + AudioSocket), voice cloning, SSO login | SIP: **written, tested only against a simulated Asterisk**; cloning and SSO: **not built** | SIP details and limits: docs/SIP.md (never run against real Asterisk or VoxSun; outbound SIP originates the call but nothing yet attaches it to the agent; warm transfer not done on SIP). Cloning: Kokoro cannot clone, and no commercially licensed cloning model was verified. SSO: only API-key roles exist |

## Harness numbers with fake models (platform overhead only, NOT model latency)
Fake LLM 50 ms, fake TTS ~10 ms, localhost, 20 concurrent callers: response p50 652 ms / p95 657 ms;
barge-in time-to-clear p50 194 ms / p95 198 ms; 20/20 calls completed.
Reading: the response floor is the 640 ms end-of-turn silence rule, and barge-in floor is the 192 ms
start-confirmation window, i.e. platform overhead is small and the thresholds dominate. Real numbers need the GPU.

## Real-stack load test (1x H100, all callers speak at the same instant = worst case)
Response = caller speech end -> first agent audio, measured by the client.
The published "v2, 4 procs" numbers require `VOX_PROCS=4` and `VOX_NODE_MAX_CALLS` split across those processes (`VOX_NODE_MAX_CALLS // VOX_PROCS` per process). The sum of those per-process caps does not exceed the node cap; a positive cap smaller than the process count is rejected rather than rounded up to 1 each. That is four separate uvicorn processes on ports 8080–8083, not `--workers 4`: a WebSocket sticks to the process that accepted it, and in-process state (live calls, VAD) is per process. Those processes need a Postgres `VOX_DB` (`postgres://` or `postgresql://`); the default SQLite file raises `database is locked`.

| Calls | v1 (eager TTS) p50 / p95 | v2 just-in-time TTS, 1 proc | v2, 4 procs | completed (v2, 4 procs) |
|---|---|---|---|---|
| 1  | 0.71 s | 0.71 s | - | 1/1 |
| 8  | 0.98 / 1.07 s | 0.73 / 0.84 s | - | 8/8 |
| 16 | 1.98 / 2.17 s | 0.95 / 1.34 s | 1.02 / 1.49 s | 16/16 |
| 24 | 1.91 / 4.91 s | 1.61 / 1.89 s | 1.42 / 1.70 s | 24/24 |
| 32 | 4.02 / 4.38 s, 25/32 ok | 2.33 / 2.96 s | 1.77 / 2.25 s | 32/32 |
| 48 | - | - | 3.69 / 5.93 s | 48/48 |

Stage timing at 16 calls (v1): model first token p50 63 ms; TTS per sentence p50 1433 ms -> TTS was the bottleneck,
fixed by synthesizing only the first sentence eagerly and later ones just in time (1.5 s lookahead).
Remaining limit above ~16 calls: TTS throughput on one GPU (Kokoro runs batch-1). Next levers: batched TTS worker, CUDA MPS.

## Added since (labelled)
- **verified (real processes, fake models):** 2-node HA behind Caddy with Postgres, Redis, Prometheus, Alertmanager; node kill, drain, alert fire/resolve (tests/e2e/ha_test.py, 21 checks).
- **written + unit tested:** multilingual voices, call variables, OpenAPI import, signed call.ended webhook with transcript, call recovery, cost model, resource logging scripts.
- **written, never run:** Docker images, deploy/up.sh against real Docker, Kubernetes; Stripe/billing ledger groundwork (not wired); non-English audio.
- **not done:** VoxSun integration, soak test, GPU resource-consumption capture, paid-service baseline.
- **written + tested against a simulated Asterisk:** SIP path via AudioSocket (transports/audiosocket.py, docs/SIP.md). **written, never run:** deploy/sip Asterisk config, Dockerfile.asterisk, compose `asterisk` service. **not done:** a real VoxSun call.
