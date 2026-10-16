# Tuning knobs (environment variables on the API process)

| Variable | Default | Effect |
|---|---|---|
| `VOX_END_MS` | 640 | silence before a caller turn ends (ms). Lower = snappier, more cut-offs |
| `VOX_PATIENT_MS` | 1000 | longer wait when the end-of-turn model says the caller probably is not done |
| `VOX_EOT` | off | `model` asks Ultravox "has the caller finished?" and can end the turn at ~320 ms; opt-in because it misjudged a human "Um," |
| `VOX_EOT_THRESHOLD` | 0.8 (deploy script: 0.9) | confidence needed to end early; higher = fewer cut-offs |
| `VOX_EOT_PROMPT` | v1 | v2 was measured worse; leave v1 |
| `VOX_TTS_CONCURRENCY` | 4 | simultaneous syntheses per process; measured: 1 thread is fastest per sentence, more threads add contention. Process-based scaling is untested |
| `VOX_NO_ESPEAK` | off | `1` removes the GPL espeak-ng fallback (docs/LICENSES.md) |
| `VOX_DEFAULT_CONCURRENCY` | 10 | per-tenant active-call cap; per-tenant override via `/admin/tenants` |
| `VOX_LOG_EVENTS` | off | `1` prints every session event (transcripts, latencies); contains caller speech, so keep off in production |
| `VOX_RECORD_DIR` / `VOX_DATASET` | off | recordings; `VOX_DATASET=1` also saves each recorded caller turn with transcript and end-of-turn score |
See `api/main.py` for the full list.
