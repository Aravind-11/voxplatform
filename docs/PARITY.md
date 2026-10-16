# Parity targets vs paid Ultravox (proposals until a baseline is measured)

Same harness (`harness/caller.py`) is meant to run against both. Targets are my proposals, not Ultravox's published figures.

| Metric | Target | How measured |
|---|---|---|
| Response latency (speech end -> first agent audio) p50 / p95 | <= 1.0 s / <= 1.8 s at 1 call; p95 <= 2.5 s at N calls | caller `response_ms` |
| Barge-in time-to-silence p95 | <= 400 ms | caller `silence_ms` |
| Calls completed under load | 100% at the claimed concurrency | caller `ok` |
| Telephony | real inbound Twilio call completes | manual, recorded |
| Cost/min | report GPU $/hr / (calls x minutes) vs $0.05/min paid | computed from measured concurrency |

Known gaps vs paid: model is v0.6 (not the newer hosted one); TTS is Kokoro (not their voices); no voice cloning;
end-of-turn is silence-based; single node; no proven uptime.
