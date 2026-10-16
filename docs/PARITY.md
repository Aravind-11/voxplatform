# Parity with the paid Ultravox Realtime service (feature map, not a benchmark)

I never measured the paid service, so this compares **features**, not quality or speed. Quality and latency parity are unproven.

| Capability | Self-hosted here | Notes |
|---|---|---|
| Real-time voice, interruptions | yes (verified on a real call) | model is Ultravox v0.6; paid is newer |
| Telephony inbound | yes (verified once) | Twilio; BYO number |
| Telephony outbound, voicemail, DTMF | written, mocked | not placed on a real call yet |
| SIP trunk | **no** | possibly works through a Twilio SIP domain, untested |
| Transfer | cold and warm (written, mocked) | warm: summary whispered to the human; caller returns to the agent if no answer |
| Tools: HTTP, static/dynamic/automatic params, built-ins (hangUp, transfer, playDtmf, leaveVoicemail, queryCorpus) | yes | confirm-before-acting is an addition |
| Client tools (run in the caller's browser) | yes (written) | |
| Shared tool library | yes | |
| Knowledge-base search | keyword (BM25), optional hybrid with embeddings (toy-tested) | |
| Concurrent calls | ~24 per H100 comfortable, 32 usable (synthetic) | add nodes for more; paid scales for you |
| Multi-tenant API, keys, rotation, audit, key roles | yes | no SSO login |
| Usage, cost, CSV | yes (internal chargeback; no invoicing) | |
| Recording, dataset export | yes | consent is your responsibility |
| Dashboard, browser demo | yes | |
| Metrics | Prometheus text | |
| Voices | Kokoro presets | no cloning |
| Uptime / SLA | none | single operator, no soak test |
| Cost per minute | cheaper above ~1.5 average concurrent calls | docs/FINDINGS.md |
