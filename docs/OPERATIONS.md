# Operations

**Secrets**: admin key, metrics token, Twilio token, Postgres password. Never commit them; the deploy script reads the Twilio token
with hidden input. Rotate tenant API keys from the dashboard (Keys) or `POST /v1/keys/{id}/rotate`; the old key stops at once.
**Recording and consent**: recording is off unless an agent sets `record: true` and `VOX_RECORD_DIR` is set. Many places require
telling callers a call is recorded (some require everyone's consent); put it in the agent's `greeting`. `DELETE /v1/calls/{id}`
removes the call, transcript, recording and saved utterances. Retention: set `retentionDays` per tenant (`PATCH /admin/tenants/{id}`) or `VOX_RETENTION_DAYS` for all; ended calls older than that are erased hourly (`POST /admin/purge` runs it now). Live calls are never touched.
**Roles**: create keys as admin, developer (agents, tools, calls) or viewer (read only) from the dashboard Keys tab; use viewer keys for dashboards and reporting.
**Backups**: Postgres (`pg_dump`) or the SQLite file; recordings directory. Redis holds only call counters: losing it is harmless.
**Monitoring**: `GET /metrics` with `Authorization: Bearer $VOX_METRICS_TOKEN`: `vox_active_calls`, `vox_calls_total`,
`vox_first_audio_ms` (histogram), `vox_errors_total{where}`, `vox_tool_calls_total`. Suggested alerts: `vox_errors_total` rate > 0 for
5 min; p95 `vox_first_audio_ms` > 2500 ms; `vox_active_calls` near your measured limit.
**Scaling out**: shared Postgres + Redis; any node can take any call (call state is in the database). One limit: the answering-machine
callback must reach the node that holds the live call, so use sticky routing by call id or a single node if you use voicemail detection.
**Taking a node out**: `POST /admin/drain` (readiness goes 503, live calls finish); the Kubernetes preStop hook does this.
**Outbound dialing** costs money and can be abused: per-tenant daily cap (`VOX_OUTBOUND_PER_DAY`, default 50) and an optional
country-prefix allow-list (`VOX_OUTBOUND_PREFIXES=+1`).
**Instance housekeeping (Lambda)**: terminate the instance when not in use; the persistent filesystem keeps weights and database and bills weekly.
