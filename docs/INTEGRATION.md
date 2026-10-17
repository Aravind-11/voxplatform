# Integrating with BoostMyDeal-style APIs (and VoxSun, as a design)

Status labels: **tested** = unit/integration tests in this repo; **hypothetical** = an example I wrote to show the shape (BoostMyDeal's real API was not seen);
**design only** = not built, not tested (VoxSun).

## 1. Give the agent your business tools (tested with a mock server)
If your system has an OpenAPI 3 document, import it once. Each operation becomes a tool the agent can call.

```bash
curl -s -X POST $URL/v1/tools/import-openapi -H "X-API-Key: $KEY" -H 'content-type: application/json' -d '{
  "spec": { ...your OpenAPI JSON... },
  "baseUrl": "https://api.example.com",
  "headers": {"Authorization": "Bearer <service token>"},
  "prefix": "bmd_", "include": ["getDeal", "updateDeal"], "confirmWrites": true }'
```
- Path parameters (`/deals/{dealId}`), query parameters and JSON bodies are supported. Headers are stored write-only; the model never sees them.
- Non-GET operations require a spoken read-back and a "yes" (`confirmWrites`) so a misheard value is caught before data changes.
- Use an imported tool in an agent as `{"library": "bmd_getDeal"}`. Pin values the model must not change (an account id) as fixed parameters; see TOOLS.md.
- Tool URLs must resolve to public addresses (SSRF guard), no redirects, timeouts and size limits apply.

## 2. Start a call with context (tested)
Put `{{placeholders}}` in the agent's prompt/greeting: `"Hello {{customerName}}, calling about {{offer}}."`
Supply values per call (outbound or browser/app):

```bash
curl -s -X POST $URL/v1/calls/outbound -H "X-API-Key: $KEY" -H 'content-type: application/json' -d '{
  "agentId": "agt_...", "to": "+1YYYYYYYYYY",
  "variables": {"customerName": "Priya", "offer": "annual plan"},
  "metadata": {"dealId": "D-1042"} }'
```
A placeholder nothing fills is rejected before dialling. Agent defaults < call values < built-ins (call id, numbers).

## 3. Receive the outcome (tested)
Register one URL: `PUT /v1/webhook {"url": "https://you.example.com/hooks/voice"}` (account admin). The platform POSTs JSON for `call.started` and `call.ended`:

```json
{"type": "call.ended", "t": 1760000000.0,
 "call": {"id": "call_...", "agentId": "agt_...", "metadata": {"dealId": "D-1042"}, "...": "..."},
 "durationSeconds": 74.2,
 "transcript": [{"role": "agent", "text": "Hello Priya...", "t": 1760000001.2}]}
```
Verify the header `X-Vox-Signature: sha256=<hex>` = HMAC-SHA256 of the **raw body** with your webhook secret:

```python
import hmac, hashlib
def ok(secret: str, body: bytes, header: str) -> bool:
    return hmac.compare_digest("sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(), header)
```
Deliveries are retried by the dispatcher; make your handler idempotent on `call.id`.

## 4. Hypothetical flow (BoostMyDeal real endpoints unknown)
1. CRM decides to call lead L. It calls `POST /v1/calls/outbound` with `variables` and `metadata.dealId`.
2. During the call the agent uses imported tools, e.g. `getDeal` (read) and `updateDeal` (write, read-back first).
3. On `call.ended` the CRM stores the transcript and outcome against `metadata.dealId`.

## 5. VoxSun (SIP path: built, tested only against a simulation)
VoxSun is a SIP/PBX platform, so calls reach the engine through Asterisk and an AudioSocket transport. What exists, what is untested and what we need from VoxSun: **docs/SIP.md**.

## 6. Security notes
Use a least-privilege service token for tool calls; keep `confirmWrites` on; terminate TLS at the proxy; verify webhook signatures; keep API keys per role.
