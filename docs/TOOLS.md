# Tools

An agent's `tools` list holds names or definition objects. Created with `POST /v1/agents` (validated; bad
definitions are rejected with HTTP 422 at creation, not mid-call) and overridable per call in `POST /v1/calls`.

## HTTP tool
```json
{"name": "lookup_order", "description": "Find an order by its number.",
 "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
 "http": {"url": "https://api.example.com/orders", "method": "POST",
          "headers": {"Authorization": "Bearer sk-..."}, "timeoutS": 8},
 "static": {"tenant": "acme"},
 "automatic": ["callId", "from"]}
```
- `parameters`: JSON Schema of what the **model** fills in (dynamic). Unknown arguments are dropped.
- `static`: always sent, never shown to the model, and the model cannot override them.
- `automatic`: platform-filled: `callId`, `from`, `to`, `callSid`.
- The model sees only name, description and `parameters`. Header values are write-only: the API returns `***`.
- POST/PUT/PATCH send JSON; GET sends query parameters. The response (text or JSON, max 4000 chars) goes back to
  the model; HTTP errors and timeouts go back as `{"error": ...}` so the agent can say something sensible.
- Response header `X-Vox-Response-Type: hang-up` (or `X-Ultravox-Response-Type`) ends the call after the agent speaks.
- Tools never run for a reply that is later cancelled (speculative reply) — only after the turn is committed.

## Confirm before acting (guards against mishearing)
Add `"confirm": true` to an HTTP tool, or write a named tool as `{"tool": "book_slot", "confirm": true}`. The first
call does not run the tool: the model gets the details and must read them back to the caller. The tool runs only
on a second call with `confirmed: true`, the same details, made in a **later caller turn**; one confirmation
authorizes one action, and changed details need a new confirmation. A misheard request becomes an audible
"Tuesday at 11, correct?" the caller can fix, instead of a silent action. Limit: it relies on the model hearing a
short yes/no correctly, which is far easier than hearing a day or a name.

## Built-ins
- `"hangUp"` or `{"builtin": "hangUp", "strict": false}`: ends the call after the agent finishes speaking.
  `strict: true` (default) hangs up even if the caller talks over the goodbye; `false` lets the caller keep the call.
- `{"builtin": "transfer", "targets": {"billing": "+15551234567"}}`: the model picks a target **name**; it can never
  dial a number of its own. Phone calls only; needs `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` on the server.
  Implemented as a Twilio call redirect after the agent has said it is transferring (not SIP REFER; no warm transfer).

## Safety
URLs must be http(s) and resolve to public addresses only (loopback, private, link-local and metadata addresses are
refused; `VOX_TOOL_ALLOW_PRIVATE=1` lifts this for local tests). Redirects are not followed. Known gap: the address is
checked just before the request, so DNS rebinding could race it; production should send tool traffic through an egress proxy.

## Versus the paid product (from Ultravox's public tool docs; details they do not document are marked ?)
| | Ultravox Realtime | This platform |
|---|---|---|
| HTTP (server) tools | yes | yes (written, unit-tested; real-model check pending) |
| Client tools (run in the caller's app via SDK) | yes | no |
| Reusable tool library across agents | yes (durable tools) | no (definitions live in each agent) |
| Dynamic / static / automatic parameters | yes (+ per-call overrides) | dynamic, static, automatic; no call-time overrides |
| Per-call auth tokens | yes | no; credentials stored in the agent (write-only) |
| Response-type control | hang-up, new-stage | hang-up only |
| hangUp | yes | yes |
| Call transfer | cold + warm over SIP | cold, via Twilio redirect, to configured targets only |
| playDtmfSounds, leaveVoicemail, queryCorpus (RAG) | yes | no |
| Tool timeouts / async tools | not documented ? | 0.5-30 s timeout per tool; no async tools |
| Egress protection | not documented ? | SSRF guard, no redirects, truncation |

## Added later
- `{"builtin":"queryCorpus","corpus":"faq","topK":3}`: keyword (BM25) search over a knowledge base created with `POST /v1/corpora` and `POST /v1/corpora/{name}/documents`.
- `"playDtmf"`: the agent presses keypad digits (phone menus, extensions). Keypad digits the caller presses arrive as a text turn: `[The caller pressed these keypad digits: 123]` (ends on `#` or after 2 s).
- `{"builtin":"leaveVoicemail","message":"..."}`: speaks the message and hangs up; the agent calls it when it hears a voicemail greeting. For outbound calls, `voicemailMessage` also turns on Twilio answering-machine detection and plays the message automatically.
- `{"builtin":"client","name":"open_page","description":"...","parameters":{...}}`: a function that runs in the caller's browser on `/demo`-style WebSocket calls. The server sends `{"type":"client_tool_invocation","id","name","args"}`; the page answers `{"type":"client_tool_result","id","result"}` (or `"error"`).
- `{"library":"name"}`: use a tool stored once with `PUT /v1/tools/{name}`; edits apply to every agent that uses it at its next call.
