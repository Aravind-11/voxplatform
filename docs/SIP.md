# SIP path (VoxSun and any SIP provider or PBX)

## What we found out about VoxSun (from VoxSun's own published technical description, https://voxsun.com/llms-full.txt)
- VoxSun is a VoIP business-phone platform (virtual numbers, PBX). It lists SIP authentication, SRTP/TLS and the codecs G.711, G.722, G.729 and Opus.
- It has a REST API (`POST /api/v1/calls/initiate`, call recording endpoints) and webhooks (`call.incoming`, call started/answered/ended) with signatures. Its detailed API documentation is given only through sales.
- It does **not** describe WebSocket audio streaming or AI voice agents. So the realistic way in is **SIP**: a call to a VoxSun number is routed to a SIP endpoint we run.
- Not known (needs VoxSun): whether the account lets us point numbers at an outside SIP server, the SIP server address and credentials, and whether they want registration or IP-based trust.

## How it works here
```
caller -> VoxSun -- SIP/RTP --> Asterisk -- AudioSocket (TCP, 8 kHz or 16 kHz PCM) --> voice engine (this repo)
```
Asterisk (open source) does the SIP and audio-over-IP work. Before AudioSocket opens, the dialplan posts the call
UUID, the caller number, and the dialed number to `POST /internal/sip/setup` with header `X-Vox-Sip-Token`.
The engine receives each call's audio over a plain TCP connection
(`transports/audiosocket.py`) and runs the same agent, tools, interruption handling and storage as the Twilio path.
Calls are stored with medium `sip`.

`PUT /v1/numbers/{e164}` stores which agent answers that dialed number for the tenant. A fresh setup (60 seconds)
stores the caller on `metadata.from` and the dialed number on `metadata.to`. The session copies those into
`CallContext.caller`, `CallContext.callee`, and the `{{caller}}` / `{{callee}}` variables. With no fresh setup,
or no number row, the call is answered as `VOX_SIP_AGENT`. An empty `VOX_SIP_TOKEN` turns the setup route off
(404); a wrong token is 403.

## What is verified and what is not
| Part | State |
|---|---|
| AudioSocket transport: message format, 8 kHz or 16 kHz (`VOX_AUDIOSOCKET_RATE`, g722/slin16) audio in, paced real-time audio out, interruption (queue cleared), DTMF in, hangup both ways | **tested** (unit tests incl. a simulated Asterisk talking to the real app; no real Asterisk) |
| `VOX_AUDIOSOCKET_*` settings and the call lifecycle (create, admit, session, finish, webhook) | **tested** with the simulated Asterisk |
| Asterisk config (`deploy/sip/`), `deploy/Dockerfile.asterisk`, compose `asterisk` service (profile `sip`) | **written from Asterisk's documentation, never run**: compose file parses; no Asterisk was available here |
| Real call from VoxSun | **not done**: needs VoxSun credentials and a test number |

## Set it up (once VoxSun gives SIP details)
1. Fill the uppercase values in `deploy/sip/pjsip.conf` (server, username, password or the provider's IP).
2. Create the agent that should answer calls with no number mapping, then in `deploy/.env` set `VOX_AUDIOSOCKET_PORT=9092`, `VOX_SIP_AGENT=<agent id>`, and `VOX_SIP_TOKEN` to a long random secret. Compose passes that same value into Asterisk as `SIP_SETUP_TOKEN`. An empty token leaves `POST /internal/sip/setup` off (404), so every SIP call stays on `VOX_SIP_AGENT`.
3. `docker compose -f deploy/docker-compose.yml --env-file deploy/.env --profile sip up -d`
4. Map a dialed number: `PUT /v1/numbers/{e164}` with body `{"agentId": "<agent id>"}` (E.164, for example `+15551234567`) and the tenant API key. 422 means that agent is not in the tenant. 409 means another tenant already owns the number.
5. Check: `docker compose exec asterisk asterisk -rx "pjsip show registrations"` and `... "module show like audiosocket"`.
6. Call the VoxSun number. In the engine logs you should see a new `sip` call on the mapped agent. `{{caller}}` and `{{callee}}` are filled from that setup.

Without Docker: install Asterisk 20.17+, 21.12+, or 22.7+ (needs the AudioSocket channel modules; older releases drop kind 0x12 and the agent is silent), copy the files from `deploy/sip/` (including `sip_setup.py`; `pjsip.conf` includes `pjsip.register.conf`) to `/etc/asterisk/`, export `SIP_SETUP_TOKEN` to the same secret, and run the engine with `VOX_AUDIOSOCKET_PORT=9092 VOX_AUDIOSOCKET_AGENT=<id> VOX_AUDIOSOCKET_HOST=127.0.0.1 VOX_AUDIOSOCKET_RATE=16000 VOX_SIP_TOKEN=<secret>` (change `api1` in `extensions.conf` to `127.0.0.1` for both the AudioSocket dial and the setup URL).

## Outbound, transfer, voicemail
`POST /v1/calls/outbound` with `"channel": "sip"` (default is still `"twilio"`) originates through Asterisk ARI: `POST {ASTERISK_ARI_URL}/ari/channels` with `endpoint=PJSIP/{to}@voxsun`, `app=vox`, and basic auth from `ASTERISK_ARI_USER` / `ASTERISK_ARI_PASSWORD`. A `voicemailMessage` sets `appArgs` to `amd,<call id>`. The Stasis app posts `AnsweredBy` (form or JSON) to `POST /sip/amd/{call id}` with header `X-Vox-Sip-Token` (`VOX_SIP_TOKEN`). An empty token turns that route off (404); a mismatch is 403. On `machine_end` while that call is live here, the engine speaks the message and hangs up. If the result arrives before a session exists, the originated ARI channel is deleted. `AnsweredBy` and `callSid` are merged into metadata, so neither write drops the other.

The dialplan POSTs `uuid` and `channelId` to `POST /sip/bind` with the same `X-Vox-Sip-Token` before `AudioSocket`. A cold transfer does not `POST /ari/channels/{id}/redirect`: the channel is inside AudioSocket, not Stasis, and Asterisk answers that redirect with 409. The tool stores `transferTo` (the configured E.164 number) on the call and, after the goodbye, hangs up the AudioSocket so the dialplan continues. The next priority GETs `/sip/transfer?uuid=` (same token) and runs `Dial(PJSIP/{number})` with no `@voxsun`, or `Hangup()` when the body is empty. If the target was not stored, the tool returns an error and does not report success. Warm transfer is not done on SIP; the tool returns an error and does not drop the AudioSocket. `DELETE /ari/channels/{id}` hangs up the originated channel id and is not sent the AudioSocket UUID. Originate sends that id as `channelId` and stores it before the request; a timeout or connect error best-effort deletes it. A machine hang-up counts as finished only on HTTP 2xx or 404. A timeout, connect error, or HTTP 5xx leaves the call `created` so a later `/sip/amd` deletes again. An empty `AnsweredBy` does not erase a value already stored. The channel id is one path segment, so a slash or `..` cannot select another ARI resource. A SIP call uses Asterisk when `ASTERISK_ARI_URL` is set. A Twilio call keeps using Twilio even if that URL is also set.

## Known limits
- **Caller id:** AudioSocket carries only the call UUID. The dialplan posts the caller id and the dialed number first (`caller`, plus `exten` or `callee`). Only digits and a leading `+` are forwarded (`+` is sent as `%2B`). Anything else is stripped before the request, so it cannot be run as a shell command. The token stays in `SIP_SETUP_TOKEN` and is not placed on the shell command line. Metadata `from` / `to` fill `{{caller}}` and `{{callee}}`. A setup older than 60 seconds is ignored for a new call. Hang-up, DELETE, and retention remove the setup row. An empty `VOX_SIP_TOKEN` leaves setup and `/sip/recover` off.
- **Two Asterisks, one registration:** `extensions.conf` dials `api1`. Compose service `asterisk2` mounts `extensions.api2.conf` (dials `api2`) and does not load the trunk registration. A lost AudioSocket can be continued; see below.
- **Security:** the AudioSocket port has no authentication. Keep it on a private network and never publish it.
- **Outbound SIP audio:** `POST /v1/calls/outbound` with `"channel": "sip"` originates through Asterisk ARI (`ASTERISK_ARI_URL`), not VoxSun's `calls/initiate`. See below. Nothing in this repo answers Stasis app `vox`, so the new channel is not attached to AudioSocket by itself.
- **Open ports for Asterisk:** UDP 5060 and the RTP range 10000-10100. Behind NAT, set the public address in `pjsip.conf`.
- Asterisk is GPL-2.0 software run as a separate program; this repo does not include or link it. Get the same licence check you would for any GPL component.

## More than one Asterisk
One Asterisk is a single point of failure. The intended layout is carrier failover, or Kamailio (`deploy/sip/kamailio.cfg`, not run in CI) in front of the compose services `asterisk` and `asterisk2` (both profile `sip`), plus a TCP load balancer in front of AudioSocket. Kamailio pins each INVITE by the last character of the Call-ID local-part (the character before `@`), not the host. Only `asterisk` registers to the carrier; `asterisk2` is inbound from that proxy and sends AudioSocket to `api2`. Each API process still applies its own concurrency cap before it admits a call, so a dead node stops taking new sockets once the balancer stops using it.

If the AudioSocket stream ends and the engine did not hang up (`ended_by_us` is false), `finish` already stores `end_reason=stream_closed`. A re-dial does not know the old call id. Whoever still has the AudioSocket UUID (the dialplan's `CALLUUID`, or the proxy) POSTs it as `uuid` to `/sip/recover` with header `X-Vox-Sip-Token` (`VOX_SIP_TOKEN`). The handler finds the call by metadata `callSid` and, when the agent did not hang up, creates a new SIP call with `resumedFrom` set so the next session keeps the transcript. An empty token leaves the endpoint off. An agent hang-up is not resumed.
