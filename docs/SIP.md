# SIP path (VoxSun and any SIP provider or PBX)

## What we found out about VoxSun (from VoxSun's own published technical description, https://voxsun.com/llms-full.txt)
- VoxSun is a VoIP business-phone platform (virtual numbers, PBX). It lists SIP authentication, SRTP/TLS and the codecs G.711, G.722, G.729 and Opus.
- It has a REST API (`POST /api/v1/calls/initiate`, call recording endpoints) and webhooks (`call.incoming`, call started/answered/ended) with signatures. Its detailed API documentation is given only through sales.
- It does **not** describe WebSocket audio streaming or AI voice agents. So the realistic way in is **SIP**: a call to a VoxSun number is routed to a SIP endpoint we run.
- Not known (needs VoxSun): whether the account lets us point numbers at an outside SIP server, the SIP server address and credentials, and whether they want registration or IP-based trust.

## How it works here
```
caller -> VoxSun -- SIP/RTP --> Asterisk -- AudioSocket (TCP, 8 kHz audio) --> voice engine (this repo)
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
| AudioSocket transport: message format, 8 kHz audio in, paced real-time audio out, interruption (queue cleared), DTMF in, hangup both ways | **tested** (4 unit tests incl. a simulated Asterisk talking to the real app; no real Asterisk) |
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

Without Docker: install Asterisk 18+ (needs the AudioSocket modules), copy the files from `deploy/sip/` (including `sip_setup.py`) to `/etc/asterisk/`, export `SIP_SETUP_TOKEN` to the same secret, and run the engine with `VOX_AUDIOSOCKET_PORT=9092 VOX_AUDIOSOCKET_AGENT=<id> VOX_AUDIOSOCKET_HOST=127.0.0.1 VOX_SIP_TOKEN=<secret>` (change `api1` in `extensions.conf` to `127.0.0.1` for both AudioSocket and the setup URL).

## Known limits
- **Caller id:** AudioSocket carries only the call UUID. The dialplan posts the caller id and the dialed number first (`caller`, plus `exten` or `callee`). Only digits and a leading `+` are forwarded (`+` is sent as `%2B`). Anything else is stripped before the request, so it cannot be run as a shell command. The token stays in `SIP_SETUP_TOKEN` and is not placed on the shell command line. Metadata `from` / `to` fill `{{caller}}` and `{{callee}}`. A setup older than 60 seconds is ignored for a new call. Hang-up, DELETE, and retention remove the setup row. An empty `VOX_SIP_TOKEN` leaves the endpoint off.
- **One engine node per Asterisk:** the dialplan points at `api1`. A TCP load balancer in front of several nodes is not built, and a call is lost if its node dies (the Twilio path can reconnect a call; the SIP path cannot).
- **Security:** the AudioSocket port has no authentication. Keep it on a private network and never publish it.
- **Outbound calls over SIP** (through VoxSun's `calls/initiate`) are not built.
- **Open ports for Asterisk:** UDP 5060 and the RTP range 10000-10100. Behind NAT, set the public address in `pjsip.conf`.
- Asterisk is GPL-2.0 software run as a separate program; this repo does not include or link it. Get the same licence check you would for any GPL component.
