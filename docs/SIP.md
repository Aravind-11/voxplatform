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
Asterisk (open source) does the SIP and audio-over-IP work. The engine receives each call's audio over a plain TCP connection
(`transports/audiosocket.py`) and runs the same agent, tools, interruption handling and storage as the Twilio path.
Calls are stored with medium `sip`.

## What is verified and what is not
| Part | State |
|---|---|
| AudioSocket transport: message format, 8 kHz audio in, paced real-time audio out, interruption (queue cleared), DTMF in, hangup both ways | **tested** (4 unit tests incl. a simulated Asterisk talking to the real app; no real Asterisk) |
| `VOX_AUDIOSOCKET_*` settings and the call lifecycle (create, admit, session, finish, webhook) | **tested** with the simulated Asterisk |
| Asterisk config (`deploy/sip/`), `deploy/Dockerfile.asterisk`, compose `asterisk` service (profile `sip`) | **written from Asterisk's documentation, never run**: compose file parses; no Asterisk was available here |
| Real call from VoxSun | **not done**: needs VoxSun credentials and a test number |

## Set it up (once VoxSun gives SIP details)
1. Fill the uppercase values in `deploy/sip/pjsip.conf` (server, username, password or the provider's IP).
2. Create the agent that should answer, then in `deploy/.env` set `VOX_AUDIOSOCKET_PORT=9092` and `VOX_SIP_AGENT=<agent id>`.
3. `docker compose -f deploy/docker-compose.yml --env-file deploy/.env --profile sip up -d`
4. Check: `docker compose exec asterisk asterisk -rx "pjsip show registrations"` and `... "module show like audiosocket"`.
5. Call the VoxSun number. In the engine logs you should see a new `sip` call.

Without Docker: install Asterisk 18+ (needs the AudioSocket modules), copy `deploy/sip/` into `/etc/asterisk/` (`pjsip.conf` includes `pjsip.register.conf`), and run the engine with `VOX_AUDIOSOCKET_PORT=9092 VOX_AUDIOSOCKET_AGENT=<id> VOX_AUDIOSOCKET_HOST=127.0.0.1` (change `api1:9092` in `extensions.conf` to `127.0.0.1:9092`).

## Known limits
- **Caller number is not passed:** AudioSocket carries only a call UUID. The agent does not know who is calling until we add it (Asterisk can pass it another way; not built).
- **One engine node per Asterisk:** `extensions.conf` dials `api1`. Compose service `asterisk2` mounts `extensions.api2.conf` (dials `api2`) and does not load the trunk registration. Continuing a call whose node died is below.
- **Security:** the AudioSocket port has no authentication. Keep it on a private network and never publish it.
- **Outbound calls over SIP** (through VoxSun's `calls/initiate`) are not built.
- **Open ports for Asterisk:** UDP 5060 and the RTP range 10000-10100. Behind NAT, set the public address in `pjsip.conf`.
- Asterisk is GPL-2.0 software run as a separate program; this repo does not include or link it. Get the same licence check you would for any GPL component.

## More than one Asterisk
One Asterisk is a single point of failure. The intended layout is carrier failover, or Kamailio (`deploy/sip/kamailio.cfg`, not run in CI) in front of the compose services `asterisk` and `asterisk2` (both profile `sip`), plus a TCP load balancer in front of AudioSocket. Kamailio pins each INVITE by the last character of the Call-ID local-part (the character before `@`), not the host. Only `asterisk` registers to the carrier; `asterisk2` is inbound from that proxy and sends AudioSocket to `api2`. Each API process still applies its own concurrency cap before it admits a call, so a dead node stops taking new sockets once the balancer stops using it.

If the AudioSocket stream ends and the engine did not hang up (`ended_by_us` is false), `finish` already stores `end_reason=stream_closed`. A re-dial does not know the old call id. Whoever still has the AudioSocket UUID (the dialplan's `CALLUUID`, or the proxy) POSTs it as `uuid` to `/sip/recover` with header `X-Vox-Sip-Token` (`VOX_SIP_TOKEN`). The handler finds the call by metadata `callSid` and, when the agent did not hang up, creates a new SIP call with `resumedFrom` set so the next session keeps the transcript. An empty token leaves the endpoint off. An agent hang-up is not resumed.
