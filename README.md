# voxplatform

A self-hosted AI voice-agent platform on the open-source Ultravox stack: real-time speech-to-speech conversation with
interruptions, phone calls (inbound and outbound), tools, knowledge bases, multi-tenant API, recordings, dashboard, Postgres/Redis
scale-out. Open components only (docs/LICENSES.md); you run all of it.

**Read docs/FINDINGS.md first**: what is proven, what is not, where it beats the paid service on cost and where it does not.
docs/STATUS.md labels every component *verified*, *written* or *designed*.

## Run it on a GPU box (about 10 minutes after the model is cached)
```bash
bash scripts/00_setup.sh                  # once per instance: venv + vLLM
pip install -r requirements.txt           # in that venv
tmux new -s serve; bash scripts/01_serve_vllm.sh        # wait for "Application startup complete", detach with Ctrl-b d
bash scripts/deploy.sh                    # tunnel + API + tenant + agent; prints dashboard, demo and phone URLs
# optional: TWILIO_ACCOUNT_SID=AC... VOX_TWILIO_NUMBER=+1... bash scripts/deploy.sh   (also points your number at the agent)
```
Open the printed `/demo` URL to talk from a browser, `/dashboard` to manage agents, or call your number.

## Run it with Docker (written, not yet run: see docs/STATUS.md)
`cp deploy/env.example deploy/.env`, edit, then `docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build`.
Kubernetes reference manifests: `deploy/k8s/`.

## Use the API (header `X-API-Key`)
```bash
curl -X POST $URL/v1/agents -H "X-API-Key: $KEY" -H 'content-type: application/json' -d '{
  "name":"reception","greeting":"Hi, how can I help?","systemPrompt":"You are brief and friendly.",
  "tools":["hangUp",{"builtin":"queryCorpus","corpus":"faq"}]}'
curl -X POST $URL/v1/calls -H "X-API-Key: $KEY" -d '{"agentId":"agt_..."}'            # returns a WebSocket joinUrl
curl -X POST $URL/v1/calls/outbound -H "X-API-Key: $KEY" -d '{"agentId":"agt_...","to":"+15551234567"}'
```
Endpoints: agents, calls (+ messages, recording, delete), outbound, keys, audit, tools library, corpora (knowledge bases), usage (+ CSV),
voices, `/metrics`, `/healthz`, `/demo`, `/dashboard`. Tools: docs/TOOLS.md. Tuning: docs/TUNING.md. Operations: docs/OPERATIONS.md.

## Test it (no GPU needed)
```bash
pip install -r requirements-dev.txt
python -m pytest tests/unit                                     # 195 tests, ~20 s
VOX_TEST_DB=postgresql://user@host/db python -m pytest tests/unit      # same suite on PostgreSQL
python tests/e2e/browser_test.py                                # dashboard + demo in headless Chromium, fake model
bash tests/e2e/deploy_dryrun.sh                                 # runs scripts/deploy.sh against stand-ins
```
On the GPU box: `bash scripts/05_verify.sh` (tool-calling and end-of-turn checks), `python tests/bringup/tts_procs.py` (TTS scaling).

## Layout
`core/` conversation engine (VAD, turn-taking, session, TTS, recorder, metrics, BM25) · `transports/` Twilio and browser ·
`api/` HTTP/WebSocket API, store, tools, dashboard pages · `scripts/` GPU-box setup and deploy · `deploy/` Docker, compose, Kubernetes ·
`tests/` unit, e2e, bring-up · `docs/` findings, status, licenses, tools, tuning, operations.
