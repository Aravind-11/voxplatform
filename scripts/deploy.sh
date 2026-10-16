#!/usr/bin/env bash
# One command from "vLLM is up" to "call the number" on the GPU box (replaces 04_twilio_up.sh).
#   bash scripts/deploy.sh
# Prompts for the Twilio Auth Token with hidden input; it is never printed or written to disk.
# Optional env: TWILIO_ACCOUNT_SID, VOX_TWILIO_NUMBER (+E.164)  -> also sets the number's webhook for you.
#               VOX_DB (default $FS/vox.db; a postgresql:// URL works), VOX_REDIS_URL, VOX_RECORD_DIR.
set -euo pipefail
FS="${FS:-/lambda/nfs/arvi}"
cd "$(dirname "$0")/.."
ROOT="$PWD"
DEFAULT_GREETING="Hi, this is Arvindh's voice agent. How can I help you today?"
GREETING="${GREETING:-$DEFAULT_GREETING}"

curl -fsS http://127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM is not answering on :8000. Start it first (scripts/01_serve_vllm.sh)."; exit 1; }

if [ -z "${TWILIO_AUTH_TOKEN:-}" ]; then
  read -r -s -p "Twilio Auth Token (hidden, not saved; Enter to skip phone support): " TWILIO_AUTH_TOKEN; echo
fi
if [ -n "$TWILIO_AUTH_TOKEN" ] && [ -z "${TWILIO_ACCOUNT_SID:-}" ]; then
  read -r -p "Twilio Account SID (starts with AC): " TWILIO_ACCOUNT_SID
fi

# 1. cloudflared (the instance disk is wiped on termination, so reinstall when missing)
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared && sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
fi

# 2. public HTTPS tunnel. Lambda images ship a cloudflared config for Jupyter that hijacks quick tunnels (404s),
#    so run with an empty environment and an empty config file.
tmux kill-session -t tunnel 2>/dev/null || true
rm -f /tmp/tunnel.log
tmux new -d -s tunnel "env -i HOME=/tmp PATH=\"$PATH\" cloudflared tunnel --no-autoupdate --config /dev/null --url http://127.0.0.1:8080 2>&1 | tee /tmp/tunnel.log"
URL=""
for _ in $(seq 1 40); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/tunnel.log | head -1 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -n "$URL" ] || { echo "no tunnel URL after 40 s; see /tmp/tunnel.log"; exit 1; }

# 3. API
ADMIN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
METRICS="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
REC="${VOX_RECORD_DIR:-$FS/recordings}"; mkdir -p "$REC"
tmux kill-session -t api 2>/dev/null || true
tmux new -d -s api "cd $ROOT && source $HOME/venv/bin/activate && \
  export HF_HOME=$FS/hf HF_HUB_OFFLINE=1 VOX_DB='${VOX_DB:-$FS/vox.db}' VOX_PUBLIC_URL=$URL VOX_ADMIN_KEY=$ADMIN \
         TWILIO_AUTH_TOKEN='$TWILIO_AUTH_TOKEN' TWILIO_ACCOUNT_SID='${TWILIO_ACCOUNT_SID:-}' \
         VOX_TWILIO_NUMBER='${VOX_TWILIO_NUMBER:-}' VOX_METRICS_TOKEN=$METRICS VOX_RECORD_DIR=$REC \
         VOX_REDIS_URL='${VOX_REDIS_URL:-}' VOX_DEFAULT_CONCURRENCY=64 VOX_LOG_EVENTS=1 VOX_EOT=model VOX_EOT_THRESHOLD=0.9 && \
  uvicorn api.main:app --host 127.0.0.1 --port 8080"
for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8080/healthz >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:8080/healthz >/dev/null || { echo "API did not start: tmux attach -t api"; exit 1; }

# 4. tenant + agent (auth header is X-API-Key)
J='content-type: application/json'
T=$(curl -fsS -X POST http://127.0.0.1:8080/admin/tenants -H "X-Admin-Key: $ADMIN" -H "$J" -d '{"name":"demo","concurrencyLimit":64}')
KEY=$(echo "$T" | python3 -c 'import sys,json;print(json.load(sys.stdin)["apiKey"])')
BODY=$(GREETING="$GREETING" python3 - <<'PY'
import json, os
print(json.dumps({"name": "receptionist", "voice": "af_heart", "greeting": os.environ["GREETING"],
  "systemPrompt": "You are a friendly receptionist for a dental clinic. Keep answers to one or two short sentences. You can check availability and book a slot. Always read the day and time back before booking.",
  "tools": ["check_availability", {"tool": "book_slot", "confirm": True}, "hangUp"]}))
PY
)
A=$(curl -fsS -X POST http://127.0.0.1:8080/v1/agents -H "X-API-Key: $KEY" -H "$J" -d "$BODY")
AID=$(echo "$A" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 5. point the phone number at the agent (needs Twilio SID + number); otherwise print the URL to paste
HOOK="$URL/twilio/voice/$AID"
if [ -n "${TWILIO_ACCOUNT_SID:-}" ] && [ -n "$TWILIO_AUTH_TOKEN" ] && [ -n "${VOX_TWILIO_NUMBER:-}" ]; then
  source "$HOME/venv/bin/activate"
  TWILIO_ACCOUNT_SID="$TWILIO_ACCOUNT_SID" TWILIO_AUTH_TOKEN="$TWILIO_AUTH_TOKEN" \
    python3 scripts/twilio_set_webhook.py "$VOX_TWILIO_NUMBER" "$HOOK" && WEBHOOK_SET=1 || WEBHOOK_SET=0
else
  WEBHOOK_SET=0
fi

echo
echo "================ DONE ================"
echo "Dashboard:   $URL/dashboard   (API key below)"
echo "Browser demo: $URL/demo"
echo "API key:     $KEY"
echo "Admin key:   $ADMIN      Metrics token: $METRICS"
echo "Agent id:    $AID"
if [ "$WEBHOOK_SET" = 1 ]; then
  echo "Phone:       $VOX_TWILIO_NUMBER now routes to the agent. Call it."
else
  echo "Phone:       Twilio console -> your number -> Voice -> 'A call comes in' -> Webhook, HTTP POST, URL:"
  echo "             $HOOK"
fi
echo "Logs:        tmux capture-pane -p -S -300 -t api | grep -E 'EV|ERROR|Traceback|twilio' | tail -60"
