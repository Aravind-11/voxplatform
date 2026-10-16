#!/usr/bin/env bash
# One command from "vLLM is up" to "call the number". Run on the GPU box.
#   bash scripts/04_twilio_up.sh
# Asks for the Twilio Auth Token with hidden input (it is never printed or stored on disk).
# Prints the exact URL to paste into the Twilio console.
set -euo pipefail
FS="${FS:-/lambda/nfs/arvi}"
cd "$(dirname "$0")/.."
ROOT="$PWD"

curl -fsS http://127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM is not answering on :8000. Start it first."; exit 1; }

read -r -s -p "Twilio Auth Token (hidden, not saved): " TWILIO_AUTH_TOKEN; echo
[ -n "$TWILIO_AUTH_TOKEN" ] || { echo "empty token"; exit 1; }

# 1. cloudflared (instance disk is wiped on termination, so reinstall when missing)
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared && sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
fi

# 2. public HTTPS tunnel (new URL on every launch)
tmux kill-session -t tunnel 2>/dev/null || true
rm -f /tmp/tunnel.log
tmux new -d -s tunnel "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080 2>&1 | tee /tmp/tunnel.log"
URL=""
for _ in $(seq 1 40); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/tunnel.log | head -1 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -n "$URL" ] || { echo "no tunnel URL after 40 s; see /tmp/tunnel.log"; exit 1; }

# 3. API with phone-tuned end-of-turn detector (Ultravox yes/no verdict, threshold from eot_eval --phone)
ADMIN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
tmux kill-session -t api 2>/dev/null || true
tmux new -d -s api "cd $ROOT && source $HOME/venv/bin/activate && \
  export HF_HOME=$FS/hf HF_HUB_OFFLINE=1 VOX_DB=$FS/vox.db VOX_PUBLIC_URL=$URL \
         VOX_ADMIN_KEY=$ADMIN TWILIO_AUTH_TOKEN='$TWILIO_AUTH_TOKEN' \
         VOX_DEFAULT_CONCURRENCY=64 VOX_LOG_EVENTS=1 VOX_EOT=model VOX_EOT_THRESHOLD=0.9 && \
  uvicorn api.main:app --host 127.0.0.1 --port 8080"
for _ in $(seq 1 90); do curl -fsS http://127.0.0.1:8080/healthz >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:8080/healthz >/dev/null || { echo "API did not start: tmux attach -t api"; exit 1; }

# 4. tenant + agent (Utah has its own database, so the old key does not exist here)
T=$(curl -fsS -X POST http://127.0.0.1:8080/admin/tenants -H "X-Admin-Key: $ADMIN" \
     -H 'content-type: application/json' -d '{"name":"demo","concurrencyLimit":64}')
KEY=$(echo "$T" | python3 -c 'import sys,json;print(json.load(sys.stdin)["apiKey"])')
A=$(curl -fsS -X POST http://127.0.0.1:8080/v1/agents -H "Authorization: Bearer $KEY" \
     -H 'content-type: application/json' \
     -d '{"name":"receptionist","systemPrompt":"You are a friendly receptionist for a dental clinic. Keep answers to one or two short sentences. You can check availability and book a slot.","voice":"af_heart","tools":["check_availability","book_slot"]}')
AID=$(echo "$A" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

echo
echo "================ DONE ================"
echo "Twilio console -> Phone Numbers -> +1 737 232 4091 -> Voice configuration:"
echo "  A call comes in:  Webhook, HTTP POST"
echo "  URL:              $URL/twilio/voice/$AID"
echo
echo "Trial accounts only ring verified numbers: add your cell under Verified Caller IDs."
echo "Then call +1 737 232 4091 from that cell."
echo "Watch it live:   tmux capture-pane -p -S -300 -t api | grep -E 'EV|ERROR|Traceback|twilio' | tail -60"
