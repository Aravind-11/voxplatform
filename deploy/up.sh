#!/usr/bin/env bash
# One command to bring up the whole production stack with Docker Compose on a GPU host.
#   bash deploy/up.sh voice.example.com
# First run: creates deploy/.env and deploy/secrets/ with random passwords/keys, asks for the Twilio token with hidden
# input (Enter to skip phone support), builds and starts every service, creates a first account and agent and prints
# the keys once. Re-running keeps your .env and secrets and just updates the stack.
set -euo pipefail
cd "$(dirname "$0")"
DOMAIN="${1:-${VOX_DOMAIN:-}}"
command -v docker >/dev/null || { echo "Docker is not installed."; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required."; exit 1; }
rand() { python3 -c 'import secrets;print(secrets.token_urlsafe(24))'; }
mkdir -p secrets

if [ ! -f .env ]; then
  [ -n "$DOMAIN" ] || { read -r -p "Domain name pointing at this host (e.g. voice.example.com): " DOMAIN; }
  [ -n "$DOMAIN" ] || { echo "A domain is required (Twilio needs https)."; exit 1; }
  cp env.example .env
  sed -i "s|^VOX_DOMAIN=.*|VOX_DOMAIN=$DOMAIN|; s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(rand)|; s|^VOX_ADMIN_KEY=.*|VOX_ADMIN_KEY=$(rand)|; s|^VOX_METRICS_TOKEN=.*|VOX_METRICS_TOKEN=$(rand)|" .env
  read -r -s -p "Twilio Auth Token (hidden, Enter to skip phone support): " TT; echo
  if [ -n "$TT" ]; then
    read -r -p "Twilio Account SID (AC...): " SID
    read -r -p "Your Twilio phone number (+1...): " NUM
    sed -i "s|^TWILIO_AUTH_TOKEN=.*|TWILIO_AUTH_TOKEN=$TT|; s|^TWILIO_ACCOUNT_SID=.*|TWILIO_ACCOUNT_SID=$SID|; s|^VOX_TWILIO_NUMBER=.*|VOX_TWILIO_NUMBER=$NUM|" .env
  fi
  chmod 600 .env
fi
set -a; . ./.env; set +a
[ -f secrets/vox_metrics_token ] || printf '%s' "$VOX_METRICS_TOKEN" > secrets/vox_metrics_token
[ -f secrets/grafana_admin_password ] || rand > secrets/grafana_admin_password
if [ ! -f secrets/alert_webhook_url ]; then
  read -r -p "Alert webhook URL (Slack-compatible or any HTTPS endpoint; Enter to skip): " AW
  printf '%s' "${AW:-http://127.0.0.1:9/alerts-not-configured}" > secrets/alert_webhook_url
fi
chmod 600 secrets/*

docker compose config -q
docker compose up -d --build
echo "Waiting for the first API node (the model takes a few minutes to load on first start)..."
for _ in $(seq 1 240); do
  docker compose exec -T api1 python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" >/dev/null 2>&1 && break
  sleep 5
done
docker compose exec -T api1 python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" >/dev/null \
  || { echo "API did not become healthy: docker compose logs api1 vllm"; exit 1; }

if [ ! -f secrets/.bootstrapped ]; then
  OUT=$(docker compose exec -T -e VOX_ADMIN_KEY="$VOX_ADMIN_KEY" api1 python /app/bootstrap_tenant.py http://127.0.0.1:8080)
  touch secrets/.bootstrapped
  KEY=$(echo "$OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["apiKey"])')
  AID=$(echo "$OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["agentId"])')
  if [ -n "${TWILIO_AUTH_TOKEN:-}" ] && [ -n "${VOX_TWILIO_NUMBER:-}" ]; then
    docker compose exec -T -e TWILIO_ACCOUNT_SID="$TWILIO_ACCOUNT_SID" -e TWILIO_AUTH_TOKEN="$TWILIO_AUTH_TOKEN" api1 \
      python /app/twilio_set_webhook.py "$VOX_TWILIO_NUMBER" "https://$VOX_DOMAIN/twilio/voice/$AID" && PH="$VOX_TWILIO_NUMBER now routes to the agent. Call it."
  fi
  echo
  echo "================ DONE ================"
  echo "Dashboard:    https://$VOX_DOMAIN/dashboard"
  echo "Browser demo: https://$VOX_DOMAIN/demo"
  echo "API key:      $KEY        <- save it now: only a hash is stored"
  echo "Agent id:     $AID"
  echo "Admin key:    $VOX_ADMIN_KEY   (also in deploy/.env)"
  [ -n "${PH:-}" ] || PH="in Twilio, set the Voice webhook (HTTP POST) of your number to https://$VOX_DOMAIN/twilio/voice/$AID"
  echo "Phone:        $PH"
  echo "Grafana:      ssh -L 3000:localhost:3000 <this host>  then http://localhost:3000  (user admin, password in deploy/secrets/grafana_admin_password)"
else
  echo "Stack updated. Dashboard: https://$VOX_DOMAIN/dashboard"
fi
