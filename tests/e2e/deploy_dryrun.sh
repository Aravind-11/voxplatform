#!/usr/bin/env bash
# Runs scripts/deploy.sh for real with shims for vLLM, cloudflared and uvicorn (fake model server), then checks the
# result through the API. Proves the script's own logic (tunnel URL parsing, env wiring, tenant/agent creation, output).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W=$(mktemp -d); mkdir -p "$W/bin" "$W/home/venv/bin" "$W/fs" "$W/vllm/v1"; touch "$W/home/venv/bin/activate" "$W/vllm/v1/models"
cat > "$W/bin/cloudflared" <<'S'
#!/usr/bin/env bash
[ "$1" = tunnel ] || exit 1
echo "INF +--------------+"; echo "https://dry-run-abc.trycloudflare.com"; exec sleep 600
S
cat > "$W/bin/uvicorn" <<S
#!/usr/bin/env bash
exec python3 "$ROOT/tests/e2e/serve_fake.py" 8080
S
chmod +x "$W/bin/"*
(cd "$W/vllm" && python3 -m http.server 8000 --bind 127.0.0.1 >/dev/null 2>&1) & VP=$!
cleanup() { kill $VP 2>/dev/null; tmux kill-session -t tunnel 2>/dev/null; tmux kill-session -t api 2>/dev/null; true; }
trap cleanup EXIT
sleep 1
OUT=$(PATH="$W/bin:$PATH" HOME="$W/home" FS="$W/fs" TWILIO_AUTH_TOKEN=tok TWILIO_ACCOUNT_SID=ACx bash "$ROOT/scripts/deploy.sh")
echo "$OUT" | tail -12
KEY=$(echo "$OUT" | sed -n 's/^API key: *//p')
AID=$(echo "$OUT" | sed -n 's/^Agent id: *//p')
URL=$(echo "$OUT" | sed -n 's/^Dashboard: *\(https[^ ]*\)\/dashboard.*/\1/p')
[ "$URL" = "https://dry-run-abc.trycloudflare.com" ] || { echo "FAIL tunnel url: $URL"; exit 1; }
echo "$OUT" | grep -q "/twilio/voice/$AID" || { echo "FAIL webhook line"; exit 1; }
curl -fsS http://127.0.0.1:8080/v1/agents -H "X-API-Key: $KEY" | python3 -c '
import sys, json
a = json.load(sys.stdin)[0]["config"]
assert a["greeting"].startswith("Hi, this is Arvindh"), a
assert a["tools"] == ["check_availability", {"tool": "book_slot", "confirm": True}, "hangUp"], a["tools"]
print("agent config OK")'
curl -fsS http://127.0.0.1:8080/dashboard | grep -q "Voice platform dashboard"
echo "deploy dry-run OK"
