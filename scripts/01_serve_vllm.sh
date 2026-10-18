#!/usr/bin/env bash
# Serve Ultravox v0.6 (Qwen3-32B backbone) with vLLM's OpenAI-compatible server.
# Binds to localhost only: never expose this port publicly. The public entry point
# is the reverse proxy on 443 in front of our own API.
#
# Tunables (env): QUANT=fp8|none  MAXLEN=8192  UTIL=0.60  PORT=8000
#                 TOOLS=1 (default; enables the tool-call parser, TOOLS=0 turns it off)  TOOL_PARSER=hermes
#                 OFFLINE=0 (allow Hub access if the offline cache is incomplete)
set -euo pipefail

FS="${FS:-/lambda/nfs/arvi}"
VENV="${VENV:-$HOME/venv}"
export HF_HOME="$FS/hf"
# Offline by default: if anything is missing from the cache, fail loudly instead of
# silently downloading during billed GPU time.
export HF_HUB_OFFLINE="${OFFLINE:-1}"
export HF_HUB_OFFLINE
# shellcheck disable=SC1091
source "$VENV/bin/activate"

MODEL="${MODEL:-fixie-ai/ultravox-v0_6-qwen-3-32b}"
QUANT="${QUANT:-fp8}"
MAXLEN="${MAXLEN:-8192}"
UTIL="${UTIL:-0.60}"   # leave headroom on the 80 GB card for TTS + VAD
PORT="${PORT:-8000}"

ARGS=(serve "$MODEL"
  --served-model-name ultravox
  --host 127.0.0.1 --port "$PORT"
  --trust-remote-code
  --dtype bfloat16
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$UTIL"
  --limit-mm-per-prompt '{"audio": 1}')  # one audio clip per request; VOX_ASR_URL transcribes earlier turns without a second 32B pass

[ "$QUANT" != "none" ] && ARGS+=(--quantization "$QUANT")
[ "${TOOLS:-1}" = "1" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-hermes}")

mkdir -p "$FS/logs"
LOG="$FS/logs/vllm-$(date +%Y%m%d-%H%M).log"
echo "Starting: vllm ${ARGS[*]}" | tee "$LOG"
vllm "${ARGS[@]}" 2>&1 | tee -a "$LOG"
