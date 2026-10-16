#!/usr/bin/env bash
# One command: restart vLLM with tool calling ON, wait for it, then run every real-model check and save the
# output to /lambda/nfs/arvi/results/verify-HHMM.txt.   Takes ~10-15 min (most of it vLLM loading).
#   bash scripts/05_verify.sh
set -uo pipefail
FS="${FS:-/lambda/nfs/arvi}"
cd "$(dirname "$0")/.."
ROOT="$PWD"
source "$HOME/venv/bin/activate"
export HF_HOME="$FS/hf" HF_HUB_OFFLINE=1
mkdir -p "$FS/results"
OUT="$FS/results/verify-$(date +%H%M).txt"
exec > >(tee "$OUT") 2>&1

echo "== restarting vLLM with tool calling on"
tmux kill-session -t vllm 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
for _ in $(seq 1 30); do                                     # wait until the GPU memory is actually free
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-99999}" -lt 3000 ] && break; sleep 2
done
tmux new -d -s vllm "cd $ROOT && TOOLS=1 bash scripts/01_serve_vllm.sh"
up=0
for i in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { up=1; break; }; sleep 10
done
[ "$up" = 1 ] || { echo "vLLM did not come up in 20 min: run 'tmux attach -t vllm' and send me the last 30 lines"; exit 1; }
echo "vLLM is up after about $((i * 10)) s"

echo; echo "== 1. tool calling with real audio (clean and phone-degraded)"
python tests/bringup/tools_check.py 2>/dev/null

echo; echo "== 2. end-of-turn: original prompt (v1) vs filler-aware prompt (v2), model verdict only"
for p in v1 v2; do
  for ch in "" "--phone --snr 20"; do
    # shellcheck disable=SC2086
    python tests/bringup/eot_eval.py --prompt "$p" $ch 2>/dev/null | awk '/^channel:/{f=1} /=== PROSODY/{f=0} f'
  done
done
echo; echo "== done. Saved to $OUT"
