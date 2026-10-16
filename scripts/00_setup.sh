#!/usr/bin/env bash
# One-time setup per instance launch (Lambda H100, Ubuntu 24.04).
# The venv lives on LOCAL disk (fast; thousands of small files).
# The model cache lives on the PERSISTENT filesystem ($FS/hf) so it survives termination.
set -euo pipefail

FS="${FS:-/lambda/nfs/arvi}"
VENV="${VENV:-$HOME/venv}"
export HF_HOME="$FS/hf"

echo "== sanity: filesystem + cache =="
test -d "$FS" || { echo "ERROR: $FS not mounted. Attach the 'arvi' filesystem at launch."; exit 1; }
mkdir -p "$FS/logs" "$FS/assets" "$FS/results"
ls "$HF_HOME/hub" 2>/dev/null || echo "WARNING: no model cache found under $HF_HOME/hub (run the hf download commands)"
du -sh "$HF_HOME" 2>/dev/null || true

echo "== sanity: GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

command -v tmux >/dev/null || sudo apt-get install -y tmux

echo "== python venv + vllm =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip uv
uv pip install vllm openai soundfile numpy httpx

echo "== versions =="
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok:', torch.cuda.is_available())"
pip freeze > "$FS/logs/freeze-$(date +%Y%m%d-%H%M).txt"
echo "== setup done =="
