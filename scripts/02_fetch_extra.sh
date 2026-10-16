#!/usr/bin/env bash
# Run once with network on the GPU box: installs the voice-platform deps and caches Kokoro weights.
set -euo pipefail
FS=/lambda/nfs/arvi
source "$HOME/venv/bin/activate"
export HF_HOME=$FS/hf
sudo apt-get install -y espeak-ng
uv pip install scipy fastapi "uvicorn[standard]" websockets pydantic kokoro silero-vad
hf download hexgrad/Kokoro-82M
python - <<'PY'
from silero_vad import load_silero_vad
load_silero_vad(); print("silero ok")
PY
pip freeze > $FS/logs/freeze-extra-$(date +%s).txt
