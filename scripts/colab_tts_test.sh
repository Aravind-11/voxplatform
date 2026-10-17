#!/usr/bin/env bash
# Colab (or any small-GPU box): measures whether running several Kokoro speech processes raises text-to-speech throughput.
# This is the biggest measured bottleneck (docs/ASSESSMENT.md section 6). It needs NO Ultravox model, so a small GPU is enough.
# In a Colab notebook cell:   !git clone https://github.com/Aravind-11/voxplatform && bash voxplatform/scripts/colab_tts_test.sh
# Keep the tab open and the cell running: an idle Colab runtime disconnects.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q kokoro soundfile numpy scipy
apt-get install -y -qq espeak-ng >/dev/null 2>&1 || echo "(no espeak-ng: English still works via misaki's dictionary)"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU visible: results are CPU-only"
python tests/bringup/tts_procs.py 2>&1 | tee colab_tts_result.txt
echo "Result saved in colab_tts_result.txt. Compare with docs/FINDINGS.md (one process: about 11 sentences/s)."
