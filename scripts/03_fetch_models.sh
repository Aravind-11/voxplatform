#!/usr/bin/env bash
# Downloads every model the platform needs into the filesystem cache. Safe to re-run: files already
# present are skipped. Use this on any region's fresh filesystem (name it "arvi" at launch).
set -euo pipefail
FS=/lambda/nfs/arvi
source "$HOME/venv/bin/activate"
export HF_HOME=$FS/hf
hf download fixie-ai/ultravox-v0_6-qwen-3-32b
hf download Qwen/Qwen3-32B
hf download openai/whisper-large-v3-turbo      # vLLM needs its feature-extractor config
hf download hexgrad/Kokoro-82M
du -sh $HF_HOME
