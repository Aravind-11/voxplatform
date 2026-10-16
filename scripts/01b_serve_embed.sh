#!/usr/bin/env bash
# Small embeddings server for hybrid knowledge-base search (optional). WRITTEN, NOT RUN: vLLM's flag for embedding
# models has changed between versions (--task embed, or --runner pooling); run `vllm serve --help | grep -i -E 'task|runner'`.
# Model: BAAI/bge-small-en-v1.5 (MIT per its model card; verify before commercial use). Uses ~1-2 GB of the card.
set -euo pipefail
FS="${FS:-/lambda/nfs/arvi}"; VENV="${VENV:-$HOME/venv}"
export HF_HOME="$FS/hf"; source "$VENV/bin/activate"
MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
vllm serve "$MODEL" --task embed --served-model-name "$MODEL" --host 127.0.0.1 --port "${EMBED_PORT:-8001}" \
  --gpu-memory-utilization "${EMBED_UTIL:-0.04}" 2>&1 | tee -a "$FS/logs/embed-$(date +%Y%m%d-%H%M).log"
# then start the API with:  VOX_EMBED_URL=http://127.0.0.1:8001/v1
