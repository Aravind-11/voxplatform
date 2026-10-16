# voxplatform

Self-hosted voice-agent platform on open-weight Ultravox (v0.6 / Qwen3-32B), served by vLLM.
Status of every component is tracked in `docs/STATUS.md` as **verified**, **written (untested)** or **designed**.

## Step 0: bring-up on the H100 (what to run, in order)

Prereqs: Lambda 1x H100 in Georgia (us-southeast-1), filesystem `arvi` attached, SSH key `arvi.pem`.
The model cache already lives at `/lambda/nfs/arvi/hf` (Qwen/Qwen3-32B + fixie-ai/ultravox-v0_6-qwen-3-32b).

```bash
# from your Mac: copy the repo to the persistent filesystem (use the new instance IP)
scp -i ~/Downloads/arvi.pem voxplatform.tar.gz ubuntu@<IP>:/lambda/nfs/arvi/
ssh -i ~/Downloads/arvi.pem ubuntu@<IP>

# on the instance
cd /lambda/nfs/arvi && tar xzf voxplatform.tar.gz && cd voxplatform
bash scripts/00_setup.sh            # venv on local disk + vLLM; prints versions

tmux new -s serve
bash scripts/01_serve_vllm.sh       # wait for "Application startup complete"
# detach: Ctrl-b then d

# in a second shell / tmux window
source ~/venv/bin/activate
python tests/bringup/bringup_test.py
```

Send back: the full terminal output of `01_serve_vllm.sh` (last ~60 lines, or any error) and of `bringup_test.py`.

### If the server fails to start
- Cache miss with offline mode: re-run with `OFFLINE=0 bash scripts/01_serve_vllm.sh`.
- Out of memory: `UTIL=0.70 MAXLEN=4096 bash scripts/01_serve_vllm.sh`, or `QUANT=none` to rule out fp8.
- fp8 + Ultravox incompatibility: `QUANT=none UTIL=0.90 MAXLEN=4096` (bf16 weights ~65 GB, little KV room).
- Unknown-architecture / audio errors: paste the traceback; the repo's `tools/update_vllm_impl.py` patch path is the fallback.

## Layout
- `scripts/` instance setup and server launch
- `tests/bringup/` model-server bring-up and concurrency probe
- `core/ transports/ api/ harness/ deploy/` the platform (next)
- `docs/` parity spec, status table, findings
