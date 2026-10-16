# Licenses

Goal: everything in the default stack is usable commercially, and you control the infrastructure.
Checked 2026-10-15 against each project's own page (model card, repo or PyPI). Re-check before shipping; licenses change.

| Component | Role | License | Commercial use | Notes |
|---|---|---|---|---|
| Ultravox v0.6 (`fixie-ai/ultravox-v0_6-qwen-3-32b`) | speech-in LLM | MIT (model card) | yes | built on Qwen3-32B and whisper-large-v3-turbo |
| Qwen3-32B | Ultravox language backbone | Apache-2.0 | yes | |
| whisper-large-v3-turbo | Ultravox audio encoder | MIT | yes | |
| Kokoro-82M | text-to-speech | Apache-2.0 | yes | |
| misaki | Kokoro's grapheme-to-phoneme library | Apache-2.0 | yes | espeak-ng is an optional fallback (below) |
| Silero VAD | voice activity detection | MIT | yes | |
| vLLM | model server | Apache-2.0 | yes | not re-checked in this pass |
| FastAPI / Starlette / uvicorn / httpx / pydantic | API | BSD/MIT | yes | not re-checked in this pass |
| **espeak-ng** | misaki fallback for out-of-dictionary words | **GPL-3.0** | see below | the one copyleft item |
| Twilio / Cloudflare | telephony / tunnel | commercial services | n/a | pay-per-use; not open source |

## The espeak-ng flag
misaki pronounces words from its own dictionary and, if installed, falls back to espeak-ng for words not in it.
espeak-ng is GPL-3.0. If you distribute software that links it, GPL obligations can apply to what you distribute.
Running it on your own servers behind a network service is a different case from distributing it, and some
readings differ, so **get a lawyer's view before you ship a product that bundles it**. This is not legal advice.

Two ways to avoid the question:
1. `VOX_NO_ESPEAK=1` makes the server use misaki's dictionary-only G2P (`fallback=None`), so no GPL code runs.
   Cost: rare names, codes and unusual words are pronounced worse. Do not install espeak-ng in the image.
2. Keep espeak-ng as a separate, optional install that the operator adds themselves.

The provided Dockerfile does not install espeak-ng unless you build with `--build-arg WITH_ESPEAK=1`.
Status: the switch is written; it has not been run on the GPU box, and pronunciation impact is unmeasured.

## Not verified
- The Hugging Face model cards can change; the MIT label on Ultravox v0.6 was read from its card, not from a license file.
- The "CC BY-NC" badge image text on the silero-vad PyPI page contradicts its MIT license field; the repo LICENSE file
  should be read directly before commercial use.
- Third-party Python dependencies were not audited one by one. Run `pip-licenses` on your final image.
