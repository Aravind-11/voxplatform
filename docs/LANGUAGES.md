# Languages

Set `"language"` (ISO code) and/or `"voice"` on an agent. The voice decides which Kokoro pipeline speaks; `language` adds a prompt instruction to reply in that language.
List what a server supports: `GET /v1/voices` (includes languages and availability).

| Language | Code | Default voice | Needs |
|---|---|---|---|
| English (US/UK) | en | af_heart | nothing extra |
| Spanish | es | ef_dora | espeak-ng (GPL-3.0) |
| French | fr | ff_siwis | espeak-ng (GPL-3.0) |
| Hindi | hi | hf_alpha | espeak-ng (GPL-3.0) |
| Italian | it | if_sara | espeak-ng (GPL-3.0) |
| Portuguese | pt | pf_dora | espeak-ng (GPL-3.0) |
| Japanese | ja | jf_alpha | `pyopenjtalk` package |
| Mandarin | zh | zf_xiaoxiao | `jieba` package |

## Status (honest)
- **Tested:** voice/language selection, availability checks, per-language pipeline loading logic (unit tests with stand-ins).
- **Not tested:** no non-English audio was ever generated or heard; no accuracy check of the model's understanding per language; the "resume" phrases for call recovery in 7 non-English languages are unreviewed machine-written text.
- **Licence:** espeak-ng is GPL-3.0. `VOX_NO_ESPEAK=1` (the default recommendation) disables es/fr/hi/it/pt. Read docs/LICENSES.md and get legal advice before enabling them in something you distribute.
- Preload voices at start with `VOX_PRELOAD_LANGS=en,es` to avoid a first-call delay.
- Have a native speaker review each language before real customers hear it.
