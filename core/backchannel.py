"""Short utterances that overlap the agent but are not real barge-ins.

A backchannel ("yeah", "mm-hm") is both short and lexically empty of a new request.
Duration is the gate: anything longer than `max_ms` is a real interruption even if the
words are in the set. The turn manager uses the same idea before a transcript exists.
"""
from __future__ import annotations

import re

# Per-agent barge-in sensitivity -> milliseconds of caller speech required while the agent talks.
# "high" matches the old start_ms_agent_speaking gate (192 ms).
BACKCHANNEL_MS = {"high": 192.0, "normal": 450.0, "low": 700.0}

_PHRASES = frozenset({
    "mm-hm", "mm", "mhm", "uh-huh", "uh huh", "yeah", "yep", "yes",
    "okay", "ok", "right", "sure", "sí", "si", "oui", "haan",
})

_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = _PUNCT.sub("", text.strip().lower())
    return _SPACE.sub(" ", t).strip()


def is_backchannel(text: str, duration_ms: float, max_ms: float = 450) -> bool:
    """True when a short overlap is a backchannel, not a request to take the floor.

    The empty string counts only under 320 ms (a cough / clipped noise with no words).
    """
    if duration_ms > max_ms:
        return False
    norm = _normalize(text)
    if norm in _PHRASES:
        return True
    return norm == "" and duration_ms < 320
