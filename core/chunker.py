"""Streaming text -> speakable chunks for TTS.

Goals: start speaking as early as possible (short first chunk, split at a clause), keep
later chunks sentence-sized for natural prosody, never read markup aloud, and drop
Qwen-style <think>...</think> blocks.
"""
from __future__ import annotations

import re

_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "inc", "ltd"}
_END = ".!?…\n"
_CLAUSE = ",;:—"
_MD = re.compile(r"[*_`#>]+")
_BULLET = re.compile(r"(?m)^\s*(?:[-+]|\d+[.)])\s+")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    text = _BULLET.sub("", text)
    text = _MD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self, first_min_chars: int = 24):
        self.first_min = first_min_chars
        self._buf = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        self._buf = _THINK.sub("", self._buf)
        visible = self._buf
        withheld = ""
        if "<think>" in visible:  # unterminated reasoning block: hold it back
            i = visible.index("<think>")
            visible, withheld = visible[:i], visible[i:]
        out: list[str] = []
        while True:
            cut = self._find_boundary(visible)
            if cut is None:
                break
            piece, visible = visible[: cut + 1], visible[cut + 1 :].lstrip(" ")
            text = clean(piece)
            if text:
                out.append(text)
                self._emitted += 1
        self._buf = visible + withheld
        return out

    def flush(self) -> list[str]:
        rest = clean(_THINK.sub("", self._buf).split("<think>")[0])
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def flush_finished(self) -> list[str]:
        """Like flush(), but only when the held tail already ends a sentence.

        push() will not emit ".!?…" until another character arrives, and it also holds a
        final newline. flush() would speak an unfinished fragment; a model error must not."""
        visible = _THINK.sub("", self._buf).split("<think>", 1)[0]
        text = clean(visible)
        if text and text[-1] in ".!?…":
            return self.flush()
        if visible.rstrip(" \t").endswith("\n"):
            return self.flush()
        return []

    def _find_boundary(self, s: str) -> int | None:
        for i, ch in enumerate(s[:-1]):  # need a following char to confirm the boundary
            nxt = s[i + 1]
            if ch in _END and (nxt.isspace() or ch == "\n"):
                if ch == "." and self._is_abbrev_or_number(s, i):
                    continue
                return i
            if (self._emitted == 0 and ch in _CLAUSE and nxt.isspace()
                    and len(clean(s[: i + 1])) >= self.first_min):
                return i
        return None

    @staticmethod
    def _is_abbrev_or_number(s: str, i: int) -> bool:
        j = i
        while j > 0 and (s[j - 1].isalnum() or s[j - 1] == "."):
            j -= 1
        token = s[j:i].lower()
        return token in _ABBREV or (token.isdigit() and len(token) <= 2 and len(s) > i + 2 and s[i + 2].isdigit())
