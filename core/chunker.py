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


def clean(text: str, *, at_line_start: bool = True) -> str:
    # A leading list marker is markup only on a source line. A mid-line cut
    # (word cap, comma) must keep tokens such as "1)", "1.", "-", and "+".
    if at_line_start:
        text = _BULLET.sub("", text)
    else:
        nl = text.find("\n")
        if nl != -1:
            text = text[: nl + 1] + _BULLET.sub("", text[nl + 1 :])
    text = _MD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self, first_min_chars: int = 24, first_max_words: int = 12):
        self.first_min = first_min_chars
        self.first_max_words = first_max_words
        self._buf = ""
        self._emitted = 0
        self._at_line_start = True

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        self._buf = _THINK.sub("", self._buf)
        visible = self._buf
        withheld = ""
        if "<think>" in visible:  # unterminated reasoning block: hold it back
            i = visible.index("<think>")
            visible, withheld = visible[:i], visible[i:]
        out: list[str] = []
        at_line_start = self._at_line_start
        while True:
            cut = self._find_boundary(visible, at_line_start)
            if cut is None:
                break
            piece = visible[: cut + 1]
            visible = visible[cut + 1 :].lstrip(" ")
            text = clean(piece, at_line_start=at_line_start)
            # The next slice is a source line only when this cut was the newline.
            at_line_start = piece.endswith("\n")
            if text:
                out.append(text)
                self._emitted += 1
        self._buf = visible + withheld
        self._at_line_start = at_line_start
        return out

    def flush(self) -> list[str]:
        rest = clean(_THINK.sub("", self._buf).split("<think>")[0], at_line_start=self._at_line_start)
        self._buf = ""
        self._at_line_start = True
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _find_boundary(self, s: str, at_line_start: bool = True) -> int | None:
        # Cut only once another word has started, so a still-growing token is not split.
        word_cut = self._first_word_cut(s) if self._emitted == 0 else None
        for i, ch in enumerate(s[:-1]):  # need a following char to confirm the boundary
            nxt = s[i + 1]
            if ch in _END and (nxt.isspace() or ch == "\n"):
                if ch == "." and self._is_abbrev_or_number(s, i):
                    continue
                return i
            if (self._emitted == 0 and ch in _CLAUSE and nxt.isspace()
                    and len(clean(s[: i + 1], at_line_start=at_line_start)) >= self.first_min):
                return i
            if word_cut is not None and i == word_cut:
                return i
        return None

    def _first_word_cut(self, s: str) -> int | None:
        """Index of the space after the first chunk's word cap, once a further word has started."""
        if self.first_max_words < 1:
            return None
        words = 0
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                return None
            while i < n and not s[i].isspace():
                i += 1
            words += 1
            if words == self.first_max_words:
                if i >= n or not s[i].isspace():
                    return None
                j = i
                while j < n and s[j].isspace():
                    j += 1
                if j >= n:
                    return None
                return i
        return None

    @staticmethod
    def _is_abbrev_or_number(s: str, i: int) -> bool:
        j = i
        while j > 0 and (s[j - 1].isalnum() or s[j - 1] == "."):
            j -= 1
        token = s[j:i].lower()
        return token in _ABBREV or (token.isdigit() and len(token) <= 2 and len(s) > i + 2 and s[i + 2].isdigit())
