"""Small BM25 index for knowledge-base search (no dependencies). Documents are split into overlapping
passages so a long document returns the relevant part, not the whole thing."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be but by for from has have i if in is it its of on or that the this to was we what when where which who will with you your do does how can".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP]


def passages(text: str, size: int = 90, overlap: int = 20) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        """docs: [{"id","title","text"}]"""
        self.k1, self.b = k1, b
        self.items: list[dict] = []
        for d in docs:
            for i, p in enumerate(passages(d["text"])):
                self.items.append({"doc": d["id"], "title": d["title"], "passage": i, "text": p,
                                   "tf": Counter(tokens(d["title"] + " " + p)), })
        self.n = len(self.items)
        self.avg = (sum(sum(it["tf"].values()) for it in self.items) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for it in self.items:
            df.update(it["tf"].keys())
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def rank(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """[(passage index, score)] best first, only passages that share a word with the query."""
        q = tokens(query)
        if not q or not self.n:
            return []
        out = []
        for i, it in enumerate(self.items):
            dl = sum(it["tf"].values()) or 1
            s = 0.0
            for t in q:
                f = it["tf"].get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = tokens(query)
        if not q or not self.n:
            return []
        out = []
        for it in self.items:
            dl = sum(it["tf"].values()) or 1
            s = 0.0
            for t in q:
                f = it["tf"].get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            if s > 0:
                out.append((s, it))
        out.sort(key=lambda x: -x[0])
        return [{"title": it["title"], "text": it["text"], "score": round(s, 3)} for s, it in out[:max(1, min(top_k, 8))]]
