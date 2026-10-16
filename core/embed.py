"""Optional embedding search for knowledge bases: dense similarity fused with BM25 (reciprocal rank fusion).

Needs an OpenAI-compatible embeddings server (VOX_EMBED_URL, e.g. a small model served by vLLM; scripts/01b_serve_embed.sh).
Without one, knowledge bases use keyword search only. Passage vectors are cached in memory per process, so the first
question on a knowledge base pays for embedding it; if the server errors, search falls back to keywords."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

from .bm25 import BM25


class OpenAIEmbedder:
    def __init__(self, base_url: str, model: str, client: Any, api_key: str = "none", batch: int = 64, min_sim: float = 0.0):
        self.base_url, self.model, self.client, self.api_key, self.batch = base_url.rstrip("/"), model, client, api_key, batch
        self.min_sim = min_sim                      # drop dense matches below this cosine (model dependent: calibrate)
        self.cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            r = await self.client.post(f"{self.base_url}/embeddings", timeout=30,
                                       headers={"Authorization": f"Bearer {self.api_key}"},
                                       json={"model": self.model, "input": [t for _, t in chunk]})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            for (k, _), d in zip(chunk, data):
                self.cache[k] = d["embedding"]
            if len(self.cache) > 50_000:            # crude bound: forget everything rather than grow forever
                self.cache = {k: self.cache[k] for k in keys if k in self.cache}
        return [self.cache[k] for k in keys]


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


async def hybrid_search(bm: BM25, embedder, query: str, top_k: int = 3, rrf_k: int = 60) -> list[dict]:
    """Fuses BM25 and dense rankings. Raises if the embedder fails (callers fall back to BM25 alone)."""
    if not bm.n:
        return []
    vecs = await embedder.embed([it["text"] for it in bm.items] + [query])
    qv = vecs[-1]
    dense = sorted(((cosine(qv, v), i) for i, v in enumerate(vecs[:-1])), reverse=True)
    dense = [(s, i) for s, i in dense[:20] if s >= getattr(embedder, "min_sim", 0.0)]
    score: dict[int, float] = {}
    for rank, (_, i) in enumerate(dense):
        score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (i, _) in enumerate(bm.rank(query)):
        score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
    best = sorted(score.items(), key=lambda x: -x[1])[:max(1, min(top_k, 8))]
    return [{"title": bm.items[i]["title"], "text": bm.items[i]["text"], "score": round(s * 100, 3)} for i, s in best]
