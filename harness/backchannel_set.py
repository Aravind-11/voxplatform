"""Score a labelled backchannel set.

Rows are `{text, duration_ms, label}` where `label` is true when the clip is a backchannel.
They can be built by hand or from calls recorded with `VOX_DATASET=1`; nothing is read from disk here.
"""
from __future__ import annotations

from core.backchannel import is_backchannel


def score_set(rows: list[dict]) -> dict:
    correct = 0
    for row in rows:
        pred = is_backchannel(str(row.get("text", "")), float(row["duration_ms"]))
        if pred == bool(row["label"]):
            correct += 1
    return {"correct": correct, "total": len(rows)}
