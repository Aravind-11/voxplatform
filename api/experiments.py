"""Per-call experiment assignment. The bucket is a sha256 of the call id, not Python's hash(),
which is salted per process and would send the same call to different variants after a restart.
"""
from __future__ import annotations

import hashlib
import math
import types
from typing import Any


class ExperimentError(ValueError):
    pass


def validate_experiment(experiment: Any) -> dict:
    """Weights must be positive, variant ids unique, at most 4 variants. Raises ExperimentError."""
    if not isinstance(experiment, dict):
        raise ExperimentError("experiment must be an object")
    extra = set(experiment) - {"id", "variants"}
    if extra:
        raise ExperimentError(f"unknown experiment field {sorted(extra)[0]!r}")
    eid = experiment.get("id")
    if not isinstance(eid, str) or not eid.strip():
        raise ExperimentError("experiment.id must be a non-empty string")
    variants = experiment.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ExperimentError("experiment.variants must be a non-empty list")
    if len(variants) > 4:
        raise ExperimentError("an experiment has at most 4 variants")
    seen: list[str] = []
    cleaned: list[dict] = []
    for raw in variants:
        if not isinstance(raw, dict):
            raise ExperimentError("each variant must be an object")
        unknown = set(raw) - {"id", "weight", "voice", "systemPrompt", "model"}
        if unknown:
            raise ExperimentError(f"unknown variant field {sorted(unknown)[0]!r}")
        vid = raw.get("id")
        if not isinstance(vid, str) or not vid.strip():
            raise ExperimentError("variant id must be a non-empty string")
        if vid in seen:
            raise ExperimentError(f"duplicate variant id {vid!r}")
        seen.append(vid)
        weight = raw.get("weight")
        # bool is a subclass of int; True would otherwise pass as weight 1.
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ExperimentError("variant weight must be a positive number")
        item: dict[str, Any] = {"id": vid, "weight": weight if isinstance(weight, int) else float(weight)}
        for key in ("voice", "systemPrompt", "model"):
            if key not in raw or raw[key] is None:
                continue
            if not isinstance(raw[key], str):
                raise ExperimentError(f"variant {key} must be a string")
            if raw[key].strip():
                item[key] = raw[key]
        cleaned.append(item)
    return {"id": eid.strip(), "variants": cleaned}


def choose_variant(experiment: dict, call_id: str) -> dict:
    """Stable bucket: same call id always returns the same variant. Weights are a line of length sum(weight)."""
    variants = list(experiment.get("variants") or [])
    if not variants:
        raise ExperimentError("experiment has no variants")
    weights: list[float] = []
    for variant in variants:
        weight = variant.get("weight", 0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ExperimentError("variant weight must be a positive number")
        weights.append(float(weight))
    total = sum(weights)
    slot = int.from_bytes(hashlib.sha256(call_id.encode("utf-8")).digest()[:8], "big")
    point = (slot / float(2**64)) * total
    acc = 0.0
    for variant, weight in zip(variants, weights):
        acc += weight
        if point < acc:
            return dict(variant)
    return dict(variants[-1])


class VariantLLM:
    """One call's model name, delegating everything else to the shared client.

    VllmClient methods read `self.model`. Rebinding them onto this wrapper makes the variant model
    the one that call sends, without writing the shared client's `model` attribute.
    """

    def __init__(self, inner: object, model: str):
        self._inner = inner
        self.model = model

    def __getattr__(self, name: str):
        func = getattr(type(self._inner), name, None)
        if isinstance(func, types.FunctionType):
            return func.__get__(self, type(self))
        return getattr(self._inner, name)
