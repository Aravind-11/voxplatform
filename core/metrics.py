"""Minimal Prometheus text-format metrics (counters, gauges from callbacks, histograms). No dependencies."""
from __future__ import annotations

import threading
from typing import Callable

LAT_BUCKETS = (50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000)   # milliseconds


def _fmt(labels: tuple) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels) + "}"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.help: dict[str, str] = {}
        self.counters: dict[tuple, float] = {}
        self.hists: dict[tuple, list] = {}          # (name, labels) -> [bucket counts..., sum, count]
        self.buckets: dict[str, tuple] = {}
        self.gauges: dict[str, Callable[[], float]] = {}

    def describe(self, name: str, text: str) -> None:
        self.help[name] = text

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, buckets: tuple = LAT_BUCKETS, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.buckets.setdefault(name, buckets)
            h = self.hists.setdefault(key, [0] * len(buckets) + [0.0, 0])
            for i, b in enumerate(buckets):
                if value <= b:
                    h[i] += 1
            h[-2] += value
            h[-1] += 1

    def gauge(self, name: str, fn: Callable[[], float]) -> None:
        self.gauges[name] = fn

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            names = sorted({k[0] for k in self.counters})
            for n in names:
                lines += [f"# HELP {n} {self.help.get(n, n)}", f"# TYPE {n} counter"]
                lines += [f"{n}{_fmt(l)} {v}" for (nn, l), v in sorted(self.counters.items()) if nn == n]
            for n in sorted({k[0] for k in self.hists}):
                lines += [f"# HELP {n} {self.help.get(n, n)}", f"# TYPE {n} histogram"]
                bk = self.buckets[n]
                for (nn, l), h in sorted(self.hists.items()):
                    if nn != n:
                        continue
                    for i, b in enumerate(bk):
                        lines.append(f"{n}_bucket{_fmt(l + (('le', b),))} {h[i]}")
                    lines.append(f"{n}_bucket{_fmt(l + (('le', '+Inf'),))} {h[-1]}")
                    lines += [f"{n}_sum{_fmt(l)} {h[-2]}", f"{n}_count{_fmt(l)} {h[-1]}"]
        for n, fn in sorted(self.gauges.items()):
            try:
                v = float(fn())
            except Exception:  # noqa: BLE001
                continue
            lines += [f"# HELP {n} {self.help.get(n, n)}", f"# TYPE {n} gauge", f"{n} {v}"]
        return "\n".join(lines) + "\n"


METRICS = Metrics()
for _n, _t in {
    "vox_calls_total": "Calls finished, by medium and end reason",
    "vox_turns_total": "Caller turns answered",
    "vox_interruptions_total": "Barge-ins",
    "vox_tool_calls_total": "Tool calls by tool name",
    "vox_errors_total": "Errors inside calls by component",
    "vox_first_audio_ms": "Caller stops talking -> first agent audio, server side (ms)",
    "vox_llm_first_token_ms": "Speech model time to first token (ms)",
    "vox_tts_ms": "Text-to-speech time per sentence (ms)",
    "vox_tool_ms": "Tool call duration (ms)",
    "vox_active_calls": "Calls in progress on this node",
}.items():
    METRICS.describe(_n, _t)
