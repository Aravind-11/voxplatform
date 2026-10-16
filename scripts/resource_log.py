"""Summarizes a CSV written by scripts/resource_log.sh:  python scripts/resource_log.py summarize resources.csv"""
import csv
import math
import statistics
import sys


def summarize(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, KeyError, TypeError):
                pass
        return out
    res = {"samples": len(rows)}
    for name in ("gpu_util_pct", "gpu_mem_used_mib", "gpu_power_w", "cpu_pct", "host_mem_used_mib", "api_rss_mib", "active_calls"):
        v = col(name)
        if v:
            res[name] = {"mean": round(statistics.fmean(v), 1), "max": max(v), "p95": round(sorted(v)[math.ceil(0.95 * len(v)) - 1], 1)}
    return res


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "summarize":
        sys.exit(__doc__)
    for k, v in summarize(sys.argv[2]).items():
        print(k, v)
