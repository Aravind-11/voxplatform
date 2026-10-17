"""Print mean task success and first-audio latency for two JSONL score files.

Each line is a dict already produced by harness.eval_conv.replay (or the same shape from
a paid service). This script does not call that service and does not need the audio.

    python scripts/eval_compare.py ours.jsonl baseline.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.eval_conv import compare  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare conversation-eval JSONL scores.")
    p.add_argument("ours", help="JSONL of our score dicts")
    p.add_argument("baseline", help="JSONL of baseline score dicts")
    args = p.parse_args(argv)
    print(json.dumps(compare(load_jsonl(args.ours), load_jsonl(args.baseline)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
