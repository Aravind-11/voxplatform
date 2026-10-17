"""How many calls each API process may hold when one host runs several processes.

WebSockets stick to the process that accepted them, and in-process state (live calls, VAD)
is not shared. `scripts/deploy.sh` therefore starts one uvicorn process per port when
VOX_PROCS > 1, and each process gets `per_process_cap` of VOX_NODE_MAX_CALLS — not uvicorn's
shared `--workers` model, which would split one listener across workers.
"""
from __future__ import annotations


def per_process_cap(node_max: int, procs: int) -> int:
    """`max(1, node_max // procs)`. Rejects `procs < 1` (there is no process to cap)."""
    if isinstance(procs, bool) or not isinstance(procs, int) or procs < 1:
        raise ValueError(f"procs must be an integer >= 1, got {procs!r}")
    return max(1, int(node_max) // procs)
