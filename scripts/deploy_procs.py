"""How many calls each API process may hold when one host runs several processes.

WebSockets stick to the process that accepted them, and in-process state (live calls, VAD)
is not shared. `scripts/deploy.sh` therefore starts one uvicorn process per port when
VOX_PROCS > 1, and each process gets `per_process_cap` of VOX_NODE_MAX_CALLS — not uvicorn's
shared `--workers` model, which would split one listener across workers. The shares sum to
at most the node cap.
"""
from __future__ import annotations


def per_process_cap(node_max: int, procs: int) -> int:
    """Calls one process may hold so ``procs * cap <= node_max``.

    Integer floor of ``node_max / procs``. Rejects ``procs < 1``. A quotient of
    0 is not rounded up to 1: that would let the processes together admit more
    calls than ``node_max`` (4 processes with a cap of 3 would admit 4). A node
    cap of 0 stays 0. A positive cap smaller than ``procs`` is rejected: no
    positive uniform share stays inside the cap.
    """
    if isinstance(procs, bool) or not isinstance(procs, int) or procs < 1:
        raise ValueError(f"procs must be an integer >= 1, got {procs!r}")
    if isinstance(node_max, bool):
        raise ValueError(f"node_max must be an integer >= 0, got {node_max!r}")
    try:
        node_max = int(node_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"node_max must be an integer >= 0, got {node_max!r}") from exc
    if node_max < 0:
        raise ValueError(f"node_max must be an integer >= 0, got {node_max!r}")
    if node_max == 0:
        return 0
    cap = node_max // procs
    if cap < 1:
        raise ValueError(
            f"node cap {node_max} cannot be split across {procs} processes without exceeding it"
        )
    return cap
