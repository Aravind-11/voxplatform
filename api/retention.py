"""Data retention: erases ended calls (record, transcript, recording, saved utterances) older than a tenant's limit.

Limit per tenant: `retention_days` (set with PATCH /admin/tenants/{id}); tenants without one use the server default
VOX_RETENTION_DAYS (unset = keep forever). The API runs `purge_expired` hourly and `POST /admin/purge` runs it on demand."""
from __future__ import annotations

import os
import time
from typing import Optional

from .store import Store


def delete_call_files(record_dir: str, call_id: str) -> None:
    if not record_dir:
        return
    wav = os.path.join(record_dir, f"{call_id}.wav")
    if os.path.isfile(wav):
        os.remove(wav)
    udir = os.path.join(record_dir, "utterances")
    if os.path.isdir(udir):
        for f in os.listdir(udir):
            if f.startswith(call_id + "-"):
                os.remove(os.path.join(udir, f))


def purge_expired(store: Store, record_dir: str, default_days: Optional[int] = None, now: Optional[float] = None) -> dict:
    """Returns {tenant_id: number_of_calls_erased}. Never touches calls that are still live."""
    now = time.time() if now is None else now
    out: dict[str, int] = {}
    for t in store.list_tenants():
        days = t.get("retention_days") or default_days
        if not days:
            continue
        ids = store.expired_calls(t["id"], now - days * 86400)
        for cid in ids:
            delete_call_files(record_dir, cid)
            store.delete_call(t["id"], cid)
        if ids:
            out[t["id"]] = len(ids)
            store.audit(t["id"], "system", "retention.purge", f"{len(ids)} calls", {"olderThanDays": days})
    return out
