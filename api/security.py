"""API keys, concurrency limits, Twilio request signatures, webhook signing/delivery."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
from typing import Awaitable, Callable, Mapping, Optional

log = logging.getLogger(__name__)

import httpx


# ---------------------------------------------------------------- API keys
def generate_key() -> str:
    return "vox_" + secrets.token_urlsafe(24)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ------------------------------------------------------------ concurrency
def process_call_ceiling(env: Mapping[str, str]) -> Optional[int]:
    """Local ceiling already stored in VOX_NODE_MAX_CALLS.

    deploy.sh sets that variable to this process's share and leaves VOX_PROCS as the host
    count. Dividing by VOX_PROCS again would shrink the share (8 with VOX_PROCS=4 becomes 2).
    Missing means no extra ceiling. 0 admits nobody. A negative or non-integer value fails startup.
    """
    if env is None or "VOX_NODE_MAX_CALLS" not in env:
        return None
    raw = env["VOX_NODE_MAX_CALLS"]
    if not isinstance(raw, str):
        raise ValueError(f"VOX_NODE_MAX_CALLS must be an integer >= 0, got {raw!r}")
    text = raw.strip()
    digits = text[1:] if text.startswith(("+", "-")) else text
    if not digits.isascii() or not digits.isdigit():
        raise ValueError(f"VOX_NODE_MAX_CALLS must be an integer >= 0, got {raw!r}")
    value = int(text)
    if value < 0:
        raise ValueError(f"VOX_NODE_MAX_CALLS must be an integer >= 0, got {raw!r}")
    return value


def _checked_ceiling(process_ceiling: Optional[int]) -> Optional[int]:
    if process_ceiling is None:
        return None
    if isinstance(process_ceiling, bool) or not isinstance(process_ceiling, int) or process_ceiling < 0:
        raise ValueError(f"process call ceiling must be an integer >= 0, got {process_ceiling!r}")
    return process_ceiling


def make_limiter(env: Mapping[str, str], default_concurrency: int, *, redis_client=None):
    """One process's limiter. The ceiling stays on this process; the tenant cap may be shared via Redis."""
    ceiling = process_call_ceiling(env)
    redis_url = env.get("VOX_REDIS_URL", "") if env else ""
    if redis_url:
        return RedisLimiter(redis_url, default_concurrency, client=redis_client, process_ceiling=ceiling)
    return ConcurrencyLimiter(default_concurrency, process_ceiling=ceiling)


class ConcurrencyLimiter:
    """In-process per-tenant active-call cap (single node). A multi-node deployment would back
    this with Redis (INCR/DECR with expiry); only this in-process version is implemented."""

    def __init__(self, default_limit: int = 10, process_ceiling: Optional[int] = None):
        self.default_limit = default_limit
        self.process_ceiling = _checked_ceiling(process_ceiling)
        self._active: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def try_admit(self, tenant_id: str, call_id: str, limit: Optional[int] = None, node_limit: int = 0) -> str:
        """Reserve one call under the tenant cap and the per-process cap, in one lock.

        Returns "ok", "node", or "tenant". A refusal does not take a slot. A call that
        already holds a slot is "ok" and is not counted twice. ``node_limit <= 0``
        does not apply that argument. ``process_ceiling`` still applies: None adds
        nothing, 0 admits nobody, and N caps this process. When both are set, the
        smaller cap wins. The check happens before the insert, so two admits cannot
        both pass a cap of N.
        """
        cap = self.default_limit if limit is None else limit
        with self._lock:
            calls = self._active.get(tenant_id)
            if calls is not None and call_id in calls:
                return "ok"
            held = sum(len(v) for v in self._active.values())
            # Across tenants, before a tenant slot is taken. 0 admits nobody; None adds no ceiling.
            if self.process_ceiling is not None and held >= self.process_ceiling:
                return "node"
            if node_limit > 0 and held >= node_limit:
                return "node"
            if (len(calls) if calls is not None else 0) >= cap:
                return "tenant"
            if calls is None:
                calls = self._active[tenant_id] = set()
            calls.add(call_id)
            return "ok"

    def acquire(self, tenant_id: str, call_id: str, limit: Optional[int] = None) -> bool:
        return self.try_admit(tenant_id, call_id, limit, 0) == "ok"

    def release(self, tenant_id: str, call_id: str) -> None:
        with self._lock:
            self._active.get(tenant_id, set()).discard(call_id)

    def active(self, tenant_id: str) -> int:
        with self._lock:
            return len(self._active.get(tenant_id, ()))

    def total_active(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._active.values())

    local_active = total_active                    # one process: this node's calls are all the calls


class RedisLimiter:
    """Per-tenant active-call cap shared by every API node (Redis). Same interface as ConcurrencyLimiter.

    Each active call is a member of a sorted set whose score is its expiry time. The check-and-add is one Lua script
    (atomic across nodes). A node that crashes stops refreshing its calls, so they expire after `ttl_s` instead of
    holding capacity forever; live nodes refresh theirs every ttl_s/3 via `heartbeat()`."""

    LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZSCORE', KEYS[1], ARGV[3]) then redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3]) return 1 end
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[4]) then return 0 end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
return 1
"""

    def __init__(self, url: str = "", default_limit: int = 10, ttl_s: float = 90.0, client=None, prefix: str = "vox:active:",
                 process_ceiling: Optional[int] = None):
        if client is None:
            import redis
            client = redis.Redis.from_url(url, decode_responses=True)
        self.r, self.default_limit, self.ttl_s, self.prefix = client, default_limit, ttl_s, prefix
        self.process_ceiling = _checked_ceiling(process_ceiling)
        self._script = self.r.register_script(self.LUA)
        self._held: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _key(self, tenant_id: str) -> str:
        return self.prefix + tenant_id

    def ping(self) -> None:
        self.r.ping()

    def try_admit(self, tenant_id: str, call_id: str, limit: Optional[int] = None, node_limit: int = 0) -> str:
        """Same contract as ConcurrencyLimiter.try_admit, with the tenant cap in Redis.

        The per-process cap is this node's ``_held`` set, not the shared Redis key.
        ``node_limit <= 0`` does not apply that argument. ``process_ceiling`` still
        applies: None adds nothing, 0 admits nobody, and N caps this process. When
        either local cap is active, the slot is reserved under the lock before the
        Redis round trip and dropped again if Redis refuses, so two admits cannot
        both pass. With neither local cap, Redis is checked first.
        """
        import time
        now = time.time()
        cap = self.default_limit if limit is None else limit
        local_cap = self.process_ceiling is not None or node_limit > 0
        already = False
        if local_cap:
            with self._lock:
                current = self._held.get(tenant_id)
                if current is not None and call_id in current:
                    already = True
                else:
                    held = sum(len(v) for v in self._held.values())
                    if self.process_ceiling is not None and held >= self.process_ceiling:
                        return "node"
                    if node_limit > 0 and held >= node_limit:
                        return "node"
                    self._held.setdefault(tenant_id, set()).add(call_id)
        try:
            ok = self._script(keys=[self._key(tenant_id)], args=[now, now + self.ttl_s, call_id, cap]) == 1
        except Exception:
            if local_cap and not already:
                with self._lock:
                    self._held.get(tenant_id, set()).discard(call_id)
            raise
        if ok:
            if not local_cap:
                with self._lock:
                    self._held.setdefault(tenant_id, set()).add(call_id)
            return "ok"
        if local_cap and not already:
            with self._lock:
                self._held.get(tenant_id, set()).discard(call_id)
        return "tenant"

    def acquire(self, tenant_id: str, call_id: str, limit: Optional[int] = None) -> bool:
        return self.try_admit(tenant_id, call_id, limit, 0) == "ok"

    def release(self, tenant_id: str, call_id: str) -> None:
        # Drop the local slot even when ZREM fails, so a Redis blip cannot pin this node at the cap
        # or skip the rest of finish() (the call would stay active and /healthz would stay 503).
        try:
            self.r.zrem(self._key(tenant_id), call_id)
        except Exception:  # noqa: BLE001
            log.exception("redis ZREM failed; releasing the local slot for %s", call_id)
        finally:
            with self._lock:
                self._held.get(tenant_id, set()).discard(call_id)

    def active(self, tenant_id: str) -> int:
        import time
        return int(self.r.zcount(self._key(tenant_id), time.time(), "+inf"))

    def local_active(self) -> int:
        """Calls held by THIS node (what a drain waits for, and what one node's capacity is measured against)."""
        with self._lock:
            return sum(len(v) for v in self._held.values())

    def total_active(self) -> int:
        """Calls on every node (scans Redis; for dashboards, not for health checks)."""
        import time
        now = time.time()
        return sum(int(self.r.zcount(k, now, "+inf")) for k in self.r.scan_iter(self.prefix + "*"))

    def refresh(self) -> None:
        """Extends the expiry of every call this node holds."""
        import time
        exp = time.time() + self.ttl_s
        with self._lock:
            held = [(t, c) for t, cs in self._held.items() for c in cs]
        for t, c in held:
            self.r.zadd(self._key(t), {c: exp}, xx=True)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.ttl_s / 3)
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.refresh)
            except Exception:  # noqa: BLE001  (Redis blip: try again next tick)
                pass


# ------------------------------------------------------- Twilio signatures
def twilio_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """base64(HMAC-SHA1(auth_token, url + sorted key/value pairs concatenated))."""
    data = url + "".join(k + params[k] for k in sorted(params))
    mac = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return base64.b64encode(mac).decode()


def validate_twilio(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    return hmac.compare_digest(twilio_signature(auth_token, url, params), signature or "")


# ---------------------------------------------------------------- webhooks
def sign_body(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class WebhookDispatcher:
    """POSTs signed JSON events with exponential-backoff retries. Fire-and-forget via spawn()."""

    def __init__(self, client: httpx.AsyncClient, retries: int = 3, backoff_s: float = 0.5,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.client, self.retries, self.backoff_s, self.sleep = client, retries, backoff_s, sleep
        self._tasks: set[asyncio.Task] = set()

    async def send(self, url: str, secret: str, event: dict) -> bool:
        body = json.dumps(event, separators=(",", ":"), default=str).encode()
        headers = {"Content-Type": "application/json", "X-Vox-Event": event.get("type", ""),
                   "X-Vox-Signature": sign_body(secret, body)}
        for attempt in range(self.retries):
            try:
                r = await self.client.post(url, content=body, headers=headers, timeout=10)
                if 200 <= r.status_code < 300:
                    return True
            except httpx.HTTPError:
                pass
            if attempt < self.retries - 1:
                await self.sleep(self.backoff_s * (2 ** attempt))
        return False

    def spawn(self, url: str, secret: str, event: dict) -> None:
        t = asyncio.get_running_loop().create_task(self.send(url, secret, event))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
