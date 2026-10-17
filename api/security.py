"""API keys, concurrency limits, Twilio request signatures, webhook signing/delivery."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Awaitable, Callable, Optional

import httpx


# ---------------------------------------------------------------- API keys
def generate_key() -> str:
    return "vox_" + secrets.token_urlsafe(24)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ------------------------------------------------------------ concurrency
class ConcurrencyLimiter:
    """In-process per-tenant active-call cap (single node). A multi-node deployment would back
    this with Redis (INCR/DECR with expiry); only this in-process version is implemented."""

    def __init__(self, default_limit: int = 10):
        self.default_limit = default_limit
        self._active: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def acquire(self, tenant_id: str, call_id: str, limit: Optional[int] = None) -> bool:
        cap = limit if limit is not None else self.default_limit
        with self._lock:
            calls = self._active.setdefault(tenant_id, set())
            if call_id in calls:
                return True
            if len(calls) >= cap:
                return False
            calls.add(call_id)
            return True

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

    def __init__(self, url: str = "", default_limit: int = 10, ttl_s: float = 90.0, client=None, prefix: str = "vox:active:"):
        if client is None:
            import redis
            client = redis.Redis.from_url(url, decode_responses=True)
        self.r, self.default_limit, self.ttl_s, self.prefix = client, default_limit, ttl_s, prefix
        self._script = self.r.register_script(self.LUA)
        self._held: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _key(self, tenant_id: str) -> str:
        return self.prefix + tenant_id

    def ping(self) -> None:
        self.r.ping()

    def acquire(self, tenant_id: str, call_id: str, limit: Optional[int] = None) -> bool:
        import time
        now = time.time()
        cap = limit if limit is not None else self.default_limit
        ok = self._script(keys=[self._key(tenant_id)], args=[now, now + self.ttl_s, call_id, cap]) == 1
        if ok:
            with self._lock:
                self._held.setdefault(tenant_id, set()).add(call_id)
        return ok

    def release(self, tenant_id: str, call_id: str) -> None:
        self.r.zrem(self._key(tenant_id), call_id)
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


class OutboundCap:
    """Daily outbound cap. Redis INCR is atomic across nodes; without Redis the count runs under store.lock.

    A rejected attempt must not consume a slot: the Lua script DECRs when the INCR goes over the cap.
    """

    LUA = """
local n = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
if tonumber(n) > tonumber(ARGV[2]) then
  redis.call('DECR', KEYS[1])
  return 0
end
return 1
"""
    TTL_S = 48 * 60 * 60

    def __init__(self, limiter, store):
        self.limiter, self.store = limiter, store
        self._script = limiter.r.register_script(self.LUA) if isinstance(limiter, RedisLimiter) else None

    def allow(self, tenant_id: str, per_day: int) -> bool:
        """True if this tenant may place another outbound call. Redis key is vox:outbound:{tenant}:{UTC date}."""
        if self._script is not None:
            from datetime import datetime, timezone
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = f"vox:outbound:{tenant_id}:{day}"
            return int(self._script(keys=[key], args=[self.TTL_S, int(per_day)])) == 1
        since = time.time() - 86400
        return self.store.count_outbound(tenant_id, since) < int(per_day)

    @contextmanager
    def hold(self, tenant_id: str, per_day: int):
        """Yield the allow() result. No Redis: hold store.lock across the yield so create_call cannot race the count."""
        if self._script is not None:
            yield self.allow(tenant_id, per_day)
            return
        with self.store.lock:
            yield self.allow(tenant_id, per_day)


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
