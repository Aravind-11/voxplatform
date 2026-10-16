"""API keys, concurrency limits, Twilio request signatures, webhook signing/delivery."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import threading
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
