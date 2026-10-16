"""Prepaid billing: Stripe Checkout top-ups, webhook verification, call pricing.

Model: a self-serve tenant buys credit (Stripe Checkout, one-off payment). Each finished call debits
seconds x rate from the credit; calls are refused when the balance is empty and a live call is ended politely
when it runs out. Money is kept as integer micro-dollars (1 USD = 1_000_000) in an append-only ledger
(api/store.py), so a Stripe event or a call can only ever be applied once.

Only the Stripe REST API and webhook signature scheme are used (no SDK). Tested here against a fake Stripe server
and Stripe's documented signature format; NOT yet exercised against real Stripe (needs your keys).
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

MICROS = 1_000_000
CENTS_TO_MICROS = 10_000


def usd_to_micros(usd: float) -> int:
    return int(round(usd * MICROS))


def micros_to_usd(micros: int) -> float:
    return round(micros / MICROS, 6)


def call_cost_micros(seconds: float, rate_per_minute: float) -> int:
    """Per-second billing, rounded to the nearest micro-dollar. Calls under half a second cost nothing."""
    if seconds <= 0.5 or rate_per_minute <= 0:
        return 0
    return int(round(seconds / 60.0 * rate_per_minute * MICROS))


def verify_stripe_signature(payload: bytes, header: str, secret: str, now: Optional[float] = None,
                            tolerance_s: int = 300) -> bool:
    """Stripe-Signature: `t=<unix>,v1=<hex hmac-sha256 of "<t>.<payload>">[,v1=...]`. Rejects stale timestamps."""
    if not secret or not header:
        return False
    parts = [p.split("=", 1) for p in header.split(",") if "=" in p]
    ts = next((v for k, v in parts if k.strip() == "t"), "")
    sigs = [v.strip() for k, v in parts if k.strip() == "v1"]
    try:
        ts_i = int(ts)
    except ValueError:
        return False
    if abs((now if now is not None else time.time()) - ts_i) > tolerance_s:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, s) for s in sigs)


async def create_checkout(http, secret_key: str, api_base: str, *, tenant_id: str, email: str, amount_cents: int,
                          success_url: str, cancel_url: str, product_name: str = "Voice agent credit") -> dict:
    """Creates a Stripe Checkout Session for a one-off top-up. Returns {"id", "url"}. Raises RuntimeError on failure."""
    data = {"mode": "payment", "success_url": success_url, "cancel_url": cancel_url,
            "client_reference_id": tenant_id, "metadata[tenant_id]": tenant_id,
            "line_items[0][quantity]": "1", "line_items[0][price_data][currency]": "usd",
            "line_items[0][price_data][unit_amount]": str(amount_cents),
            "line_items[0][price_data][product_data][name]": product_name}
    if email:
        data["customer_email"] = email
    r = await http.post(f"{api_base.rstrip('/')}/v1/checkout/sessions", data=data, timeout=15,
                        headers={"Authorization": f"Bearer {secret_key}"})
    if r.status_code >= 300:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        raise RuntimeError(f"Stripe refused the request (HTTP {r.status_code}): {msg}")
    j = r.json()
    return {"id": j["id"], "url": j["url"]}
