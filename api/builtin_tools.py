"""Built-in call-control tools: hangUp and transfer. Both act on the live call, and both take effect only
after the agent has finished its spoken reply (see core.session.ToolResult)."""
from __future__ import annotations

import re
from typing import Any
from xml.sax.saxutils import escape

from core.session import ToolResult
from .http_tools import NAME_RE, ToolConfigError

E164 = re.compile(r"^\+[1-9]\d{6,14}$")
BUILTINS = {"hangUp", "transfer"}


def validate(defn: dict) -> dict:
    kind = defn.get("builtin")
    if kind == "hangUp":
        return {"builtin": "hangUp", "strict": bool(defn.get("strict", True))}
    if kind == "transfer":
        targets = defn.get("targets")
        if not isinstance(targets, dict) or not targets:
            raise ToolConfigError("transfer needs 'targets': {name: +E164 number}")
        for n, num in targets.items():
            if not NAME_RE.match(str(n)) or not isinstance(num, str) or not E164.match(num):
                raise ToolConfigError(f"transfer target {n!r}: name must be a word and number E.164 (+15551234567)")
        return {"builtin": "transfer", "targets": dict(targets),
                "description": str(defn.get("description") or "Transfer the caller to a human. Tell them you are transferring first.")[:1000]}
    raise ToolConfigError(f"unknown builtin {kind!r}; available: {sorted(BUILTINS)}")


def schema(defn: dict) -> dict:
    if defn["builtin"] == "hangUp":
        return {"type": "function", "function": {
            "name": "hangUp", "description": "End the call. Say a short goodbye in the same reply; the line drops after you finish speaking.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}}}
    return {"type": "function", "function": {
        "name": "transfer", "description": defn["description"],
        "parameters": {"type": "object", "properties": {"to": {"type": "string", "enum": sorted(defn["targets"])}},
                       "required": ["to"]}}}


def make(defn: dict, ctx, client):
    if defn["builtin"] == "hangUp":
        async def hang_up(args: dict) -> Any:
            return ToolResult({"ok": True, "reason": str((args or {}).get("reason", ""))[:200]},
                              after=ctx.end, strict=defn["strict"])
        return hang_up

    async def transfer(args: dict) -> Any:
        to = str((args or {}).get("to", ""))
        number = defn["targets"].get(to)             # the model can only pick a configured name, never a raw number
        if number is None:
            return {"error": f"unknown target {to!r}", "targets": sorted(defn["targets"])}
        if not ctx.call_sid or not ctx.twilio or client is None:
            return {"error": "transfer is only available on phone calls with Twilio credentials configured"}

        async def do() -> None:
            r = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{ctx.twilio['sid']}/Calls/{ctx.call_sid}.json",
                auth=(ctx.twilio["sid"], ctx.twilio["token"]), timeout=10,
                data={"Twiml": f"<Response><Dial>{escape(number)}</Dial></Response>"})
            if r.status_code >= 300:
                raise RuntimeError(f"twilio transfer failed: HTTP {r.status_code}")
        return ToolResult({"transferring": to}, after=do, strict=True)
    return transfer
