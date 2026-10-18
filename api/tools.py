"""Tool registry: turns an agent's `tools` list into (schemas for the model, callables for the Session).

Entries: a name string (demo tools below, or "hangUp"), or an object: an HTTP tool (see http_tools) or a
built-in with options, e.g. {"builtin": "transfer", "targets": {"billing": "+15551234567"}}.
Later entries override earlier ones with the same name (so a tenant can replace a built-in).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from . import builtin_tools, http_tools
from .http_tools import ToolConfigError

SLOTS = {"monday": ["09:00", "14:00"], "tuesday": ["11:00"], "wednesday": []}

DEMO_SCHEMAS = {
    "check_availability": {"type": "function", "function": {
        "name": "check_availability", "description": "List free appointment slots for a weekday.",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string", "description": "Weekday name in lowercase, e.g. monday"}}, "required": ["day"]}}},
    "book_slot": {"type": "function", "function": {
        "name": "book_slot", "description": "Book an appointment slot.",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string", "description": "Weekday name in lowercase, e.g. monday"},
            "time": {"type": "string", "description": "24-hour HH:MM, e.g. 14:00 for 2 pm"}},
                       "required": ["day", "time"]}}},
}


_TIME = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?m?\.?)?\s*$")


def norm_time(t) -> str:
    """'2 pm' / '2:00 PM' / '14' / '9 a.m.' -> 'HH:MM'. The model often says times the way people do."""
    m = _TIME.match(str(t).lower())
    if not m:
        return str(t).strip()
    h, mi, ap = int(m[1]), int(m[2] or 0), m[3]
    if ap:
        h = h % 12 + (12 if ap == "p" else 0)
    return f"{h:02d}:{mi:02d}"


async def _noop_end() -> None:
    return None


@dataclass
class CallContext:
    """Facts about the live call that tools may use. `end` closes the call's transport."""
    call_id: str = ""
    caller: str = ""
    callee: str = ""
    call_sid: str = ""
    end: Callable[[], Awaitable[None]] = _noop_end
    twilio: Optional[dict] = None          # {"sid": ..., "token": ...} for transfer
    turn_fn: Callable[[], int] = lambda: 0  # how many caller turns have ended so far (set by the runtime)
    docs: Optional[Callable[[str], tuple]] = None             # knowledge base name -> (version, [{id,title,text}])
    send_dtmf: Optional[Callable[[str], Awaitable[None]]] = None    # press keypad digits on a phone call
    voicemail: Optional[Callable[[str], Awaitable[None]]] = None    # speak a message, then hang up
    client_call: Optional[Callable[..., Awaitable[Any]]] = None     # run a tool in the caller's browser
    embedder: Any = None                                            # optional core.embed.OpenAIEmbedder (hybrid KB search)
    base_url: str = ""                                              # public URL of this API (warm-transfer callbacks)


def _canon(args: dict) -> str:
    """Order- and case-insensitive key for 'the same request', so '  Tuesday' and 'tuesday' match."""
    norm = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in args.items()}
    return json.dumps(norm, sort_keys=True, default=str)


def with_confirmation(fn, schema: dict, ctx):
    """Makes a tool that changes things safe against mishearing. The first call does NOT run it: the model gets
    the details to read back and must ask the caller. Only a second call with confirmed=true, with the same
    arguments, made in a LATER caller turn, runs it. A mis-heard booking becomes an audible 'Tuesday at 11,
    correct?' that the caller can fix, instead of a silent action."""
    sch = copy.deepcopy(schema)
    f = sch["function"]
    f["parameters"].setdefault("properties", {})["confirmed"] = {
        "type": "boolean",
        "description": "true ONLY after the caller heard these exact details read back and clearly said yes."}
    f["description"] += (" This changes things for real. First call it WITHOUT confirmed to get the details to read "
                         "back to the caller; call again with confirmed=true only after they agree.")
    pending: dict[str, int] = {}

    async def run(args: dict):
        args = dict(args or {})
        confirmed = args.pop("confirmed", False) is True
        key, turn = _canon(args), ctx.turn_fn()
        if confirmed and key in pending and pending[key] < turn:
            pending.pop(key)
            return await fn(args)
        pending[key] = turn
        return {"executed": False,
                "next_step": ("NOT DONE YET. Read these details back to the caller in one short sentence and ask them "
                              "to confirm. Do not say it is done. Only after the caller clearly says yes, call this "
                              "tool again with the same arguments and confirmed=true."),
                "details": args}
    return sch, run


def resolve_library(specs: list, lookup: Callable[[str], Optional[Any]]) -> list:
    """Replaces {"library": "name"} entries with the tenant's stored tool definition (resolved again at every call,
    so editing a library tool updates every agent that uses it). `lookup(name)` returns the spec or None."""
    out = []
    for sp in specs or []:
        if isinstance(sp, dict) and "library" in sp:
            found = lookup(str(sp["library"]))
            if found is None:
                raise ToolConfigError(f"tool library has no tool named {sp['library']!r}")
            out.append(found)
        else:
            out.append(sp)
    return out


def schema_tokens(schemas) -> int:
    """Rough token count of the tool schemas shown to the model (4 characters ≈ 1 token)."""
    if not schemas:
        return 0
    return max(1, len(json.dumps(schemas, separators=(",", ":"))) // 4)


def warn_schema_tokens(schemas, limit: int = 2000) -> str | None:
    """A warning when tool schemas would crowd out the conversation. Never a reason to reject the agent."""
    n = schema_tokens(schemas)
    if n > limit:
        return (f"tool schemas are about {n} tokens, over the {limit} token budget; "
                "they may crowd out the conversation")
    return None


def validate_specs(specs: list) -> list[dict]:
    """Normalizes an agent's tool list or raises ToolConfigError (surfaced as HTTP 422 at agent creation)."""
    out: list[dict] = []
    for sp in specs or []:
        if isinstance(sp, str):
            if sp in DEMO_SCHEMAS:
                out.append({"demo": sp})
            elif sp in ("hangUp", "playDtmf"):
                out.append(builtin_tools.validate({"builtin": sp}))
            else:
                raise ToolConfigError(f"unknown tool {sp!r}; use a name from "
                                      f"{sorted(DEMO_SCHEMAS) + ['hangUp', 'playDtmf']} or a tool definition object")
        elif isinstance(sp, dict) and "tool" in sp:          # a named tool with options: {"tool": "book_slot", "confirm": true}
            if sp["tool"] not in DEMO_SCHEMAS:
                raise ToolConfigError(f"unknown tool {sp['tool']!r}; options apply to {sorted(DEMO_SCHEMAS)}")
            out.append({"demo": sp["tool"], "confirm": bool(sp.get("confirm", False))})
        elif isinstance(sp, dict) and "builtin" in sp:
            out.append(builtin_tools.validate(sp))
        else:
            out.append(http_tools.validate(sp))
    seen: set[str] = set()
    for d in out:
        n = d.get("name") or d.get("demo") or d.get("builtin")      # client tools are named by their own name
        if n in seen:
            raise ToolConfigError(f"tool {n!r} is listed twice")
        seen.add(n)
    if len(out) > 32:
        raise ToolConfigError("at most 32 tools per agent")
    return out


def redact_specs(specs: list) -> list:
    return [http_tools.redact(s) if isinstance(s, dict) and "http" in s else s for s in specs or []]


def build(specs: list, ctx: Optional[CallContext] = None, client: Any = None):
    """Returns (schemas, callables). `ctx`/`client` are needed only by HTTP tools and transfer."""
    ctx = ctx or CallContext()
    slots = {k: list(v) for k, v in SLOTS.items()}

    async def check_availability(args: dict):
        return {"day": args.get("day"), "free": slots.get(str(args.get("day", "")).lower(), [])}

    async def book_slot(args: dict):
        day, t = str(args.get("day", "")).lower().strip(), norm_time(args.get("time", ""))
        if t in slots.get(day, []):
            slots[day].remove(t)
            return {"booked": True, "day": day, "time": t}
        return {"booked": False, "reason": "slot not available"}

    demo = {"check_availability": check_availability, "book_slot": book_slot}
    schemas: dict[str, dict] = {}
    fns: dict[str, Callable] = {}
    for d in validate_specs(specs):
        if "demo" in d:
            sch, fn = DEMO_SCHEMAS[d["demo"]], demo[d["demo"]]
            if d.get("confirm"):
                sch, fn = with_confirmation(fn, sch, ctx)
            schemas[d["demo"]], fns[d["demo"]] = sch, fn
        elif "builtin" in d:
            key = d.get("name") or d["builtin"]
            schemas[key], fns[key] = builtin_tools.schema(d), builtin_tools.make(d, ctx, client)
        else:
            sch, fn = http_tools.schema(d), http_tools.make_http_tool(d, ctx, client)
            if d.get("confirm"):
                sch, fn = with_confirmation(fn, sch, ctx)
            schemas[d["name"]], fns[d["name"]] = sch, fn
    return list(schemas.values()), fns
