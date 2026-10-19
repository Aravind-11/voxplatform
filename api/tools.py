"""Tool registry: turns an agent's `tools` list into (schemas for the model, callables for the Session).

Entries: a name string (demo tools below, or "hangUp"), or an object: an HTTP tool (see http_tools) or a
built-in with options, e.g. {"builtin": "transfer", "targets": {"billing": "+15551234567"}}.
Later entries override earlier ones with the same name (so a tenant can replace a built-in).
"""
from __future__ import annotations

import asyncio
import copy
import inspect
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
    guard_slots: bool = False                                       # require spoken read-back before state-changing tools
    last_assistant_text: Any = ""                                   # str, or a callable -> the previous assistant message
    call_control: Any = None                                        # TwilioCallControl or AsteriskCallControl; None uses ctx.twilio
    medium: str = ""                                                # sip, twilio, webSocket; which controller a live call may use
    record_sip_transfer: Optional[Callable[[str], bool]] = None    # SIP: store the dialplan target; None keeps ARI redirect
    tool_budget: int = 8                                            # side-effect calls allowed on this call; the next one is refused
    tool_calls: int = 0
    last_user_text: Callable[[], Any] = lambda: ""                  # str or awaitable; every caller line, for the injection check
    latest_user_line: Any = None                                    # sync () -> latest line only, placeholder included; None uses last_user_text
    audit: Callable[..., Any] = lambda *a, **k: None                # (action, detail); a failure must not undo the tool


# Spoken read-back the model is told to do. It does not correct a mishearing: a wrong value that was
# said out loud still passes the guard below.
READBACK_INSTRUCTION = ("Critical slots (names, dates, times, numbers, amounts) must be read back "
                        "in the agent's own words before a tool that changes state.")
_SLOT_MARKS = ("book", "create", "update", "delete", "transfer", "purchase")
_READBACK_STEP = "Read these values back to the caller before calling the tool."
# Sentence marks and currency only. A sign, accounting parentheses, percent, or unicode minus is part of the amount.
_SLOT_EDGE = ".,!?;:'\"$£€"

# Phrase list only. A caller who avoids these words is not stopped, and a quoted phrase can still false-positive.
_INJECTION = re.compile(
    r"ignore (?:all |your |previous )?instructions|system prompt|you are now|jailbreak|developer message",
    re.IGNORECASE)
_BUDGET_ERROR = "tool call budget for this call is used up"
_INJECTION_ERROR = "refused: the caller's words look like an instruction to the model, not a request to the business"


def needs_slot_guard(specs: list) -> bool:
    """True when any tool asks for confirmation or an explicit slot read-back."""
    for sp in specs or []:
        if isinstance(sp, dict) and (sp.get("confirm") or sp.get("confirmSlots")):
            return True
    return False


def _changes_state(name: str, confirmed: bool) -> bool:
    low = name.lower()
    return bool(confirmed) or any(mark in low for mark in _SLOT_MARKS)


def _assistant_text(ctx) -> str:
    raw = getattr(ctx, "last_assistant_text", "")
    if callable(raw):
        try:
            raw = raw()
        except Exception:  # noqa: BLE001  (a read-back lookup must not become a tool error)
            return ""
    return "" if raw is None else str(raw)


def _slot_tokens(text: str) -> list[str]:
    """Whitespace tokens. Only sentence marks and currency are stripped at the edges, so 9:10 stays atomic and -100.50 does not become 100.50."""
    tokens: list[str] = []
    for raw in str(text).lower().split():
        token = raw.strip(_SLOT_EDGE)
        if token:
            tokens.append(token)
    return tokens


def _slot_was_said(value: str, said: str) -> bool:
    raw = value.strip()
    if len(raw) < 3:
        return True
    needle = _slot_tokens(raw)
    if not needle:
        return False
    hay = _slot_tokens(said)
    n = len(needle)
    if len(hay) < n:
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def injection_risk(text) -> bool:
    return bool(text) and _INJECTION.search(str(text)) is not None


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


async def _injection_blocked(ctx, args) -> bool:
    """True when a caller line or a string argument matches the injection phrase list."""
    if injection_risk(await _caller_text(ctx)):
        return True
    return any(injection_risk(text) for text in _iter_strings(args))


async def _caller_text(ctx) -> str:
    """Caller text from the provider. A coroutine provider is awaited; anything else non-str is empty."""
    provider = getattr(ctx, "last_user_text", None)
    if provider is None:
        return ""
    try:
        text = provider() if callable(provider) else provider
        if inspect.isawaitable(text):
            text = await text
    except asyncio.CancelledError:
        raise  # an empty line would fail open and run the side effect
    except Exception:  # noqa: BLE001
        return ""
    return text if isinstance(text, str) else ""


def _injection_refusal() -> dict:
    return {"executed": False, "error": _INJECTION_ERROR}


def _with_budget(fn, ctx):
    """Shared per-call cap. The refused call does not run `fn` and does not increment the counter."""
    async def run(args: dict):
        args = dict(args or {})
        if await _injection_blocked(ctx, args):
            return _injection_refusal()
        used = int(getattr(ctx, "tool_calls", 0) or 0)
        budget = int(getattr(ctx, "tool_budget", 8))
        if used >= budget:
            return {"error": _BUDGET_ERROR}
        ctx.tool_calls = used + 1
        return await fn(args)
    return run


def slot_guard(fn, ctx):
    """Block a state-changing call until each string argument (length >= 3) was in the assistant's last message.

    Runs outside `with_confirmation`. A value the model invented and never said is refused; a value it
    misheard and then repeated is not. That limit stays with the speech model. An injection phrase is
    refused before the read-back check. The phrase list is not a complete defense.
    """
    async def run(args: dict):
        args = dict(args or {})
        if await _injection_blocked(ctx, args):
            return _injection_refusal()
        if not getattr(ctx, "guard_slots", False):
            return await fn(args)
        said = _assistant_text(ctx)
        for value in args.values():
            if isinstance(value, str) and not _slot_was_said(value, said):
                return {"executed": False, "next_step": _READBACK_STEP, "details": args}
        return await fn(args)
    return run



def _canon(args: dict) -> str:
    """Order- and case-insensitive key for 'the same request', so '  Tuesday' and 'tuesday' match."""
    norm = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in args.items()}
    return json.dumps(norm, sort_keys=True, default=str)


# Whole-utterance agreements. A longer phrase is listed before the word it contains ("that's right" before "right")
# so a stack of them ("yes that's right") still matches, and a hedge ("yes but make it Monday") does not.
_AFFIRM = (
    "that's right", "thats right", "that is right", "go ahead", "do it",
    "yes", "yeah", "yep", "correct", "right", "ok", "okay", "sure", "confirm",
    "sí", "si", "oui", "correcto", "vale", "haan", "hai",
)
_REJECT = re.compile(r"\b(?:not|don't|dont|actually)\b")
_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _normalize_utterance(text: str) -> str:
    t = _PUNCT.sub(" ", str(text).strip().lower())
    return _SPACE.sub(" ", t).strip()


def _is_affirmative(text: str) -> bool:
    """True only when the caller agreed: the whole line is one or more affirmative phrases, with no hedge."""
    norm = _normalize_utterance(text)
    if not norm or _REJECT.search(norm):
        return False
    rest = norm
    while rest:
        match = next((p for p in _AFFIRM if rest == p or rest.startswith(p + " ")), None)
        if match is None:
            return False
        rest = rest[len(match):].lstrip()
    return True


def _confirm_transcript(ctx) -> str:
    """The latest caller line, for consent. Not the joined injection text.

    `latest_user_line` wins. Otherwise a synchronous `last_user_text` is that line.
    A coroutine provider joins every line for the injection check, so it is closed
    and not treated as a yes.
    """
    provider = getattr(ctx, "latest_user_line", None)
    if not callable(provider):
        provider = getattr(ctx, "last_user_text", None)
    if provider is None:
        return ""
    try:
        text = provider() if callable(provider) else provider
    except Exception:  # noqa: BLE001  (a transcript lookup must not become a tool error)
        return ""
    if inspect.isawaitable(text):
        close = getattr(text, "close", None)
        if callable(close):
            close()
        return ""
    return text if isinstance(text, str) else ""


def with_confirmation(fn, schema: dict, ctx):
    """Makes a tool that changes things safe against mishearing. The first call does NOT run it: the model gets
    the details to read back and must ask the caller. Only a second call with confirmed=true, with the same
    arguments, made in a LATER caller turn, AND an affirmative caller transcript, runs it. A mis-heard booking
    becomes an audible 'Tuesday at 11, correct?' that the caller can fix, instead of a silent action."""
    sch = copy.deepcopy(schema)
    f = sch["function"]
    f["parameters"].setdefault("properties", {})["confirmed"] = {
        "type": "boolean",
        "description": "true ONLY after the caller heard these exact details read back and clearly said yes."}
    f["description"] += (" This changes things for real. First call it WITHOUT confirmed to get the details to read "
                         "back to the caller; call again with confirmed=true only after they agree.")
    name = f.get("name", "")
    pending: dict[str, int] = {}

    async def run(args: dict):
        args = dict(args or {})
        if await _injection_blocked(ctx, args):
            return _injection_refusal()
        confirmed = args.pop("confirmed", False) is True
        key, turn = _canon(args), ctx.turn_fn()
        transcript = _confirm_transcript(ctx)
        if confirmed and key in pending and pending[key] < turn and _is_affirmative(transcript):
            detail = {"name": name, "args": dict(args), "transcript": transcript}
            result = await fn(args)
            pending.pop(key)  # a raise leaves the proposal; a failed audit must not undo this
            try:
                audit = getattr(ctx, "audit", None)
                if callable(audit):
                    audit("tool.confirmed", detail)
            except Exception:  # noqa: BLE001 (a logging failure is not the tool result)
                pass
            return result
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


def filler_texts(specs: list) -> dict[str, str]:
    """Per-tool spoken filler. Missing fillerText uses the session default; an explicit blank disables it."""
    out: dict[str, str] = {}
    for d in validate_specs(specs):
        if "fillerText" not in d:
            continue
        text = d.get("fillerText")
        if not isinstance(text, str):
            continue
        name = d.get("name") or d.get("demo") or d.get("builtin")
        if name:
            out[str(name)] = text.strip()
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

    def prepare(name: str, sch: dict, fn, confirmed: bool, *, budget: bool = False):
        if budget:
            # Inside confirmation: a proposal that does not run the tool does not spend a slot.
            fn = _with_budget(fn, ctx)
        if confirmed:
            sch, fn = with_confirmation(fn, sch, ctx)
        # Slot guard is the outer wrapper, so a missing read-back never reaches confirm or the budget.
        if getattr(ctx, "guard_slots", False) and _changes_state(name, confirmed):
            fn = slot_guard(fn, ctx)
        return sch, fn

    for d in validate_specs(specs):
        if "demo" in d:
            sch, fn = prepare(d["demo"], DEMO_SCHEMAS[d["demo"]], demo[d["demo"]], bool(d.get("confirm")),
                              budget=d["demo"] == "book_slot")
            schemas[d["demo"]], fns[d["demo"]] = sch, fn
        elif "builtin" in d:
            key = d.get("name") or d["builtin"]
            sch, fn = prepare(key, builtin_tools.schema(d), builtin_tools.make(d, ctx, client),
                              bool(d.get("confirm")), budget=d["builtin"] in ("hangUp", "transfer"))
            schemas[key], fns[key] = sch, fn
        else:
            sch, fn = prepare(d["name"], http_tools.schema(d), http_tools.make_http_tool(d, ctx, client),
                              bool(d.get("confirm")), budget=True)
            schemas[d["name"]], fns[d["name"]] = sch, fn
    return list(schemas.values()), fns
