"""Built-in tools: call control (hangUp, transfer, leaveVoicemail, playDtmf), knowledge-base search (queryCorpus) and
client tools (run in the caller's browser). Call-control actions take effect only after the agent has finished its
spoken reply (see core.session.ToolResult)."""
from __future__ import annotations

import re
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from core.session import ToolResult
from .http_tools import NAME_RE, ToolConfigError
from .writebehind import to_thread

E164 = re.compile(r"^\+[1-9]\d{6,14}$")
BUILTINS = {"hangUp", "transfer", "leaveVoicemail", "playDtmf", "queryCorpus", "client"}
DTMF_RE = re.compile(r"^[0-9*#A-Da-dwW]{1,32}$")


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
        mode = str(defn.get("mode", "cold"))
        if mode not in ("cold", "warm"):
            raise ToolConfigError("transfer mode must be 'cold' or 'warm'")
        return {"builtin": "transfer", "targets": dict(targets), "mode": mode,
                "description": str(defn.get("description") or "Transfer the caller to a human. Tell them you are transferring first.")[:1000]}
    if kind == "leaveVoicemail":
        msg = str(defn.get("message") or "").strip()
        if not msg or len(msg) > 600:
            raise ToolConfigError("leaveVoicemail needs 'message' (1-600 chars)")
        return {"builtin": "leaveVoicemail", "message": msg}
    if kind == "playDtmf":
        return {"builtin": "playDtmf"}
    if kind == "queryCorpus":
        corpus = str(defn.get("corpus") or "")
        if not NAME_RE.match(corpus):
            raise ToolConfigError("queryCorpus needs 'corpus': the name of a knowledge base")
        top_k = int(defn.get("topK", 3))
        if not 1 <= top_k <= 8:
            raise ToolConfigError("queryCorpus topK must be 1-8")
        return {"builtin": "queryCorpus", "corpus": corpus, "topK": top_k,
                "description": str(defn.get("description") or f"Search the {corpus} knowledge base for facts before answering questions about it.")[:1000]}
    if kind == "client":
        name = str(defn.get("name") or "")
        if not NAME_RE.match(name):
            raise ToolConfigError(f"client tool name {name!r} must match {NAME_RE.pattern}")
        desc = str(defn.get("description") or "").strip()
        params = defn.get("parameters") or {"type": "object", "properties": {}}
        if not desc or len(desc) > 1000:
            raise ToolConfigError(f"client tool {name}: description is required (max 1000 chars)")
        if not isinstance(params, dict) or params.get("type") != "object" or not isinstance(params.get("properties", {}), dict):
            raise ToolConfigError(f"client tool {name}: parameters must be a JSON Schema object")
        timeout = float(defn.get("timeoutS", 10))
        if not 0.5 <= timeout <= 60:
            raise ToolConfigError(f"client tool {name}: timeoutS must be between 0.5 and 60")
        return {"builtin": "client", "name": name, "description": desc, "parameters": params, "timeoutS": timeout}
    raise ToolConfigError(f"unknown builtin {kind!r}; available: {sorted(BUILTINS)}")


def schema(defn: dict) -> dict:
    kind = defn["builtin"]
    if kind == "leaveVoicemail":
        return {"type": "function", "function": {
            "name": "leaveVoicemail", "parameters": {"type": "object", "properties": {}},
            "description": "Call this ONLY when you have reached an answering machine or voicemail greeting (a recorded "
                           "message, 'leave a message after the tone'). It speaks the prepared message and hangs up."}}
    if kind == "playDtmf":
        return {"type": "function", "function": {
            "name": "playDtmf", "description": "Press keypad digits on the call, e.g. to navigate a phone menu or enter an extension.",
            "parameters": {"type": "object", "properties": {"digits": {
                "type": "string", "description": "Digits to press: 0-9 * # (w = half-second pause)"}}, "required": ["digits"]}}}
    if kind == "queryCorpus":
        return {"type": "function", "function": {
            "name": "queryCorpus", "description": defn["description"],
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What to look up, in a few keywords"}},
                           "required": ["query"]}}}
    if kind == "client":
        return {"type": "function", "function": {"name": defn["name"], "description": defn["description"],
                                                  "parameters": defn["parameters"]}}
    if defn["builtin"] == "hangUp":
        return {"type": "function", "function": {
            "name": "hangUp", "description": "End the call. Say a short goodbye in the same reply; the line drops after you finish speaking.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}}}
    return {"type": "function", "function": {
        "name": "transfer", "description": defn["description"],
        "parameters": {"type": "object", "properties": {"to": {"type": "string", "enum": sorted(defn["targets"])}},
                       "required": ["to"]}}}


def make(defn: dict, ctx, client):
    kind = defn["builtin"]
    if kind == "leaveVoicemail":
        async def voicemail(args: dict) -> Any:
            if ctx.voicemail is None:
                return {"error": "voicemail is only available on live calls"}

            async def do() -> None:
                await ctx.voicemail(defn["message"])
            return ToolResult({"ok": True, "note": "message will be played, then the call ends"}, after=do, strict=True)
        return voicemail
    if kind == "playDtmf":
        async def play(args: dict) -> Any:
            digits = str((args or {}).get("digits", ""))
            if not DTMF_RE.match(digits):
                return {"error": "digits must be 1-32 characters from 0-9 * # A-D w"}
            if ctx.send_dtmf is None:
                return {"error": "playDtmf is only available on phone calls"}
            await ctx.send_dtmf(digits)
            return {"pressed": digits}
        return play
    if kind == "queryCorpus":
        from core.bm25 import BM25
        cache: dict = {}

        async def query(args: dict) -> Any:
            if ctx.docs is None:
                return {"error": "knowledge base unavailable"}
            version, docs = await to_thread(ctx.docs, defn["corpus"])
            if version is None:
                return {"error": f"knowledge base {defn['corpus']!r} not found"}
            if cache.get("v") != version:
                cache["v"], cache["idx"] = version, BM25(docs)
            q, mode = str((args or {}).get("query", "")), "keyword"
            hits = None
            if getattr(ctx, "embedder", None) is not None:
                try:
                    from core.embed import hybrid_search
                    hits, mode = await hybrid_search(cache["idx"], ctx.embedder, q, defn["topK"]), "hybrid"
                except Exception:  # noqa: BLE001  (embedding server down: keyword search still answers)
                    hits = None
            if hits is None:
                hits = cache["idx"].search(q, defn["topK"])
            return {"results": hits, "mode": mode} if hits else {"results": [], "note": "nothing relevant found; say you don't know"}
        return query
    if kind == "client":
        async def client_tool(args: dict) -> Any:
            if ctx.client_call is None:
                return {"error": "client tools are only available on browser (WebSocket) calls"}
            return await ctx.client_call(defn["name"], args or {}, defn["timeoutS"])
        return client_tool
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

        if defn.get("mode") == "warm":
            if not ctx.base_url or not ctx.call_id:
                return {"error": "warm transfer needs the public URL configured"}
            # Warm: the person who picks up first hears a short summary of what the caller wanted (<Number url=...>),
            # and if nobody answers the caller is returned to the agent (the <Dial action=...> callback).
            twiml = (f'<Response><Dial timeout="25" action={quoteattr(f"{ctx.base_url}/twilio/transfer-result/{ctx.call_id}?to={to}")}>'
                     f'<Number url={quoteattr(f"{ctx.base_url}/twilio/whisper/{ctx.call_id}")}>{escape(number)}</Number></Dial></Response>')
        else:
            twiml = f"<Response><Dial>{escape(number)}</Dial></Response>"

        async def do() -> None:
            r = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{ctx.twilio['sid']}/Calls/{ctx.call_sid}.json",
                auth=(ctx.twilio["sid"], ctx.twilio["token"]), timeout=10,
                data={"Twiml": twiml})
            if r.status_code >= 300:
                raise RuntimeError(f"twilio transfer failed: HTTP {r.status_code}")
        return ToolResult({"transferring": to}, after=do, strict=True)
    return transfer
