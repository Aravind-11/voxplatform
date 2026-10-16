"""Tool definitions an agent can carry, and the HTTP tool that calls the tenant's own endpoint.

An agent's `tools` list holds names (demo tools, built-ins such as "hangUp") or definition objects:

  {"name": "lookup_order", "description": "Find an order by its number.",
   "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
   "http": {"url": "https://api.example.com/orders", "method": "POST",
            "headers": {"Authorization": "Bearer sk-..."}, "timeoutS": 8},
   "static": {"tenant": "acme"},            # always sent, never shown to the model, cannot be overridden by it
   "automatic": ["callId", "from"],         # filled in by the platform: callId, from, to, callSid
   "confirm": true}                         # side-effect tool: agent must read details back and get a yes first

Parameter kinds mirror the paid product: dynamic (the model fills them: `parameters`), static, automatic.
A tool's HTTP response goes back to the model. A response header `X-Vox-Response-Type: hang-up`
(`X-Ultravox-Response-Type` is accepted too) ends the call once the agent has finished speaking.

Safety: URLs must be http(s) and resolve only to public addresses (override for local tests with
VOX_TOOL_ALLOW_PRIVATE=1); redirects are not followed; responses are truncated; every tool call has a
timeout. Known limit: the address is checked just before the request, so a DNS-rebinding attacker could
still race it; production should route tool traffic through an egress proxy.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
AUTOMATIC = {"callId", "from", "to", "callSid"}
MAX_RESPONSE_CHARS = 4000
RESPONSE_TYPE_HEADERS = ("x-vox-response-type", "x-ultravox-response-type")


class ToolConfigError(ValueError):
    pass


def _check_url_shape(url: str) -> str:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolConfigError("http.url must be an http(s) URL")
    if u.username or u.password:
        raise ToolConfigError("http.url must not contain credentials; use http.headers")
    return u.hostname


def validate(defn: dict) -> dict:
    """Validates and normalizes an HTTP tool definition. Raises ToolConfigError with a human message."""
    if not isinstance(defn, dict):
        raise ToolConfigError("tool must be a name or an object")
    name = defn.get("name", "")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ToolConfigError(f"tool name {name!r} must match {NAME_RE.pattern}")
    desc = defn.get("description", "")
    if not isinstance(desc, str) or not desc.strip() or len(desc) > 1000:
        raise ToolConfigError(f"tool {name}: description is required (max 1000 chars)")
    http = defn.get("http")
    if not isinstance(http, dict):
        raise ToolConfigError(f"tool {name}: 'http' object is required")
    _check_url_shape(str(http.get("url", "")))
    method = str(http.get("method", "POST")).upper()
    if method not in METHODS:
        raise ToolConfigError(f"tool {name}: method must be one of {sorted(METHODS)}")
    headers = http.get("headers", {}) or {}
    if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ToolConfigError(f"tool {name}: headers must map strings to strings")
    timeout = float(http.get("timeoutS", 8))
    if not 0.5 <= timeout <= 30:
        raise ToolConfigError(f"tool {name}: timeoutS must be between 0.5 and 30")
    params = defn.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(params, dict) or params.get("type") != "object" or not isinstance(params.get("properties", {}), dict):
        raise ToolConfigError(f"tool {name}: parameters must be a JSON Schema object")
    static = defn.get("static", {}) or {}
    if not isinstance(static, dict):
        raise ToolConfigError(f"tool {name}: static must be an object")
    clash = set(static) & set(params.get("properties", {}))
    if clash:
        raise ToolConfigError(f"tool {name}: {sorted(clash)} are both static and model-filled")
    auto = defn.get("automatic", []) or []
    if not isinstance(auto, list) or not set(auto) <= AUTOMATIC:
        raise ToolConfigError(f"tool {name}: automatic must be a subset of {sorted(AUTOMATIC)}")
    return {"name": name, "description": desc.strip(), "parameters": params,
            "http": {"url": http["url"], "method": method, "headers": headers, "timeoutS": timeout},
            "static": static, "automatic": list(auto), "confirm": bool(defn.get("confirm", False))}


def schema(defn: dict) -> dict:
    """What the model sees: name, description and the model-filled parameters only. Never URL, headers or static."""
    return {"type": "function", "function": {"name": defn["name"], "description": defn["description"],
                                              "parameters": defn["parameters"]}}


def redact(defn: dict) -> dict:
    """Copy of a definition that is safe to return from the API (header values hidden)."""
    d = json.loads(json.dumps(defn))
    if isinstance(d.get("http"), dict) and isinstance(d["http"].get("headers"), dict):
        d["http"]["headers"] = {k: "***" for k in d["http"]["headers"]}
    return d


def assert_public(url: str) -> None:
    """Refuses URLs that point inside our network (SSRF): loopback, private, link-local, metadata, etc."""
    if os.environ.get("VOX_TOOL_ALLOW_PRIVATE") == "1":
        return
    host = _check_url_shape(url)
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ToolConfigError(f"cannot resolve {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified):
            raise ToolConfigError(f"{host} resolves to a non-public address")


def make_http_tool(defn: dict, ctx, client) -> Callable[[dict], Awaitable[Any]]:
    """Returns the async callable a Session runs for this tool. `ctx` supplies call facts and `end()`."""
    from core.session import ToolResult
    http, declared = defn["http"], set(defn["parameters"].get("properties", {}))

    async def run(args: dict):
        body = {k: v for k, v in (args or {}).items() if k in declared}     # unknown arguments are dropped
        body.update(defn["static"])                                          # static beats anything the model said
        auto = {"callId": ctx.call_id, "from": ctx.caller, "to": ctx.callee, "callSid": ctx.call_sid}
        body.update({k: auto[k] for k in defn["automatic"]})
        try:
            assert_public(http["url"])
            kw: dict = {"params": {k: v if isinstance(v, (str, int, float)) else json.dumps(v)
                                   for k, v in body.items()}} if http["method"] == "GET" else {"json": body}
            r = await client.request(http["method"], http["url"], headers=http["headers"],
                                     timeout=http["timeoutS"], follow_redirects=False, **kw)
        except ToolConfigError as e:
            return {"error": f"blocked: {e}"}
        except Exception as e:  # noqa: BLE001  (timeouts, connection errors)
            return {"error": f"tool request failed: {type(e).__name__}"}
        text = r.text[:MAX_RESPONSE_CHARS]
        try:
            content: Any = json.loads(text) if "json" in r.headers.get("content-type", "") else text
        except ValueError:
            content = text
        if r.status_code >= 300:            # 3xx too: redirects are never followed
            content = {"error": f"HTTP {r.status_code}", "body": content}
        rt = next((r.headers[h] for h in RESPONSE_TYPE_HEADERS if h in r.headers), "").lower()
        if rt == "hang-up":
            return ToolResult(content, after=ctx.end, strict=True)
        return content

    return run
