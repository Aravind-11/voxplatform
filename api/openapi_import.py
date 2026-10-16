"""Turns an OpenAPI 3 document into agent tools, one per operation, so connecting a company's REST API (a CRM, a
booking system) is configuration, not code.

Each operation becomes an HTTP tool (api/http_tools.py): path parameters are substituted into the URL (percent-encoded),
query parameters go in the query string, and a JSON request body's top-level fields become tool parameters. Credentials
you pass as `headers` are stored write-only and never shown to the model. Operations that change data (anything but GET)
are imported with `confirm: true` by default, so the agent must read the details back and get a yes before acting.

Limits (reported as `skipped`, never silently dropped): OpenAPI 3.x JSON only; only application/json bodies whose top
level is an object; header and cookie parameters are ignored; `$ref`s must point inside the same document;
at most 100 operations per import.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .http_tools import NAME_RE, ToolConfigError, validate

METHODS = ("get", "post", "put", "patch", "delete")
KEEP = ("type", "description", "enum", "format", "items", "minimum", "maximum", "minLength", "maxLength", "pattern",
        "properties", "required", "default")


def _deref(doc: dict, node: Any, depth: int = 0) -> Any:
    if depth > 8:
        return {}
    if isinstance(node, dict) and "$ref" in node:
        ref = str(node["$ref"])
        if not ref.startswith("#/"):
            return {}                                             # external references are not followed
        cur: Any = doc
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(cur, dict) or part not in cur:
                return {}
            cur = cur[part]
        return _deref(doc, cur, depth + 1)
    return node


def _simplify(doc: dict, schema: Any, depth: int = 0) -> dict:
    """A JSON-Schema fragment the model can fill: known keywords only, references resolved, composition flattened."""
    s = _deref(doc, schema, depth)
    if not isinstance(s, dict) or depth > 6:
        return {}
    if "allOf" in s:                                              # merge object parts
        merged: dict = {"type": "object", "properties": {}, "required": []}
        for part in s["allOf"]:
            p = _simplify(doc, part, depth + 1)
            merged["properties"].update(p.get("properties", {}))
            merged["required"] += p.get("required", [])
        return merged
    for key in ("oneOf", "anyOf"):
        if key in s and s[key]:
            return _simplify(doc, s[key][0], depth + 1)           # first alternative; the description carries the rest
    out = {k: v for k, v in s.items() if k in KEEP}
    if "properties" in out:
        out["properties"] = {k: _simplify(doc, v, depth + 1) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _simplify(doc, out["items"], depth + 1)
    if out.get("type") not in (None, "string", "number", "integer", "boolean", "array", "object"):
        out.pop("type", None)
    return out


def _tool_name(op: dict, method: str, path: str, prefix: str) -> str:
    raw = op.get("operationId") or f"{method}_{path}"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    name = (prefix + name)[:64]
    return name if NAME_RE.match(name) else "op_" + name[:60]


def import_spec(doc: dict, *, base_url: Optional[str] = None, headers: Optional[dict] = None,
                include: Optional[list] = None, prefix: str = "", confirm_writes: bool = True,
                timeout_s: float = 8, max_ops: int = 100) -> tuple[list[dict], list[dict]]:
    """Returns (tools, skipped). Raises ToolConfigError for a document that cannot be used at all."""
    if not isinstance(doc, dict) or not str(doc.get("openapi", "")).startswith("3."):
        raise ToolConfigError("only OpenAPI 3.x documents are supported (the 'openapi' field must start with 3.)")
    servers = doc.get("servers") or []
    base = (base_url or (servers[0].get("url") if servers and isinstance(servers[0], dict) else "") or "").rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ToolConfigError("no usable server URL: pass baseUrl or give the document an absolute servers[0].url")
    tools: list[dict] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    for path, item in (doc.get("paths") or {}).items():
        item = _deref(doc, item)
        if not isinstance(item, dict):
            continue
        for method in METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            label = f"{method.upper()} {path}"
            name = _tool_name(op, method, path, prefix)
            if include and name not in include and op.get("operationId") not in include:
                continue
            if len(tools) >= max_ops:
                skipped.append({"operation": label, "reason": f"more than {max_ops} operations; use `include`"})
                continue
            props: dict = {}
            required: list = []
            query: list = []
            problem = ""
            for prm in [_deref(doc, p) for p in (item.get("parameters") or []) + (op.get("parameters") or [])]:
                if not isinstance(prm, dict) or prm.get("in") not in ("path", "query"):
                    continue                                      # header and cookie parameters are not modelled
                pn = prm.get("name", "")
                sch = _simplify(doc, prm.get("schema") or {})
                if prm.get("description") and "description" not in sch:
                    sch["description"] = str(prm["description"])
                props[pn] = sch or {"type": "string"}
                if prm["in"] == "path" or prm.get("required"):
                    required.append(pn)
                if prm["in"] == "query":
                    query.append(pn)
            body = _deref(doc, op.get("requestBody") or {})
            if isinstance(body, dict) and body:
                content = body.get("content") or {}
                js = content.get("application/json")
                if js is None:
                    if body.get("required"):
                        problem = "request body is not application/json"
                else:
                    bs = _simplify(doc, js.get("schema") or {})
                    if bs.get("type") not in ("object", None) or "properties" not in bs:
                        problem = "request body is not an object with named fields"
                    else:
                        clash = set(bs["properties"]) & set(props)
                        if clash:
                            problem = f"body fields {sorted(clash)} clash with path/query parameter names"
                        props.update(bs["properties"])
                        required += [r for r in bs.get("required", []) if r in bs["properties"]]
            if problem:
                skipped.append({"operation": label, "reason": problem})
                continue
            base_name, n = name, 2
            while name in seen:
                name, n = f"{base_name}_{n}", n + 1
            desc = " ".join(x for x in (str(op.get("summary") or "").strip(), str(op.get("description") or "").strip()) if x)
            spec = {"name": name, "description": (desc or label)[:1000],
                    "parameters": {"type": "object", "properties": props, "required": sorted(set(required))},
                    "http": {"url": base + path, "method": method.upper(), "headers": dict(headers or {}),
                             "timeoutS": timeout_s, "query": query},
                    "confirm": bool(confirm_writes and method != "get")}
            try:
                tools.append(validate(spec))
                seen.add(name)
            except ToolConfigError as e:
                skipped.append({"operation": label, "reason": str(e)})
    return tools, skipped
