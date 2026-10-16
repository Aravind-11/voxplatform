"""HTTP/WebSocket API. Auth: X-API-Key (tenant), X-Admin-Key (tenant provisioning)."""
from __future__ import annotations

import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union
from xml.sax.saxutils import quoteattr

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from core.metrics import METRICS
from core.tts import VOICES
from transports.browser import BrowserTransport
from transports.twilio import TwilioTransport
from . import security, tools as toolmod
from .runtime import Runtime
from .store import Store


@dataclass
class Settings:
    admin_key: str = ""
    public_base_url: str = "http://localhost:8080"
    twilio_auth_token: str = ""
    allow_unsigned_twilio: bool = False
    default_concurrency: int = 10
    twilio_account_sid: str = ""
    twilio_number: str = ""            # default caller id for outbound calls (a number you own on Twilio)
    metrics_token: str = ""            # bearer token for /metrics (the admin key also works)
    record_dir: str = ""
    outbound_per_day: int = 50         # per-tenant cap on outbound calls in 24 h
    outbound_prefixes: tuple = ()      # e.g. ("+1",) restricts dialing to these country prefixes; empty = any
    static_dir: str = os.path.join(os.path.dirname(__file__), "static")


class AgentIn(BaseModel):
    name: str
    systemPrompt: str = ""
    voice: str = "af_heart"
    greeting: str = Field("", max_length=500)       # spoken first when the call connects; empty = wait for the caller
    tools: list[Union[str, dict[str, Any]]] = Field(default_factory=list)
    speed: float = Field(1.0, ge=0.7, le=1.3)
    record: bool = False                            # stereo recording of calls; you need the caller's consent


class AgentPatch(BaseModel):
    name: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    greeting: Optional[str] = Field(None, max_length=500)
    tools: Optional[list[Union[str, dict[str, Any]]]] = None
    speed: Optional[float] = Field(None, ge=0.7, le=1.3)
    record: Optional[bool] = None


class CallIn(BaseModel):
    agentId: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    greeting: Optional[str] = Field(None, max_length=500)
    tools: Optional[list[Union[str, dict[str, Any]]]] = None
    speed: Optional[float] = Field(None, ge=0.7, le=1.3)
    record: Optional[bool] = None
    metadata: dict = Field(default_factory=dict)


class OutboundIn(BaseModel):
    agentId: str
    to: str
    fromNumber: Optional[str] = None
    record: Optional[bool] = None
    voicemailMessage: Optional[str] = Field(None, max_length=600)   # if an answering machine picks up, leave this
    metadata: dict = Field(default_factory=dict)


class ToolIn(BaseModel):
    spec: Union[str, dict[str, Any]]


class KeyIn(BaseModel):
    label: str = ""


class CorpusIn(BaseModel):
    name: str


class DocIn(BaseModel):
    title: str = Field(max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class WebhookIn(BaseModel):
    url: str


class TenantIn(BaseModel):
    name: str
    concurrencyLimit: Optional[int] = None
    ratePerMinute: Optional[float] = Field(None, ge=0, le=1000)       # your internal/chargeback rate, used by /v1/usage


class TenantPatch(BaseModel):
    concurrencyLimit: Optional[int] = Field(None, ge=1, le=100_000)
    ratePerMinute: Optional[float] = Field(None, ge=0, le=1000)


class _TwilioWS:
    """Adapts Starlette's WebSocket to the transport's receive_text/send_text, mapping disconnects to EOFError."""

    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def receive_text(self) -> str:
        try:
            return await self.ws.receive_text()
        except WebSocketDisconnect as e:
            raise EOFError from e

    async def send_text(self, s: str) -> None:
        await self.ws.send_text(s)

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001  (already closed)
            pass


class _BrowserWS:
    """Starlette WebSocket -> the browser transport's interface (binary audio frames and JSON text frames)."""

    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def iter_messages(self):
        while True:
            try:
                m = await self.ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                return
            if m["type"] == "websocket.disconnect":
                return
            if m.get("bytes") is not None:
                yield "bytes", m["bytes"]
            elif m.get("text") is not None:
                yield "text", m["text"]

    async def send_bytes(self, b: bytes) -> None:
        await self.ws.send_bytes(b)

    async def send_text(self, s: str) -> None:
        await self.ws.send_text(s)

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass


E164 = re.compile(r"^\+[1-9]\d{6,14}$")


class _Reject(Exception):
    def __init__(self, code: int, reason: str):
        self.code, self.reason = code, reason


def create_app(store: Store, runtime: Runtime, limiter: security.ConcurrencyLimiter,
               dispatcher: Optional[security.WebhookDispatcher], settings: Settings) -> FastAPI:
    app = FastAPI(title="voxplatform")
    app.state.store, app.state.limiter = store, limiter
    METRICS.gauge("vox_active_calls", limiter.total_active)

    @app.on_event("startup")
    async def _start_heartbeat():
        if hasattr(limiter, "heartbeat"):             # Redis limiter: keep this node's calls alive
            import asyncio
            app.state.heartbeat = asyncio.create_task(limiter.heartbeat())

    def tenant(x_api_key: str = Header(default="")) -> dict:
        t = store.tenant_by_key(security.hash_key(x_api_key)) if x_api_key else None
        if not t:
            raise HTTPException(401, "invalid or missing API key")
        return t

    def check_tools(specs, tenant_id: str) -> None:
        try:
            resolved = toolmod.resolve_library(specs or [], lambda n: (store.get_tool(tenant_id, n) or {}).get("spec"))
            toolmod.validate_specs(resolved)
        except toolmod.ToolConfigError as e:
            raise HTTPException(422, str(e)) from e

    def check_voice(voice) -> None:
        if voice is not None and voice not in VOICES:
            raise HTTPException(422, f"unknown voice {voice!r}; see GET /v1/voices")

    def log(t: dict, action: str, target: str = "", **detail) -> None:
        store.audit(t["id"], t.get("key_id") or "admin", action, target, detail)

    def redacted(agent: dict) -> dict:
        """Tool credentials are write-only: header values never come back out of the API."""
        cfg = dict(agent["config"])
        cfg["tools"] = toolmod.redact_specs(cfg.get("tools", []))
        return {**agent, "config": cfg}

    def public_call(c: dict, with_join: bool = False) -> dict:
        d = {k: c[k] for k in ("id", "agent_id", "medium", "status", "created", "joined", "ended", "end_reason", "metadata")}
        if with_join and c["medium"] == "webSocket":
            ws = settings.public_base_url.replace("http", "ws", 1)
            d["joinUrl"] = f"{ws}/ws/calls/{c['id']}?token={c['join_token']}"
        return d

    def webhook(t: dict, type_: str, call: dict) -> None:
        if dispatcher and t.get("webhook_url"):
            dispatcher.spawn(t["webhook_url"], t["webhook_secret"],
                             {"type": type_, "t": time.time(), "call": public_call(call)})

    app.state.draining = False

    @app.get("/healthz")
    async def healthz():
        """Readiness: 503 while draining so the load balancer stops sending new calls to this node."""
        body = {"ok": not app.state.draining, "draining": app.state.draining, "active_calls": limiter.total_active()}
        return Response(json.dumps(body), status_code=503 if app.state.draining else 200, media_type="application/json")

    def admin_only(x_admin_key: str) -> None:
        if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(403, "admin key required")

    @app.post("/admin/drain")
    async def drain(x_admin_key: str = Header(default="")):
        """Stop taking new calls (live ones continue). Call from a pod preStop hook, then wait for active_calls == 0."""
        admin_only(x_admin_key)
        app.state.draining = True
        return {"draining": True, "active_calls": limiter.total_active()}

    @app.post("/admin/undrain")
    async def undrain(x_admin_key: str = Header(default="")):
        admin_only(x_admin_key)
        app.state.draining = False
        return {"draining": False}

    # ------------------------------------------------------------- admin
    @app.post("/admin/tenants")
    async def create_tenant(body: TenantIn, x_admin_key: str = Header(default="")):
        if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(403, "admin key required")
        tid = store.create_tenant(body.name, body.concurrencyLimit)
        if body.ratePerMinute is not None:
            store.update_tenant(tid, rate_per_minute=body.ratePerMinute)
        key = security.generate_key()
        store.add_key(tid, security.hash_key(key), "initial")
        return {"tenantId": tid, "apiKey": key}   # key is shown once; only its hash is stored

    @app.patch("/admin/tenants/{tenant_id}")
    async def patch_tenant(tenant_id: str, body: TenantPatch, x_admin_key: str = Header(default="")):
        admin_only(x_admin_key)
        t = store.update_tenant(tenant_id, body.concurrencyLimit, body.ratePerMinute)
        if not t:
            raise HTTPException(404, "tenant not found")
        return {"tenantId": tenant_id, "concurrencyLimit": t["concurrency_limit"], "ratePerMinute": t.get("rate_per_minute")}

    @app.put("/v1/webhook")
    async def set_webhook(body: WebhookIn, t: dict = Depends(tenant)):
        secret = security.generate_key()
        store.set_webhook(t["id"], body.url, secret)
        return {"url": body.url, "secret": secret}

    # ------------------------------------------------------------ agents
    @app.post("/v1/agents")
    async def create_agent(body: AgentIn, t: dict = Depends(tenant)):
        check_tools(body.tools, t["id"])
        check_voice(body.voice)
        a = store.create_agent(t["id"], body.name, body.model_dump(exclude={"name"}))
        log(t, "agent.create", a["id"], name=body.name)
        return redacted(a)

    @app.patch("/v1/agents/{agent_id}")
    async def update_agent(agent_id: str, body: AgentPatch, t: dict = Depends(tenant)):
        """Change an agent in place (same id, so a phone number's webhook keeps working). Only given fields change."""
        cur = store.get_agent(t["id"], agent_id)
        if not cur:
            raise HTTPException(404, "agent not found")
        if body.tools is not None:
            check_tools(body.tools, t["id"])
        check_voice(body.voice)
        cfg = dict(cur["config"])
        changed = []
        for k in ("systemPrompt", "voice", "greeting", "tools", "speed", "record"):
            v = getattr(body, k)
            if v is not None:
                cfg[k] = v
                changed.append(k)
        log(t, "agent.update", agent_id, fields=changed)
        return redacted(store.update_agent(t["id"], agent_id, body.name, cfg))

    @app.delete("/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str, t: dict = Depends(tenant)):
        if not store.delete_agent(t["id"], agent_id):
            raise HTTPException(404, "agent not found")
        log(t, "agent.delete", agent_id)
        return {"deleted": agent_id}

    @app.get("/v1/agents")
    async def list_agents(t: dict = Depends(tenant)):
        return [redacted(a) for a in store.list_agents(t["id"])]

    # ------------------------------------------------------------- calls
    @app.post("/v1/calls")
    async def create_call(body: CallIn, t: dict = Depends(tenant)):
        base: dict = {}
        if body.agentId:
            agent = store.get_agent(t["id"], body.agentId)
            if not agent:
                raise HTTPException(404, "agent not found")
            base = dict(agent["config"])
        for k in ("systemPrompt", "voice", "greeting", "tools", "speed", "record"):
            v = getattr(body, k)
            if v is not None:
                if k == "tools":
                    check_tools(v, t["id"])
                if k == "voice":
                    check_voice(v)
                base[k] = v
        call = store.create_call(t["id"], body.agentId, "webSocket", base, body.metadata)
        return public_call(call, with_join=True)

    @app.get("/v1/calls")
    async def list_calls(t: dict = Depends(tenant)):
        return [public_call(c) for c in store.list_calls(t["id"])]

    @app.get("/v1/calls/{call_id}")
    async def get_call(call_id: str, t: dict = Depends(tenant)):
        c = store.get_call(t["id"], call_id)
        if not c:
            raise HTTPException(404, "call not found")
        return public_call(c)

    @app.delete("/v1/calls/{call_id}")
    async def delete_call(call_id: str, t: dict = Depends(tenant)):
        """Erases a call: its record, transcript and recording file (for deletion requests and retention policies)."""
        c = store.get_call(t["id"], call_id)
        if not c:
            raise HTTPException(404, "call not found")
        if c["status"] == "active":
            raise HTTPException(409, "call is still active")
        d = settings.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        if d:
            for p in (os.path.join(d, f"{call_id}.wav"),) + tuple(
                    os.path.join(d, "utterances", f) for f in (os.listdir(os.path.join(d, "utterances")) if os.path.isdir(os.path.join(d, "utterances")) else [])
                    if f.startswith(call_id + "-")):
                if os.path.isfile(p):
                    os.remove(p)
        store.delete_call(t["id"], call_id)
        log(t, "call.delete", call_id)
        return {"deleted": call_id}

    @app.get("/v1/calls/{call_id}/messages")
    async def call_messages(call_id: str, t: dict = Depends(tenant)):
        if not store.get_call(t["id"], call_id):
            raise HTTPException(404, "call not found")
        return store.messages(call_id)

    @app.get("/v1/usage")
    async def usage(days: int = 30, t: dict = Depends(tenant)):
        u = store.usage(t["id"], max(1, min(days, 365)))
        rate = t.get("rate_per_minute")
        if rate is not None:
            u["ratePerMinute"], u["cost"] = rate, round(u["minutes"] * rate, 4)
            for row in u["byDay"] + u["byAgent"]:
                row["cost"] = round(row["minutes"] * rate, 4)
        return {**u, "active": limiter.active(t["id"])}

    @app.get("/v1/usage.csv")
    async def usage_csv(days: int = 30, t: dict = Depends(tenant)):
        """Per-day usage as CSV for invoicing: day, calls, minutes, cost (cost blank until a rate is set)."""
        u = store.usage(t["id"], max(1, min(days, 365)))
        rate = t.get("rate_per_minute")
        rows = ["day,calls,minutes,cost"] + [f"{d['key']},{d['calls']},{d['minutes']},{'' if rate is None else round(d['minutes'] * rate, 4)}"
                                            for d in u["byDay"]]
        return PlainTextResponse("\n".join(rows) + "\n", media_type="text/csv")

    @app.get("/v1/me")
    async def me(t: dict = Depends(tenant)):
        return {"tenantId": t["id"], "name": t["name"], "keyId": t.get("key_id"),
                "concurrencyLimit": t["concurrency_limit"] or settings.default_concurrency}

    @app.get("/v1/voices")
    async def voices(t: dict = Depends(tenant)):
        return {"voices": VOICES, "speed": {"min": 0.7, "max": 1.3, "default": 1.0}}

    # ------------------------------------------------------- recordings
    @app.get("/v1/calls/{call_id}/recording")
    async def recording(call_id: str, t: dict = Depends(tenant)):
        if not store.get_call(t["id"], call_id):
            raise HTTPException(404, "call not found")
        d = settings.record_dir or os.environ.get("VOX_RECORD_DIR", "")
        path = os.path.join(d, f"{call_id}.wav") if d else ""
        if not path or not os.path.isfile(path):
            raise HTTPException(404, "no recording for this call (set record=true on the agent and VOX_RECORD_DIR)")
        log(t, "recording.download", call_id)
        return FileResponse(path, media_type="audio/wav", filename=f"{call_id}.wav")

    # ------------------------------------------------------------- keys
    @app.get("/v1/keys")
    async def list_keys(t: dict = Depends(tenant)):
        return [{**k, "current": k["id"] == t.get("key_id")} for k in store.list_keys(t["id"])]

    @app.post("/v1/keys")
    async def create_key(body: KeyIn, t: dict = Depends(tenant)):
        key = security.generate_key()
        kid = store.add_key(t["id"], security.hash_key(key), body.label[:100])
        log(t, "key.create", kid, label=body.label[:100])
        return {"id": kid, "apiKey": key}            # shown once; only the hash is stored

    @app.delete("/v1/keys/{key_id}")
    async def revoke_key(key_id: str, t: dict = Depends(tenant)):
        if store.active_key_count(t["id"]) <= 1:
            raise HTTPException(409, "cannot revoke the last active key; create another first")
        if not store.revoke_key(t["id"], key_id):
            raise HTTPException(404, "key not found or already revoked")
        log(t, "key.revoke", key_id)
        return {"revoked": key_id}

    @app.post("/v1/keys/{key_id}/rotate")
    async def rotate_key(key_id: str, t: dict = Depends(tenant)):
        old = next((k for k in store.list_keys(t["id"]) if k["id"] == key_id and k["revoked"] is None), None)
        if not old:
            raise HTTPException(404, "key not found or already revoked")
        key = security.generate_key()
        kid = store.add_key(t["id"], security.hash_key(key), old["label"])
        store.revoke_key(t["id"], key_id)
        log(t, "key.rotate", key_id, newId=kid)
        return {"id": kid, "apiKey": key, "revoked": key_id}

    @app.get("/v1/audit")
    async def audit(limit: int = 100, before: Optional[int] = None, t: dict = Depends(tenant)):
        return store.list_audit(t["id"], limit, before)

    # ------------------------------------------------------ tool library
    def lib_validate(name: str, spec) -> Any:
        if not toolmod.http_tools.NAME_RE.match(name):
            raise HTTPException(422, "tool name must match [A-Za-z_][A-Za-z0-9_-]{0,63}")
        if isinstance(spec, dict) and "library" in spec:
            raise HTTPException(422, "a library tool cannot reference another library tool")
        if isinstance(spec, dict) and spec.get("name") not in (None, name):
            raise HTTPException(422, "spec.name must equal the name in the URL")
        try:
            toolmod.validate_specs([spec])
        except toolmod.ToolConfigError as e:
            raise HTTPException(422, str(e)) from e
        return spec

    @app.put("/v1/tools/{name}")
    async def put_tool(name: str, body: ToolIn, t: dict = Depends(tenant)):
        spec = lib_validate(name, body.spec)
        if len(store.list_tools(t["id"])) >= 200 and not store.get_tool(t["id"], name):
            raise HTTPException(409, "tool library is full (200)")
        store.put_tool(t["id"], name, spec)
        log(t, "tool.put", name)
        return {"name": name, "spec": toolmod.redact_specs([spec])[0]}

    @app.get("/v1/tools")
    async def list_tools(t: dict = Depends(tenant)):
        return [{"name": x["name"], "spec": toolmod.redact_specs([x["spec"]])[0]} for x in store.list_tools(t["id"])]

    @app.get("/v1/tools/{name}")
    async def get_tool(name: str, t: dict = Depends(tenant)):
        x = store.get_tool(t["id"], name)
        if not x:
            raise HTTPException(404, "tool not found")
        return {"name": name, "spec": toolmod.redact_specs([x["spec"]])[0]}

    @app.delete("/v1/tools/{name}")
    async def delete_tool(name: str, t: dict = Depends(tenant)):
        users = [a["name"] for a in store.list_agents(t["id"])
                 if any(isinstance(s, dict) and s.get("library") == name for s in a["config"].get("tools", []))]
        if users:
            raise HTTPException(409, f"tool is used by agents: {users}")
        if not store.delete_tool(t["id"], name):
            raise HTTPException(404, "tool not found")
        log(t, "tool.delete", name)
        return {"deleted": name}

    # --------------------------------------------------- knowledge bases
    def corpus_or_404(t: dict, name: str) -> dict:
        c = store.get_corpus(t["id"], name)
        if not c:
            raise HTTPException(404, "knowledge base not found")
        return c

    @app.post("/v1/corpora")
    async def create_corpus(body: CorpusIn, t: dict = Depends(tenant)):
        if not toolmod.http_tools.NAME_RE.match(body.name):
            raise HTTPException(422, "name must match [A-Za-z_][A-Za-z0-9_-]{0,63}")
        c = store.create_corpus(t["id"], body.name)
        if c is None:
            raise HTTPException(409, "a knowledge base with that name exists")
        log(t, "corpus.create", body.name)
        return c

    @app.get("/v1/corpora")
    async def list_corpora(t: dict = Depends(tenant)):
        return store.list_corpora(t["id"])

    @app.delete("/v1/corpora/{name}")
    async def delete_corpus(name: str, t: dict = Depends(tenant)):
        if not store.delete_corpus(t["id"], name):
            raise HTTPException(404, "knowledge base not found")
        log(t, "corpus.delete", name)
        return {"deleted": name}

    @app.post("/v1/corpora/{name}/documents")
    async def add_doc(name: str, body: DocIn, t: dict = Depends(tenant)):
        c = corpus_or_404(t, name)
        if len(store.list_docs(t["id"], c["id"])) >= 1000:
            raise HTTPException(409, "knowledge base is full (1000 documents)")
        d = store.add_doc(t["id"], c["id"], body.title, body.text)
        log(t, "doc.add", f"{name}/{d['id']}", title=body.title)
        return d

    @app.get("/v1/corpora/{name}/documents")
    async def list_docs(name: str, t: dict = Depends(tenant)):
        return store.list_docs(t["id"], corpus_or_404(t, name)["id"])

    @app.delete("/v1/corpora/{name}/documents/{doc_id}")
    async def delete_doc(name: str, doc_id: str, t: dict = Depends(tenant)):
        if not store.delete_doc(t["id"], corpus_or_404(t, name)["id"], doc_id):
            raise HTTPException(404, "document not found")
        log(t, "doc.delete", f"{name}/{doc_id}")
        return {"deleted": doc_id}

    @app.post("/v1/corpora/{name}/search")
    async def search_corpus(name: str, body: dict, t: dict = Depends(tenant)):
        """Try a knowledge base without a call: {"query": "...", "topK": 3}."""
        from core.bm25 import BM25
        corpus_or_404(t, name)
        _, docs = store.corpus_docs(t["id"], name)
        return {"results": BM25(docs).search(str(body.get("query", "")), int(body.get("topK", 3)))}

    # ----------------------------------------------------- ops endpoints
    @app.get("/metrics")
    async def metrics(authorization: str = Header(default=""), x_admin_key: str = Header(default="")):
        ok = False
        if settings.metrics_token and hmac.compare_digest(authorization, f"Bearer {settings.metrics_token}"):
            ok = True
        if settings.admin_key and hmac.compare_digest(x_admin_key, settings.admin_key):
            ok = True
        if not ok:
            raise HTTPException(403, "metrics token or admin key required")      # fail closed
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    def page(name: str):
        path = os.path.join(settings.static_dir, name)
        if not os.path.isfile(path):
            raise HTTPException(404, "page not installed")
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/demo")
    async def demo_page():
        return page("demo.html")

    @app.get("/dashboard")
    async def dashboard_page():
        return page("dashboard.html")

    # ------------------------------------------------------ call lifecycle
    def admit(call: dict) -> dict:
        """Common admission: status check + per-tenant concurrency. Raises _Reject."""
        if call["status"] != "created":
            raise _Reject(4401, "call not joinable")
        if app.state.draining:
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason="draining")
            raise _Reject(1013, "node is draining")
        t = store.get_tenant(call["tenant_id"])
        cap = t["concurrency_limit"] or settings.default_concurrency
        if not limiter.acquire(call["tenant_id"], call["id"], cap):
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason="concurrency_limit")
            raise _Reject(1013, "concurrency limit")
        store.update_call(call["id"], status="active", joined=time.time())
        webhook(t, "call.started", store.get_call_any(call["id"]))
        return t

    def finish(call: dict, t: dict, reason: str) -> None:
        limiter.release(call["tenant_id"], call["id"])
        cur = store.get_call_any(call["id"])
        if cur["status"] == "active":
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason=reason)
        METRICS.inc("vox_calls_total", medium=call["medium"], reason=reason)
        try:
            runtime.finalize(call["id"])
        except Exception:  # noqa: BLE001  (a failed recording write must not break call teardown)
            METRICS.inc("vox_errors_total", where="finalize")
        webhook(t, "call.ended", store.get_call_any(call["id"]))

    @app.websocket("/ws/calls/{call_id}")
    async def ws_call(ws: WebSocket, call_id: str, token: str = ""):
        await ws.accept()
        call = store.get_call_any(call_id)
        if not call or not hmac.compare_digest(token, call["join_token"]) or call["medium"] != "webSocket":
            await ws.close(4401)
            return
        try:
            t = admit(call)
        except _Reject as r:
            await ws.close(r.code, r.reason)
            return
        reason = "hangup"
        try:
            async def factory(tr):
                return runtime.build_session(call, tr, forward_events=True)
            await BrowserTransport(_BrowserWS(ws), factory).run()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            reason = "error"
            raise
        finally:
            finish(call, t, reason)

    # ---------------------------------------------------------- telephony
    @app.post("/twilio/voice/{agent_id}")
    async def twilio_voice(agent_id: str, request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        if not settings.allow_unsigned_twilio:
            if not settings.twilio_auth_token:
                raise HTTPException(403, "twilio auth token not configured")   # fail closed
            url = settings.public_base_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
            if not security.validate_twilio(settings.twilio_auth_token, url, form,
                                            request.headers.get("X-Twilio-Signature", "")):
                raise HTTPException(403, "bad twilio signature")
        if app.state.draining:
            raise HTTPException(503, "node is draining")
        agent = store.get_agent_any(agent_id)
        if not agent:
            raise HTTPException(404, "agent not found")
        call = store.create_call(agent["tenant_id"], agent_id, "twilio", agent["config"],
                                 {"from": form.get("From", ""), "to": form.get("To", ""),
                                  "callSid": form.get("CallSid", "")})
        ws = settings.public_base_url.replace("http", "ws", 1)
        twiml = ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
                 f'<Stream url={quoteattr(f"{ws}/ws/twilio/{call["id"]}")}>'
                 f'<Parameter name="token" value={quoteattr(call["join_token"])}/>'
                 '</Stream></Connect></Response>')
        return Response(twiml, media_type="text/xml")

    @app.websocket("/ws/twilio/{call_id}")
    async def ws_twilio(ws: WebSocket, call_id: str):
        await ws.accept()
        call = store.get_call_any(call_id)
        if not call or call["medium"] != "twilio":
            await ws.close(4401)
            return
        state: dict = {}

        async def factory(tr, custom):
            if not hmac.compare_digest(str(custom.get("token", "")), call["join_token"]):
                raise _Reject(4401, "bad token")
            state["t"] = admit(call)
            return runtime.build_session(call, tr)

        try:
            await TwilioTransport(_TwilioWS(ws), factory).run()
        except _Reject as r:
            await ws.close(r.code, r.reason)
        finally:
            if "t" in state:
                finish(call, state["t"], "hangup")

    # ----------------------------------------------------- outbound calls
    def twilio_check(request: Request, form: dict) -> None:
        if settings.allow_unsigned_twilio:
            return
        if not settings.twilio_auth_token:
            raise HTTPException(403, "twilio auth token not configured")
        url = settings.public_base_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        if not security.validate_twilio(settings.twilio_auth_token, url, form, request.headers.get("X-Twilio-Signature", "")):
            raise HTTPException(403, "bad twilio signature")

    @app.post("/v1/calls/outbound")
    async def outbound(body: OutboundIn, t: dict = Depends(tenant)):
        """Places a phone call through your Twilio account and connects the answered call to the agent."""
        if app.state.draining:
            raise HTTPException(503, "node is draining")
        if not (settings.twilio_account_sid and settings.twilio_auth_token):
            raise HTTPException(503, "outbound calling needs TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN on the server")
        if runtime.http is None:
            raise HTTPException(503, "no HTTP client configured")
        if not E164.match(body.to):
            raise HTTPException(422, "'to' must be E.164, e.g. +15551234567")
        if settings.outbound_prefixes and not body.to.startswith(tuple(settings.outbound_prefixes)):
            raise HTTPException(403, f"dialing is restricted to prefixes {list(settings.outbound_prefixes)}")
        caller_id = body.fromNumber or settings.twilio_number
        if not caller_id or not E164.match(caller_id):
            raise HTTPException(422, "no valid caller id: pass fromNumber or set VOX_TWILIO_NUMBER")
        agent = store.get_agent(t["id"], body.agentId)
        if not agent:
            raise HTTPException(404, "agent not found")
        day_ago = time.time() - 86400
        recent = sum(1 for c in store.list_calls(t["id"], limit=10_000)
                     if c["created"] >= day_ago and c["metadata"].get("direction") == "outbound")
        if recent >= settings.outbound_per_day:
            raise HTTPException(429, f"outbound limit reached ({settings.outbound_per_day} per 24 h)")
        cfg = dict(agent["config"])
        if body.record is not None:
            cfg["record"] = body.record
        meta = {**body.metadata, "direction": "outbound", "from": caller_id, "to": body.to}
        if body.voicemailMessage:
            meta["voicemail"] = body.voicemailMessage
        call = store.create_call(t["id"], body.agentId, "twilio", cfg, meta)
        ws = settings.public_base_url.replace("http", "ws", 1)
        twiml = ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
                 f'<Stream url={quoteattr(f"{ws}/ws/twilio/{call["id"]}")}>'
                 f'<Parameter name="token" value={quoteattr(call["join_token"])}/>'
                 '</Stream></Connect></Response>')
        data = {"To": body.to, "From": caller_id, "Twiml": twiml,
                "StatusCallback": f"{settings.public_base_url}/twilio/status/{call['id']}",
                "StatusCallbackEvent": "completed"}
        if body.voicemailMessage:
            data.update({"MachineDetection": "DetectMessageEnd", "AsyncAmd": "true",
                         "AsyncAmdStatusCallback": f"{settings.public_base_url}/twilio/amd/{call['id']}"})
        try:
            r = await runtime.http.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls.json",
                auth=(settings.twilio_account_sid, settings.twilio_auth_token), data=data, timeout=15)
        except Exception as e:  # noqa: BLE001
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason="twilio_unreachable")
            raise HTTPException(502, f"could not reach Twilio: {type(e).__name__}") from e
        if r.status_code >= 300:
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason="twilio_rejected")
            try:
                msg = r.json().get("message", "")
            except ValueError:
                msg = ""
            raise HTTPException(502, f"Twilio refused the call (HTTP {r.status_code}): {msg}")
        sid = r.json().get("sid", "")
        store.update_call(call["id"], metadata={**meta, "callSid": sid})
        log(t, "call.outbound", call["id"], to=body.to)
        return {**public_call(store.get_call_any(call["id"])), "callSid": sid}

    @app.post("/twilio/status/{call_id}")
    async def twilio_status(call_id: str, request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = store.get_call_any(call_id)
        if not call:
            raise HTTPException(404, "call not found")
        status = form.get("CallStatus", "")
        if call["status"] == "created" and status in ("busy", "failed", "no-answer", "canceled", "completed"):
            store.update_call(call_id, status="ended", ended=time.time(), end_reason=status)   # never answered / never connected
        return {"ok": True}

    @app.post("/twilio/amd/{call_id}")
    async def twilio_amd(call_id: str, request: Request):
        """Answering-machine detection result. On a machine, play the prepared message and hang up."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = store.get_call_any(call_id)
        if not call:
            raise HTTPException(404, "call not found")
        answered = form.get("AnsweredBy", "")
        msg = call["metadata"].get("voicemail")
        live = runtime.live.get(call_id)
        METRICS.inc("vox_amd_total", answered_by=answered or "unknown")
        if msg and live and answered.startswith("machine_end"):
            sess, transport = live

            async def go():
                await sess.say_and_wait(msg)
                await transport.end()
            import asyncio
            asyncio.get_running_loop().create_task(go())
            return {"action": "voicemail"}
        return {"action": "none", "answeredBy": answered}

    return app
