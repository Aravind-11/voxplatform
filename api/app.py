"""HTTP/WebSocket API. Auth: X-API-Key (tenant), X-Admin-Key (tenant provisioning)."""
from __future__ import annotations

import hmac
import time
from dataclasses import dataclass
from typing import Any, Optional, Union
from xml.sax.saxutils import quoteattr

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

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


class AgentIn(BaseModel):
    name: str
    systemPrompt: str = ""
    voice: str = "af_heart"
    greeting: str = Field("", max_length=500)       # spoken first when the call connects; empty = wait for the caller
    tools: list[Union[str, dict[str, Any]]] = Field(default_factory=list)


class AgentPatch(BaseModel):
    name: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    greeting: Optional[str] = Field(None, max_length=500)
    tools: Optional[list[Union[str, dict[str, Any]]]] = None


class CallIn(BaseModel):
    agentId: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    greeting: Optional[str] = Field(None, max_length=500)
    tools: Optional[list[Union[str, dict[str, Any]]]] = None
    metadata: dict = Field(default_factory=dict)


class WebhookIn(BaseModel):
    url: str


class TenantIn(BaseModel):
    name: str
    concurrencyLimit: Optional[int] = None


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


class _Reject(Exception):
    def __init__(self, code: int, reason: str):
        self.code, self.reason = code, reason


def create_app(store: Store, runtime: Runtime, limiter: security.ConcurrencyLimiter,
               dispatcher: Optional[security.WebhookDispatcher], settings: Settings) -> FastAPI:
    app = FastAPI(title="voxplatform")
    app.state.store, app.state.limiter = store, limiter

    def tenant(x_api_key: str = Header(default="")) -> dict:
        t = store.tenant_by_key(security.hash_key(x_api_key)) if x_api_key else None
        if not t:
            raise HTTPException(401, "invalid or missing API key")
        return t

    def check_tools(specs) -> None:
        try:
            toolmod.validate_specs(specs or [])
        except toolmod.ToolConfigError as e:
            raise HTTPException(422, str(e)) from e

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

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "active_calls": limiter.total_active()}

    # ------------------------------------------------------------- admin
    @app.post("/admin/tenants")
    async def create_tenant(body: TenantIn, x_admin_key: str = Header(default="")):
        if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(403, "admin key required")
        tid = store.create_tenant(body.name, body.concurrencyLimit)
        key = security.generate_key()
        store.add_key(tid, security.hash_key(key), "initial")
        return {"tenantId": tid, "apiKey": key}   # key is shown once; only its hash is stored

    @app.put("/v1/webhook")
    async def set_webhook(body: WebhookIn, t: dict = Depends(tenant)):
        secret = security.generate_key()
        store.set_webhook(t["id"], body.url, secret)
        return {"url": body.url, "secret": secret}

    # ------------------------------------------------------------ agents
    @app.post("/v1/agents")
    async def create_agent(body: AgentIn, t: dict = Depends(tenant)):
        check_tools(body.tools)
        return redacted(store.create_agent(t["id"], body.name, body.model_dump(exclude={"name"})))

    @app.patch("/v1/agents/{agent_id}")
    async def update_agent(agent_id: str, body: AgentPatch, t: dict = Depends(tenant)):
        """Change an agent in place (same id, so a phone number's webhook keeps working). Only given fields change."""
        cur = store.get_agent(t["id"], agent_id)
        if not cur:
            raise HTTPException(404, "agent not found")
        if body.tools is not None:
            check_tools(body.tools)
        cfg = dict(cur["config"])
        for k in ("systemPrompt", "voice", "greeting", "tools"):
            v = getattr(body, k)
            if v is not None:
                cfg[k] = v
        return redacted(store.update_agent(t["id"], agent_id, body.name, cfg))

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
        for k in ("systemPrompt", "voice", "greeting", "tools"):
            v = getattr(body, k)
            if v is not None:
                if k == "tools":
                    check_tools(v)
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

    @app.get("/v1/calls/{call_id}/messages")
    async def call_messages(call_id: str, t: dict = Depends(tenant)):
        if not store.get_call(t["id"], call_id):
            raise HTTPException(404, "call not found")
        return store.messages(call_id)

    @app.get("/v1/usage")
    async def usage(t: dict = Depends(tenant)):
        calls = store.list_calls(t["id"], limit=10_000)
        secs = sum((c["ended"] - c["joined"]) for c in calls if c["joined"] and c["ended"])
        return {"calls": len(calls), "minutes": round(secs / 60, 3), "active": limiter.active(t["id"])}

    # ------------------------------------------------------ call lifecycle
    def admit(call: dict) -> dict:
        """Common admission: status check + per-tenant concurrency. Raises _Reject."""
        if call["status"] != "created":
            raise _Reject(4401, "call not joinable")
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
            await BrowserTransport(ws, factory).run()
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
    return app
