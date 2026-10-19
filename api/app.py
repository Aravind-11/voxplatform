"""HTTP/WebSocket API. Auth: X-API-Key (tenant), X-Admin-Key (tenant provisioning)."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
import uuid as uuidlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from xml.sax.saxutils import escape, quoteattr

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from core.metrics import METRICS
from core.redact import redact_text
from core.vad_exec import watch_loop_lag
from core import tts as ttsmod
from core.tts import DEFAULT_VOICE, LANG_NAME, VOICES
from transports.browser import BrowserTransport
from transports.twilio import TwilioTransport
from . import security, tools as toolmod
from . import experiments as expmod
from . import openapi_import, retention, variables as varmod
from .callbus import make_bus
from .callcontrol import AsteriskCallControl, TwilioCallControl, control_from_env
from .runtime import Runtime
from .store import ROLES, Store, new_id, normalize_e164
from .writebehind import to_thread


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
    retention_default_days: int = 0    # 0 = keep forever unless the tenant sets retention_days
    outbound_per_day: int = 50         # per-tenant cap on outbound calls in 24 h
    outbound_prefixes: tuple = ()      # e.g. ("+1",) restricts dialing to these country prefixes; empty = any
    static_dir: str = os.path.join(os.path.dirname(__file__), "static")
    audiosocket_port: int = 0          # >0: listen for Asterisk AudioSocket connections (the SIP path) on this TCP port
    audiosocket_agent: str = ""        # agent id that answers calls arriving on that port when no number is mapped
    audiosocket_host: str = "127.0.0.1"  # bind address; keep it private, AudioSocket has no authentication
    node_max_calls: int = 32           # this process; 0 = no node cap. Checked before the per-tenant cap.
    overflow_url: str = ""             # http(s) only; a full or draining node redirects Twilio here instead of speaking
    sip_token: str = ""                # VOX_SIP_TOKEN; empty disables setup, recover, amd, bind, and transfer (404)
    audiosocket_rate: int = 8000       # VOX_AUDIOSOCKET_RATE. 8000 keeps 320-byte frames; 16000 matches g722/slin16.


def checked_overflow(url: str) -> str:
    """http(s) overflow target, or '' when unset. Other schemes are rejected (they would be an open redirect)."""
    url = (url or "").strip()
    if not url:
        return ""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("VOX_OVERFLOW_URL must be an http:// or https:// URL")
    return url


class AgentIn(BaseModel):
    name: str
    systemPrompt: str = ""
    voice: str = "af_heart"
    language: Optional[str] = None                  # ISO code (en, es, fr, hi, it, ja, pt, zh): the agent speaks and replies in it
    variables: dict[str, Any] = Field(default_factory=dict)   # defaults for {{name}} placeholders in the prompt/greeting
    greeting: str = Field("", max_length=500)       # spoken first when the call connects; empty = wait for the caller
    tools: list[Union[str, dict[str, Any]]] = Field(default_factory=list)
    speed: float = Field(1.0, ge=0.7, le=1.3)
    record: bool = False                            # stereo recording of calls; you need the caller's consent
    experiment: Optional[dict[str, Any]] = None     # optional A/B: {id, variants:[{id, weight, voice?, systemPrompt?, model?}]}
    bargeIn: Literal["high", "normal", "low"] = "normal"  # how easily the caller can talk over the agent
    confirmSlots: bool = False                      # read names, dates, times, numbers and amounts back before acting
    restateSlots: bool = False                      # remind the model that earlier turns are transcripts, not audio
    redact: bool = False                            # strip card, SSN, email, phone, and street address from stored text


class AgentPatch(BaseModel):
    name: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    variables: Optional[dict[str, Any]] = None
    greeting: Optional[str] = Field(None, max_length=500)
    tools: Optional[list[Union[str, dict[str, Any]]]] = None
    speed: Optional[float] = Field(None, ge=0.7, le=1.3)
    record: Optional[bool] = None
    experiment: Optional[dict[str, Any]] = None
    bargeIn: Optional[Literal["high", "normal", "low"]] = None
    confirmSlots: Optional[bool] = None
    restateSlots: Optional[bool] = None
    redact: Optional[bool] = None


class NumberIn(BaseModel):
    agentId: str


class CallIn(BaseModel):
    agentId: Optional[str] = None
    systemPrompt: Optional[str] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    variables: Optional[dict[str, Any]] = None
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
    variables: Optional[dict[str, Any]] = None                      # fills {{name}} in the agent's prompt and greeting
    metadata: dict = Field(default_factory=dict)
    channel: str = "twilio"                                         # twilio (default) or sip (Asterisk ARI)


class ToolIn(BaseModel):
    spec: Union[str, dict[str, Any]]


class OpenApiImportIn(BaseModel):
    spec: dict[str, Any]                                # an OpenAPI 3.x document as JSON
    baseUrl: Optional[str] = None                       # overrides servers[0].url
    headers: dict[str, str] = Field(default_factory=dict)   # credentials sent on every call; stored write-only
    prefix: str = Field("", max_length=20)
    include: Optional[list[str]] = None                 # operationIds to import (default: all, max 100)
    confirmWrites: bool = True                          # non-GET operations require read-back and a yes
    dryRun: bool = False
    overwrite: bool = False


class KeyIn(BaseModel):
    label: str = ""
    role: str = "admin"                 # admin | developer | viewer


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
    retentionDays: Optional[int] = Field(None, ge=0, le=3650)


class TenantPatch(BaseModel):
    concurrencyLimit: Optional[int] = Field(None, ge=1, le=100_000)
    ratePerMinute: Optional[float] = Field(None, ge=0, le=1000)
    retentionDays: Optional[int] = Field(None, ge=0, le=3650)         # erase ended calls older than this; 0 = off


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
# Asterisk sends the UUID as the first AudioSocket frame. Past this, a half-open peer is not a call.
AUDIOSOCKET_UUID_TIMEOUT_S = 5.0
TERMINAL_CALL_STATUS = ("busy", "failed", "no-answer", "canceled", "completed")


def _abort_writer(writer) -> None:
    """Unblock a handler stuck in readexactly. close() alone can leave that read waiting."""
    try:
        transport = getattr(writer, "transport", None)
        if transport is not None:
            transport.abort()
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        writer.close()
    except Exception:  # noqa: BLE001
        pass


class _Reject(Exception):
    def __init__(self, code: int, reason: str):
        self.code, self.reason = code, reason


def create_app(store: Store, runtime: Runtime, limiter: security.ConcurrencyLimiter,
               dispatcher: Optional[security.WebhookDispatcher], settings: Settings) -> FastAPI:
    overflow_url = checked_overflow(settings.overflow_url)
    app = FastAPI(title="voxplatform")
    if not getattr(runtime, "public_url", ""):
        runtime.public_url = settings.public_base_url
    if getattr(runtime, "bus", None) is None:
        runtime.bus = make_bus(os.environ.get("VOX_REDIS_URL", ""))   # tests leave the env unset -> local
    app.state.store, app.state.limiter, app.state.runtime = store, limiter, runtime
    outbound_cap = security.OutboundCap(limiter, store)
    METRICS.gauge("vox_active_calls", limiter.local_active)            # this node's calls
    METRICS.gauge("vox_active_calls_cluster", limiter.total_active)   # every node's calls (Redis scan)

    @app.on_event("startup")
    async def _start_heartbeat():
        if hasattr(limiter, "heartbeat"):             # Redis limiter: keep this node's calls alive
            import asyncio
            app.state.heartbeat = asyncio.create_task(limiter.heartbeat())
        import asyncio
        async def sweep():
            while True:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, lambda: retention.purge_expired(
                        store, settings.record_dir or os.environ.get("VOX_RECORD_DIR", ""),
                        settings.retention_default_days or None, setups=app.state.sip_setups))
                except Exception:  # noqa: BLE001  (try again next hour)
                    METRICS.inc("vox_errors_total", where="retention")
                await asyncio.sleep(3600)
        app.state.retention = asyncio.create_task(sweep())
        # One lag watcher per app. Startup can run again (reload, a second enter); don't stack tasks.
        if getattr(app.state, "loop_lag_task", None) is None:
            app.state.loop_lag_task = asyncio.create_task(watch_loop_lag(METRICS))

    def tenant(x_api_key: str = Header(default="")) -> dict:
        t = store.tenant_by_key(security.hash_key(x_api_key)) if x_api_key else None
        if not t:
            raise HTTPException(401, "invalid or missing API key")
        return t

    def tenant_write(t: dict = Depends(tenant)) -> dict:
        if t.get("role") == "viewer":
            raise HTTPException(403, "this API key is read-only (role: viewer)")
        return t

    def tenant_admin(t: dict = Depends(tenant)) -> dict:
        if t.get("role") not in (None, "admin"):
            raise HTTPException(403, f"this API key cannot do that (role: {t.get('role')}); an admin key is required")
        return t

    def check_tools(specs, tenant_id: str) -> None:
        try:
            resolved = toolmod.resolve_library(specs or [], lambda n: (store.get_tool(tenant_id, n) or {}).get("spec"))
            toolmod.validate_specs(resolved)
        except toolmod.ToolConfigError as e:
            raise HTTPException(422, str(e)) from e

    def tool_warnings(specs, tenant_id: str) -> list[str]:
        """Soft cap: a large tool list is saved, and the response says so. Not a 422."""
        try:
            resolved = toolmod.resolve_library(specs or [], lambda n: (store.get_tool(tenant_id, n) or {}).get("spec"))
            schemas, _fns = toolmod.build(resolved)
        except toolmod.ToolConfigError:
            return []
        warning = toolmod.warn_schema_tokens(schemas)
        return [warning] if warning else []

    def respond_agent(agent: dict, tenant_id: str) -> dict:
        out = redacted(agent)
        warnings = tool_warnings(agent["config"].get("tools") or [], tenant_id)
        if warnings:
            out["warnings"] = warnings
        return out

    def check_variant_voices(experiment, language) -> None:
        """Every variant voice must pass the same voice/language check as the agent voice."""
        if not isinstance(experiment, dict):
            return
        for variant in experiment.get("variants") or []:
            if isinstance(variant, dict) and variant.get("voice"):
                check_voice(variant["voice"], language)

    def check_experiment(experiment: dict, language=None) -> dict:
        try:
            cleaned = expmod.validate_experiment(experiment)
        except expmod.ExperimentError as e:
            raise HTTPException(422, str(e)) from e
        check_variant_voices(cleaned, language)
        return cleaned

    def check_voice(voice, language=None) -> None:
        if voice is not None and voice not in VOICES:
            raise HTTPException(422, f"unknown voice {voice!r}; see GET /v1/voices")
        if language is not None and language not in LANG_NAME:
            raise HTTPException(422, f"unknown language {language!r}; supported: {sorted(LANG_NAME)}")
        if voice is not None:
            why = ttsmod.why_unavailable(ttsmod.lang_code_of(voice))
            if why:
                raise HTTPException(422, f"voice {voice!r} ({ttsmod.language_of(voice)}) is not available here: {why}")
            if language is not None and ttsmod.language_of(voice) != language:
                raise HTTPException(422, f"voice {voice!r} speaks {ttsmod.language_of(voice)}, not {language!r}; "
                                         f"try {DEFAULT_VOICE[language]!r}")

    def call_variables(cfg: dict, supplied, *, strict: bool) -> dict:
        """Validated call-supplied variables. With strict=True (an API-created call) a placeholder that nothing fills is
        a 422, so a typo never reaches a live conversation. Every variant system prompt counts, not only the agent prompt."""
        try:
            given = varmod.validate(supplied)
            varmod.validate(cfg.get("variables") or {})
        except varmod.VariableError as e:
            raise HTTPException(422, str(e)) from e
        if strict:
            texts = [cfg.get("systemPrompt", ""), cfg.get("greeting", "")]
            exp = cfg.get("experiment")
            if isinstance(exp, dict):
                for variant in exp.get("variants") or []:
                    if isinstance(variant, dict) and isinstance(variant.get("systemPrompt"), str):
                        texts.append(variant["systemPrompt"])
            gone = varmod.missing(texts, {**(cfg.get("variables") or {}), **given})
            if gone:
                raise HTTPException(422, f"the agent's prompt, greeting, or a variant uses {{{{{gone[0]}}}}} "
                                         f"(and {len(gone) - 1} more) but no value was given: missing {gone}")
        return given

    def caller_metadata(raw: dict) -> dict:
        """experiment, variant, and promptSuffix are server writes. A caller cannot pin a bucket or extend the prompt."""
        return {k: v for k, v in raw.items() if k not in ("experiment", "variant", "promptSuffix")}

    def pick_voice(cur_voice, cur_lang, new_voice, new_lang, voice_given: bool):
        """Effective (voice, language) after a change. Setting only a language switches to that language's default voice."""
        lang = new_lang if new_lang is not None else cur_lang
        voice = new_voice if voice_given and new_voice is not None else cur_voice
        if new_lang in DEFAULT_VOICE and not voice_given and voice and ttsmod.language_of(voice) != new_lang:
            voice = DEFAULT_VOICE[new_lang]
        return voice, lang

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

    def webhook(t: dict, type_: str, call: dict, **extra) -> None:
        if dispatcher and t.get("webhook_url"):
            dispatcher.spawn(t["webhook_url"], t["webhook_secret"],
                             {"type": type_, "t": time.time(), "call": public_call(call), **extra})

    app.state.draining = False
    app.state.sip_setups = {}          # uuid -> {caller, exten, callee, created}; same rows as sip_setup
    app.state.sip_by_uuid = {}       # AudioSocket UUID -> Asterisk channel uniqueid (POST /sip/bind)
    app.state.sip_by_channel = {}    # Asterisk channel uniqueid -> call id
    app.state.sip_machine_done = {}  # call id -> "voicemail" | "hangup" once that action has finished
    app.state.sip_machine_wait = {}  # call id -> in-flight hangup shared by concurrent /sip/amd callers

    def at_node_cap() -> bool:
        """True when this process must refuse another call.

        ``settings.node_max_calls <= 0`` does not by itself fill the node. A limiter
        ``process_ceiling`` of 0 admits nobody, so the node is full. A positive ceiling
        or ``node_max_calls`` is full once this process holds that many calls.
        """
        ceiling = getattr(limiter, "process_ceiling", None)
        if ceiling == 0:
            return True
        if ceiling is not None and limiter.local_active() >= ceiling:
            return True
        return settings.node_max_calls > 0 and limiter.local_active() >= settings.node_max_calls

    def busy_twiml() -> str:
        """HTTP 200 so Twilio plays it. A 503 is a poor signal: Twilio does not retry TwiML well."""
        if overflow_url:
            return ('<?xml version="1.0" encoding="UTF-8"?><Response><Redirect>'
                    f'{escape(overflow_url)}</Redirect></Response>')
        return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                '<Say>Sorry, all of our lines are busy. Please call back shortly.</Say><Hangup/></Response>')

    @app.get("/healthz")
    async def healthz():
        """Readiness: 503 while draining, at the node cap, or when the database (or Redis, if used) is unreachable,
        so the load balancer stops sending new calls to this node."""
        checks = {}
        for name, fn in (("db", store.ping), ("redis", getattr(limiter, "ping", None))):
            if fn is None:
                continue
            try:
                # A pool checkout can block for the pool timeout. That must not freeze the loop.
                if name == "db":
                    await to_thread(fn)
                else:
                    fn()
                checks[name] = True
            except Exception:  # noqa: BLE001
                checks[name] = False
                METRICS.inc("vox_errors_total", where=f"healthz_{name}")
        node_full = at_node_cap()
        ready = not app.state.draining and not node_full and all(checks.values())
        body = {"ok": ready, "draining": app.state.draining, "nodeFull": node_full,
                "active_calls": limiter.local_active(), "checks": checks}
        return Response(json.dumps(body), status_code=200 if ready else 503, media_type="application/json")

    def admin_only(x_admin_key: str) -> None:
        if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(403, "admin key required")

    @app.post("/admin/drain")
    async def drain(x_admin_key: str = Header(default="")):
        """Stop taking new calls (live ones continue). Call from a pod preStop hook, then wait for active_calls == 0."""
        admin_only(x_admin_key)
        app.state.draining = True
        return {"draining": True, "active_calls": limiter.local_active()}

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
        if body.retentionDays is not None:
            store.update_tenant(tid, retention_days=body.retentionDays)
        key = security.generate_key()
        store.add_key(tid, security.hash_key(key), "initial")
        return {"tenantId": tid, "apiKey": key}   # key is shown once; only its hash is stored

    @app.patch("/admin/tenants/{tenant_id}")
    async def patch_tenant(tenant_id: str, body: TenantPatch, x_admin_key: str = Header(default="")):
        admin_only(x_admin_key)
        t = store.update_tenant(tenant_id, body.concurrencyLimit, body.ratePerMinute, body.retentionDays)
        if not t:
            raise HTTPException(404, "tenant not found")
        return {"tenantId": tenant_id, "concurrencyLimit": t["concurrency_limit"], "ratePerMinute": t.get("rate_per_minute"),
                "retentionDays": t.get("retention_days")}

    @app.post("/admin/purge")
    async def purge(x_admin_key: str = Header(default="")):
        """Runs the retention purge now (the API also runs it hourly). Handy for cron or after changing a limit."""
        admin_only(x_admin_key)
        return {"erased": retention.purge_expired(store, settings.record_dir or os.environ.get("VOX_RETENTION_DIR", os.environ.get("VOX_RECORD_DIR", "")),
                                                  settings.retention_default_days or None, setups=app.state.sip_setups)}

    @app.put("/v1/webhook")
    async def set_webhook(body: WebhookIn, t: dict = Depends(tenant_admin)):
        secret = security.generate_key()
        store.set_webhook(t["id"], body.url, secret)
        return {"url": body.url, "secret": secret}

    # ------------------------------------------------------------ agents
    @app.post("/v1/agents")
    async def create_agent(body: AgentIn, t: dict = Depends(tenant_write)):
        check_tools(body.tools, t["id"])
        cfg0 = body.model_dump(exclude={"name"})
        exp = cfg0.pop("experiment", None)
        if not cfg0.get("confirmSlots"):
            cfg0.pop("confirmSlots", None)
        if not cfg0.get("restateSlots"):
            cfg0.pop("restateSlots", None)          # default off: do not store a false flag on every agent
        cfg0["voice"], cfg0["language"] = pick_voice("af_heart", None, body.voice, body.language, "voice" in body.model_fields_set)
        check_voice(cfg0["voice"], cfg0["language"])
        if exp is not None:
            cfg0["experiment"] = check_experiment(exp, cfg0["language"])
        call_variables(cfg0, None, strict=False)
        a = store.create_agent(t["id"], body.name, cfg0)
        log(t, "agent.create", a["id"], name=body.name)
        return respond_agent(a, t["id"])

    @app.patch("/v1/agents/{agent_id}")
    async def update_agent(agent_id: str, body: AgentPatch, t: dict = Depends(tenant_write)):
        """Change an agent in place (same id, so a phone number's webhook keeps working). Only given fields change."""
        cur = store.get_agent(t["id"], agent_id)
        if not cur:
            raise HTTPException(404, "agent not found")
        if body.tools is not None:
            check_tools(body.tools, t["id"])
        cfg = dict(cur["config"])
        eff_voice, eff_lang = pick_voice(cfg.get("voice", "af_heart"), cfg.get("language"), body.voice, body.language,
                                         "voice" in body.model_fields_set)
        check_voice(eff_voice, eff_lang)
        changed = []
        for k in ("systemPrompt", "voice", "language", "variables", "greeting", "tools", "speed", "record", "bargeIn", "confirmSlots", "restateSlots", "redact"):
            v = getattr(body, k)
            if v is not None:
                cfg[k] = v
                changed.append(k)
        if "experiment" in body.model_fields_set:
            if body.experiment is None:
                cfg.pop("experiment", None)
            else:
                cfg["experiment"] = check_experiment(body.experiment, eff_lang)
            changed.append("experiment")
        else:
            # A language-only patch does not resend the experiment; variant voices still have to speak it.
            check_variant_voices(cfg.get("experiment"), eff_lang)
        if cfg.get("confirmSlots") is False:
            cfg.pop("confirmSlots", None)
        if cfg.get("restateSlots") is False:
            cfg.pop("restateSlots", None)           # a false patch clears the reminder; an omitted patch leaves it
        call_variables(cfg, None, strict=False)
        if eff_voice != cfg.get("voice"):
            cfg["voice"] = eff_voice
            changed.append("voice")
        log(t, "agent.update", agent_id, fields=changed)
        return respond_agent(store.update_agent(t["id"], agent_id, body.name, cfg), t["id"])

    @app.delete("/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str, t: dict = Depends(tenant_write)):
        if not store.delete_agent(t["id"], agent_id):
            raise HTTPException(404, "agent not found")
        log(t, "agent.delete", agent_id)
        return {"deleted": agent_id}

    @app.get("/v1/agents")
    async def list_agents(t: dict = Depends(tenant)):
        return [redacted(a) for a in store.list_agents(t["id"])]

    # ----------------------------------------------------- SIP numbers (dialed number -> agent)
    def _e164_or_422(raw: str) -> str:
        num = normalize_e164(raw)
        if not E164.match(num):
            raise HTTPException(422, "number must be E.164, e.g. +15551234567")
        return num

    @app.put("/v1/numbers/{e164}")
    async def put_number(e164: str, body: NumberIn, t: dict = Depends(tenant_write)):
        num = _e164_or_422(e164)
        if not store.get_agent(t["id"], body.agentId):
            raise HTTPException(422, "agent is not in this tenant")
        other = store.get_number(num)
        if other and other["tenant_id"] != t["id"]:
            raise HTTPException(409, "number is assigned to another tenant")
        if not store.put_number(t["id"], num, body.agentId):
            raise HTTPException(409, "number is assigned to another tenant")
        log(t, "number.put", num, agentId=body.agentId)
        return {"e164": num, "agentId": body.agentId}

    @app.delete("/v1/numbers/{e164}")
    async def delete_number(e164: str, t: dict = Depends(tenant_write)):
        num = _e164_or_422(e164)
        if not store.delete_number(t["id"], num):
            raise HTTPException(404, "number not found")
        log(t, "number.delete", num)
        return {"deleted": num}

    @app.get("/v1/numbers")
    async def list_numbers(t: dict = Depends(tenant)):
        return [{"e164": r["e164"], "agentId": r["agent_id"]} for r in store.list_numbers(t["id"])]

    @app.get("/v1/experiments/{experiment_id}")
    async def experiment_report(experiment_id: str, t: dict = Depends(tenant)):
        """Per-variant call counts for this tenant. First-audio means need stored timings, which this build does not keep."""
        counts: dict[str, int] = {}
        known = False
        for agent in store.list_agents(t["id"]):
            exp = (agent.get("config") or {}).get("experiment") or {}
            if isinstance(exp, dict) and exp.get("id") == experiment_id:
                known = True
                for variant in exp.get("variants") or []:
                    if isinstance(variant, dict) and variant.get("id"):
                        counts.setdefault(str(variant["id"]), 0)
        for call in store.list_calls(t["id"], 1000):
            meta = call.get("metadata") or {}
            if not isinstance(meta, dict) or meta.get("experiment") != experiment_id:
                continue
            known = True
            vid = str(meta.get("variant") or "")
            counts[vid] = counts.get(vid, 0) + 1
        if not known:
            raise HTTPException(404, "experiment not found")
        return {"id": experiment_id, "variants": [{"id": vid, "calls": counts[vid]} for vid in sorted(counts)]}

    # ------------------------------------------------------------- calls
    @app.post("/v1/calls")
    async def create_call(body: CallIn, t: dict = Depends(tenant_write)):
        base: dict = {}
        if body.agentId:
            agent = store.get_agent(t["id"], body.agentId)
            if not agent:
                raise HTTPException(404, "agent not found")
            base = dict(agent["config"])
        for k in ("systemPrompt", "voice", "language", "greeting", "tools", "speed", "record"):
            v = getattr(body, k)
            if v is not None:
                if k == "tools":
                    check_tools(v, t["id"])
                base[k] = v
        given_vars = call_variables(base, body.variables, strict=True)
        base["voice"], base["language"] = pick_voice(base.get("voice", "af_heart"), base.get("language"), body.voice,
                                                      body.language, "voice" in body.model_fields_set)
        check_voice(base["voice"], base["language"])
        meta = caller_metadata(body.metadata)
        if given_vars:
            meta["variables"] = given_vars
        call = store.create_call(t["id"], body.agentId, "webSocket", base, meta)
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
    async def delete_call(call_id: str, t: dict = Depends(tenant_admin)):
        """Erases a call: its record, transcript and recording file (for deletion requests and retention policies)."""
        c = store.get_call(t["id"], call_id)
        if not c:
            raise HTTPException(404, "call not found")
        if c["status"] == "active":
            raise HTTPException(409, "call is still active")
        retention.delete_call_files(settings.record_dir or os.environ.get("VOX_RECORD_DIR", ""), call_id)
        drop_sip_setup(c)
        store.delete_call(t["id"], call_id)
        log(t, "call.delete", call_id)
        return {"deleted": call_id}

    @app.get("/v1/calls/{call_id}/messages")
    async def call_messages(call_id: str, t: dict = Depends(tenant)):
        if not await to_thread(store.get_call, t["id"], call_id):
            raise HTTPException(404, "call not found")

        def _read():
            runtime.wait_messages()  # a row still in the writer is not an empty transcript
            # Peek: finish still has to raise this, so a reader must not pop it.
            err = runtime.peek_message_error(call_id)
            if err is not None:
                return err
            return store.messages(call_id)

        got = await to_thread(_read)
        if isinstance(got, BaseException):
            raise HTTPException(500, f"{type(got).__name__}: {got}")
        return got

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
        return {"tenantId": t["id"], "name": t["name"], "keyId": t.get("key_id"), "role": t.get("role") or "admin",
                "retentionDays": t.get("retention_days") or settings.retention_default_days or None,
                "concurrencyLimit": t["concurrency_limit"] or settings.default_concurrency}

    @app.get("/v1/voices")
    async def voices(t: dict = Depends(tenant)):
        langs = []
        for code, names in ttsmod.VOICES_BY_LANG.items():
            why = ttsmod.why_unavailable(code)
            langs.append({"code": ttsmod.LANG_ISO[code], "kokoroCode": code, "name": LANG_NAME[ttsmod.LANG_ISO[code]],
                          "voices": names, "available": not why, "reason": why or None,
                          "defaultVoice": DEFAULT_VOICE[ttsmod.LANG_ISO[code]]})
        return {"voices": VOICES, "languages": langs, "speed": {"min": 0.7, "max": 1.3, "default": 1.0}}

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
    async def list_keys(t: dict = Depends(tenant_admin)):
        return [{**k, "current": k["id"] == t.get("key_id")} for k in store.list_keys(t["id"])]

    @app.post("/v1/keys")
    async def create_key(body: KeyIn, t: dict = Depends(tenant_admin)):
        if body.role not in ROLES:
            raise HTTPException(422, f"role must be one of {list(ROLES)}")
        key = security.generate_key()
        kid = store.add_key(t["id"], security.hash_key(key), body.label[:100], body.role)
        log(t, "key.create", kid, label=body.label[:100], role=body.role)
        return {"id": kid, "apiKey": key, "role": body.role}            # shown once; only the hash is stored

    @app.delete("/v1/keys/{key_id}")
    async def revoke_key(key_id: str, t: dict = Depends(tenant_admin)):
        target = next((k for k in store.list_keys(t["id"]) if k["id"] == key_id and k["revoked"] is None), None)
        if target is None:
            raise HTTPException(404, "key not found or already revoked")
        if target["role"] in (None, "admin") and store.active_admin_key_count(t["id"]) <= 1:
            raise HTTPException(409, "cannot revoke the last active admin key; create another first")
        if not store.revoke_key(t["id"], key_id):
            raise HTTPException(404, "key not found or already revoked")
        log(t, "key.revoke", key_id)
        return {"revoked": key_id}

    @app.post("/v1/keys/{key_id}/rotate")
    async def rotate_key(key_id: str, t: dict = Depends(tenant_admin)):
        old = next((k for k in store.list_keys(t["id"]) if k["id"] == key_id and k["revoked"] is None), None)
        if not old:
            raise HTTPException(404, "key not found or already revoked")
        key = security.generate_key()
        kid = store.add_key(t["id"], security.hash_key(key), old["label"], old["role"] or "admin")
        store.revoke_key(t["id"], key_id)
        log(t, "key.rotate", key_id, newId=kid)
        return {"id": kid, "apiKey": key, "revoked": key_id}

    @app.get("/v1/audit")
    async def audit(limit: int = 100, before: Optional[int] = None, t: dict = Depends(tenant_admin)):
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
    async def put_tool(name: str, body: ToolIn, t: dict = Depends(tenant_write)):
        spec = lib_validate(name, body.spec)
        if len(store.list_tools(t["id"])) >= 200 and not store.get_tool(t["id"], name):
            raise HTTPException(409, "tool library is full (200)")
        store.put_tool(t["id"], name, spec)
        log(t, "tool.put", name)
        return {"name": name, "spec": toolmod.redact_specs([spec])[0]}

    @app.post("/v1/tools/import-openapi")
    async def import_openapi(body: OpenApiImportIn, t: dict = Depends(tenant_write)):
        """Creates one library tool per OpenAPI operation. Use them in an agent as {"library": "<name>"}."""
        try:
            tools, skipped = openapi_import.import_spec(body.spec, base_url=body.baseUrl, headers=body.headers,
                                                        include=body.include, prefix=body.prefix,
                                                        confirm_writes=body.confirmWrites)
        except toolmod.ToolConfigError as e:
            raise HTTPException(422, str(e)) from e
        existing = {x["name"] for x in store.list_tools(t["id"])}
        imported = []
        for spec in tools:
            if spec["name"] in existing and not body.overwrite:
                skipped.append({"operation": spec["name"], "reason": "a library tool with this name exists (overwrite=true)"})
                continue
            if len(existing) >= 200 and spec["name"] not in existing:
                skipped.append({"operation": spec["name"], "reason": "tool library is full (200)"})
                continue
            if not body.dryRun:
                store.put_tool(t["id"], spec["name"], spec)
                existing.add(spec["name"])
            imported.append(spec)
        if not body.dryRun:
            log(t, "tool.import_openapi", "", count=len(imported), skipped=len(skipped))
        return {"dryRun": body.dryRun, "imported": [x["name"] for x in imported], "skipped": skipped,
                "tools": toolmod.redact_specs(imported)}

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
    async def delete_tool(name: str, t: dict = Depends(tenant_write)):
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
    async def create_corpus(body: CorpusIn, t: dict = Depends(tenant_write)):
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
    async def delete_corpus(name: str, t: dict = Depends(tenant_write)):
        if not store.delete_corpus(t["id"], name):
            raise HTTPException(404, "knowledge base not found")
        log(t, "corpus.delete", name)
        return {"deleted": name}

    @app.post("/v1/corpora/{name}/documents")
    async def add_doc(name: str, body: DocIn, t: dict = Depends(tenant_write)):
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
    async def delete_doc(name: str, doc_id: str, t: dict = Depends(tenant_write)):
        if not store.delete_doc(t["id"], corpus_or_404(t, name)["id"], doc_id):
            raise HTTPException(404, "document not found")
        log(t, "doc.delete", f"{name}/{doc_id}")
        return {"deleted": doc_id}

    @app.post("/v1/corpora/{name}/search")
    async def search_corpus(name: str, body: dict, t: dict = Depends(tenant)):
        """Try a knowledge base without a call: {"query": "...", "topK": 3}."""
        from core.bm25 import BM25
        from core.embed import hybrid_search
        corpus_or_404(t, name)
        _, docs = store.corpus_docs(t["id"], name)
        bm, q, k = BM25(docs), str(body.get("query", "")), int(body.get("topK", 3))
        if getattr(runtime, "embedder", None) is not None:
            try:
                return {"results": await hybrid_search(bm, runtime.embedder, q, k), "mode": "hybrid"}
            except Exception:  # noqa: BLE001
                pass
        return {"results": bm.search(q, k), "mode": "keyword"}

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
    def reserve_slot(tenant_id: str) -> tuple[str, str]:
        """Reserve a node and tenant slot for a call that has no row yet.

        Returns (call_id, "ok") or ("", "node"|"tenant"). A refusal takes no slot. The id is what
        the later socket's admit() sees, so that socket does not take a second slot.
        """
        cid = new_id("call")
        tnt = store.get_tenant(tenant_id)
        cap = (tnt["concurrency_limit"] if tnt else None) or settings.default_concurrency
        decision = limiter.try_admit(tenant_id, cid, cap, settings.node_max_calls)
        if decision != "ok":
            return "", decision
        return cid, "ok"

    def create_reserved(cid: str, tenant_id: str, agent_id, medium: str, config: dict,
                        metadata: Optional[dict]) -> dict:
        try:
            return store.create_call(tenant_id, agent_id, medium, config, metadata, call_id=cid)
        except Exception:  # noqa: BLE001
            limiter.release(tenant_id, cid)
            raise

    def release_open(call: dict, reason: str) -> None:
        """End a leg that still holds a slot and has not reached finish()."""
        if call["status"] in ("created", "active"):
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason=reason)
            limiter.release(call["tenant_id"], call["id"])

    async def _release_unheld(call: dict) -> None:
        """Drop a reservation the socket never adopted. A row already active keeps its slot."""
        try:
            cur = await to_thread(store.get_call_any, call["id"])
        except Exception:  # noqa: BLE001
            cur = None
        if cur is not None and cur.get("status") == "active":
            return
        try:
            limiter.release(call["tenant_id"], call["id"])
        except Exception:  # noqa: BLE001
            METRICS.inc("vox_errors_total", where="limiter_release")

    async def admit(call: dict) -> dict:
        """Status, drain, claim, per-node cap, then per-tenant concurrency. Raises _Reject.

        Store calls run in a worker. webhook() stays on this task: dispatcher.spawn needs a running loop.
        The created->active claim uses update_call_if_status, which does not call update_call. A second
        update_call(status="active") runs only after the slot is held, so a lock on that write still
        releases the ceiling.
        """
        if call["status"] != "created":
            raise _Reject(4401, "call not joinable")
        if app.state.draining:
            ended = await to_thread(store.update_call_if_status, call["id"], "created",
                                    status="ended", ended=time.time(), end_reason="draining")
            if ended:
                try:
                    limiter.release(call["tenant_id"], call["id"])  # no-op unless a webhook reserved this id
                except Exception:  # noqa: BLE001
                    METRICS.inc("vox_errors_total", where="limiter_release")
            else:
                await _release_unheld(call)
            raise _Reject(1013, "node is draining")
        t = await to_thread(store.get_tenant, call["tenant_id"])
        # The status check above is from before get_tenant. A drain in that gap takes no slot.
        if app.state.draining:
            ended = await to_thread(store.update_call_if_status, call["id"], "created",
                                    status="ended", ended=time.time(), end_reason="draining")
            if ended:
                try:
                    limiter.release(call["tenant_id"], call["id"])
                except Exception:  # noqa: BLE001
                    METRICS.inc("vox_errors_total", where="limiter_release")
            else:
                await _release_unheld(call)
            raise _Reject(1013, "node is draining")
        cap = t["concurrency_limit"] or settings.default_concurrency
        joined = time.time()
        claimed = await to_thread(
            store.update_call_if_status, call["id"], "created",
            status="active", joined=joined)
        if not claimed:
            # The winner already holds the slot once the row is active. A busy/ended row never
            # took a websocket slot; releasing that id is a no-op unless a webhook reserved it.
            await _release_unheld(call)
            raise _Reject(4401, "call not joinable")
        # Node cap and tenant cap are one locked check. An id reserved before the socket returns
        # "ok" and is not counted twice.
        decision = limiter.try_admit(call["tenant_id"], call["id"], cap, settings.node_max_calls)
        if decision != "ok":
            full = decision == "node"
            # The claim above stamped joined. A refusal never joined: clear it so a SIP continuation
            # is still "never joined" and restore_continuation can put the row back to created.
            await to_thread(store.update_call, call["id"], status="ended", ended=time.time(),
                            end_reason="node_full" if full else "concurrency_limit", joined=None)
            raise _Reject(1013, "node full" if full else "concurrency limit")
        try:
            await to_thread(store.update_call, call["id"], status="active", joined=joined)
            fresh = await to_thread(store.get_call_any, call["id"])
            webhook(t, "call.started", fresh)
        except Exception:  # noqa: BLE001  (slot is held; finish runs only after admit returns)
            METRICS.inc("vox_errors_total", where="admit")
            try:
                limiter.release(call["tenant_id"], call["id"])
            except Exception:  # noqa: BLE001
                pass
            try:
                await to_thread(store.update_call, call["id"], status="ended", ended=time.time(), end_reason="error")
            except Exception:  # noqa: BLE001  (database still down: the slot is already released)
                pass
            raise _Reject(1011, "unavailable")
        return t

    def drop_sip_setup(call: Optional[dict]) -> None:
        """The setup row is only how the caller id reaches the call. It is not kept after the call is over.

        Inbound legs store the socket uuid as callSid. A bound or outbound leg stores the ARI channel
        there and the socket uuid as audioSocketUuid. Both keys are removed; a missing row is a no-op.
        """
        meta = (call or {}).get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        seen: set[str] = set()
        for key in ("callSid", "audioSocketUuid"):
            uid = str(meta.get(key) or "")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            store.delete_sip_setup(uid)
            app.state.sip_setups.pop(uid, None)

    async def finish(call: dict, t: dict, reason: str) -> None:
        # A failed transcript insert must not skip the slot release or leave the call active,
        # and it must not be reported as a clean hang-up.
        write_err = None
        try:
            await to_thread(runtime.flush_messages, call["id"])
        except Exception as exc:  # noqa: BLE001
            write_err = exc
        end_reason = "transcript_failed" if write_err is not None else reason
        try:
            limiter.release(call["tenant_id"], call["id"])
        except Exception:  # noqa: BLE001  (a Redis blip must not skip ending the call)
            METRICS.inc("vox_errors_total", where="limiter_release")

        def _mark_ended():
            # Drop the live session before the ended row is visible. A caller that polls for
            # status=ended (the AudioSocket hang-up test) must not still find the session.
            try:
                runtime.finalize(call["id"])
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="finalize")
            # A dead read must not skip the write, and one failed update is retried so the
            # row does not stay active after the slot is gone.
            last: BaseException | None = None
            for _attempt in range(2):
                try:
                    cur = None
                    try:
                        cur = store.get_call_any(call["id"])
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                    # Resume may already have ended the row as stream_lost. A failed insert still
                    # has to replace that reason; leaving it reports a lost transcript as a cut-off.
                    must_record_failure = write_err is not None and (
                        cur is None or cur.get("status") == "active" or cur.get("end_reason") != "transcript_failed")
                    if cur is None or cur.get("status") == "active" or must_record_failure:
                        store.update_call(call["id"], status="ended", ended=time.time(), end_reason=end_reason)
                    done = store.get_call_any(call["id"])
                    if done is not None and done.get("status") != "active":
                        if write_err is not None and done.get("end_reason") != "transcript_failed":
                            last = RuntimeError("transcript failure was not recorded")
                            continue
                        drop_sip_setup(done or call)
                        secs = round((done["ended"] - done["joined"]), 2) if done.get("joined") and done.get("ended") else 0.0
                        return done, secs
                except Exception as exc:  # noqa: BLE001
                    last = exc
            if last is not None:
                raise last
            raise RuntimeError("call did not end")

        def _transcript():
            rows = [{"role": m["role"], "text": m["text"], "t": m["t"]} for m in store.messages(call["id"])[:500]]
            # Redact at send time. A row written raw stays raw in the table; finish does not copy it.
            if (call.get("config") or {}).get("redact"):
                rows = [{**m, "text": redact_text(m["text"])} for m in rows]
            return rows

        snap_err = None
        done, secs = None, 0.0
        try:
            done, secs = await to_thread(_mark_ended)
        except Exception as exc:  # noqa: BLE001
            snap_err = exc
        transcript: list = []
        read_err = None
        if done is not None:
            try:
                transcript = await to_thread(_transcript)
            except Exception as exc:  # noqa: BLE001
                read_err = exc
        if done is not None:
            METRICS.inc("vox_calls_total", medium=call["medium"], reason=end_reason)
            extra: dict = {}
            failures = []
            if write_err is not None:
                failures.append(f"{type(write_err).__name__}: {write_err}")
            if read_err is not None:
                failures.append(f"{type(read_err).__name__}: {read_err}")
            if failures:
                extra["error"] = "; ".join(failures)
            # An unread transcript is not an empty one: omitting it keeps a retry from
            # delivering [] for rows that are still in the table.
            if read_err is None:
                extra["transcript"] = transcript
            webhook(t, "call.ended", done, durationSeconds=secs, **extra)
        if write_err is not None:
            raise write_err
        if snap_err is not None:
            raise snap_err
        if read_err is not None:
            raise read_err

    @app.websocket("/ws/calls/{call_id}")
    async def ws_call(ws: WebSocket, call_id: str, token: str = ""):
        await ws.accept()
        call = await to_thread(store.get_call_any, call_id)
        if not call or not hmac.compare_digest(token, call["join_token"]) or call["medium"] != "webSocket":
            await ws.close(4401)
            return
        try:
            t = await admit(call)
        except _Reject as r:
            await ws.close(r.code, r.reason)
            return
        reason = "hangup"
        try:
            async def factory(tr):
                return await to_thread(runtime.build_session, call, tr, True)
            await BrowserTransport(_BrowserWS(ws), factory).run()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            reason = "error"
            raise
        finally:
            await finish(call, t, reason)

    # ---------------------------------------------------------- telephony
    def stream_twiml(call: dict) -> str:
        """Connects the call to the agent. If the media stream ends, Twilio moves to the <Redirect>, which asks
        /twilio/resume/{id} what to do: hang up (the agent ended the call) or reconnect (the stream or node was lost)."""
        ws = settings.public_base_url.replace("http", "ws", 1)
        return ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
                f'<Stream url={quoteattr(f"{ws}/ws/twilio/{call["id"]}")}>'
                f'<Parameter name="token" value={quoteattr(call["join_token"])}/>'
                '</Stream></Connect>'
                f'<Redirect method="POST">{settings.public_base_url}/twilio/resume/{call["id"]}</Redirect></Response>')

    HANGUP_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
    MAX_RESUMES = 2

    def _twilio_rest_ready() -> bool:
        return bool(settings.twilio_account_sid and settings.twilio_auth_token and runtime.http is not None)

    async def bind_status_callback(call_id: str, call_sid: str) -> bool:
        """Point the live Twilio call at /twilio/status/{call_id}, which releases a still-created leg.

        Inbound TwiML has nowhere to put that callback. Without it, a caller who hangs up before
        the socket exists never reaches /twilio/resume, and the reservation stays held.
        No account sid or HTTP client means there is no REST call to make (inbound-only deploys,
        and tests that never dial Twilio); the leg still proceeds. When REST is configured, an
        empty CallSid or a failed update returns False and the caller must drop the reservation.
        """
        if not _twilio_rest_ready():
            return True
        if not call_sid:
            return False
        try:
            r = await runtime.http.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls/{call_sid}.json",
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                data={"StatusCallback": f"{settings.public_base_url}/twilio/status/{call_id}",
                      "StatusCallbackMethod": "POST"},
                timeout=10)
            return r.status_code < 300
        except Exception:  # noqa: BLE001
            return False

    def _drop_unbound(call: dict) -> Response:
        """The status callback was not registered. Re-read: /ws/twilio may have admitted during the REST await.

        The `call` argument is the pre-await row. Ending that snapshot marks a live leg ended and
        releases its slot while the socket is still up, so the next caller takes the only seat.
        """
        cur = store.get_call_any(call["id"]) or call
        if cur.get("status") == "active":
            return Response(stream_twiml(cur), media_type="text/xml")
        if cur.get("status") == "created":
            release_open(cur, "status_callback_failed")
        return Response(HANGUP_TWIML, media_type="text/xml")

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
            return Response(busy_twiml(), media_type="text/xml")   # no call row: nothing was admitted
        agent = await to_thread(store.get_agent_any, agent_id)
        if not agent:
            raise HTTPException(404, "agent not found")
        call_sid = str(form.get("CallSid", "") or "").strip()
        # An empty CallSid can never appear in a status callback, so the slot would stick until restart.
        if not call_sid:
            return Response(HANGUP_TWIML, media_type="text/xml")
        # Twilio retries this URL when our response is lost. The first row already holds the slot.
        existing = await to_thread(store.find_open_twilio_call, agent["tenant_id"], call_sid)
        if existing:
            return Response(stream_twiml(existing), media_type="text/xml")
        # Hold the slot before handing Twilio a stream URL, or the next webhook still sees a free node.
        cid, decision = reserve_slot(agent["tenant_id"])
        if decision != "ok":
            return Response(busy_twiml(), media_type="text/xml")   # no call row: nothing was admitted
        call = await to_thread(create_reserved, cid, agent["tenant_id"], agent_id, "twilio", agent["config"],
                               {"from": form.get("From", ""), "to": form.get("To", ""), "callSid": call_sid})
        if not await bind_status_callback(call["id"], call_sid):
            return _drop_unbound(call)
        return Response(stream_twiml(call), media_type="text/xml")

    @app.websocket("/ws/twilio/{call_id}")
    async def ws_twilio(ws: WebSocket, call_id: str):
        await ws.accept()
        call = await to_thread(store.get_call_any, call_id)
        if not call or call["medium"] != "twilio":
            await ws.close(4401)
            return
        state: dict = {}

        async def factory(tr, custom):
            if not hmac.compare_digest(str(custom.get("token", "")), call["join_token"]):
                raise _Reject(4401, "bad token")
            state["t"] = await admit(call)
            return await to_thread(runtime.build_session, call, tr)

        tr = TwilioTransport(_TwilioWS(ws), factory)
        tr.before_end = lambda: to_thread(store.update_call, call_id, end_reason="agent_hangup")
        try:
            await tr.run()
        except _Reject as r:
            state["rejected"] = True
            await ws.close(r.code, r.reason)
        finally:
            if "t" in state:
                await finish(call, state["t"], "agent_hangup" if tr.ended_by_us else "stream_closed")
            elif state.get("rejected") and (await to_thread(store.get_call_any, call_id) or {}).get("status") != "active":
                # Bad token, drain, or a cap refusal: drop a reservation the socket never adopted.
                # A clean close before "start" keeps the slot so /twilio/resume can hand it to the new leg.
                limiter.release(call["tenant_id"], call["id"])

    app.state.audiosocket_writers = set()

    def require_sip_token(header: str) -> None:
        """Empty VOX_SIP_TOKEN turns SIP setup off (404). A wrong token is 403, not a silent accept."""
        if not settings.sip_token:
            raise HTTPException(404, "not found")
        if not hmac.compare_digest(header or "", settings.sip_token):
            raise HTTPException(403, "bad sip token")

    def sip_setup_for(uid: str) -> Optional[dict]:
        """In-memory row wins when it is fresh; the table covers a lookup from another worker. Older than 60 s is ignored."""
        now = time.time()
        mem = app.state.sip_setups.get(uid)
        if mem and now - float(mem.get("created") or 0) <= 60:
            dialed = mem.get("exten") or mem.get("callee") or ""
            return {**mem, "exten": dialed, "callee": dialed}
        return store.get_sip_setup(uid, max_age=60)

    @app.post("/internal/sip/setup")
    async def sip_setup(request: Request, x_vox_sip_token: str = Header(default="")):
        """Asterisk posts the AudioSocket UUID, caller id, and dialed number before it connects the audio.

        The dialed number is `exten` or `callee` (the same value). Both names are stored so routing and
        {{callee}} see it.
        """
        require_sip_token(x_vox_sip_token)
        form = {k: str(v) for k, v in (await request.form()).items()}
        uid = form.get("uuid", "").strip()
        if not uid:
            raise HTTPException(422, "uuid is required")
        caller = form.get("caller", "")
        dialed = form.get("exten") or form.get("callee") or ""
        created = time.time()
        store.put_sip_setup(uid, caller, dialed, created)
        stale = [k for k, v in app.state.sip_setups.items() if created - float(v.get("created") or 0) > 60]
        for k in stale:
            app.state.sip_setups.pop(k, None)
        app.state.sip_setups[uid] = {"uuid": uid, "caller": caller, "exten": dialed, "callee": dialed, "created": created}
        return {"ok": True}

    async def audiosocket_call(reader, writer) -> None:
        """One SIP call from Asterisk.

        A slot is reserved before the UUID read, so a second connection already sees a full node,
        and no call row exists until that UUID arrives. A uuid that already belongs to a SIP call
        is resumed instead of minting a blank row; that path can run even when the placeholder
        reservation was refused, because continuing a dead leg frees its slot. An outbound row
        created for this socket, or a bind that arrived before the row, is adopted the same way.
        The setup row maps the UUID to the dialed number, and `numbers` picks the agent. Caller id
        is copied onto the call before the session is built. No fresh setup, or no mapping, falls
        back to settings.audiosocket_agent.
        """
        from transports.audiosocket import AudioSocketTransport
        app.state.audiosocket_writers.add(writer)
        state: dict = {}
        cid = ""
        holder = None
        tr = None
        try:
            if app.state.draining:
                return
            default = await to_thread(store.get_agent_any, settings.audiosocket_agent) if settings.audiosocket_agent else None
            if default is None:
                return
            # Reserve before reading the UUID. The read awaits, and a second connection must already see a full node.
            cid, decision = reserve_slot(default["tenant_id"])
            if decision == "ok":
                holder = default
            else:
                # Still read the UUID. A new call gets no row. A dead leg with this uuid can be resumed,
                # and sip_resume releases the slot that made this reservation fail.
                cid = ""

            async def factory(transport, custom):
                nonlocal cid, holder
                uid = str(custom.get("uuid", "") or "")
                # /sip/bind maps this socket to the ARI channel. Recover still addresses a leg by callSid,
                # which is the socket uuid when no channel was bound.
                ari_id = app.state.sip_by_uuid.get(uid, "") if uid else ""
                existing = store.find_call_by_sid(uid, medium="sip") if uid else None
                if existing is None and uid:
                    by_audio = store.find_sip_by_audio_uuid(uid)
                    if by_audio is not None and sip_socket_reuses(by_audio):
                        existing = by_audio
                if existing is not None and sip_socket_reuses(existing):
                    resumed = sip_resume(existing)
                    if resumed["action"] != "continue":
                        writer.close()
                        raise _Reject(4401, resumed.get("reason") or "sip call is not attachable")
                    call = resumed["call"]
                    # The placeholder is not this continuation. Free it before admit so a node cap of 1
                    # can take the continuation's own id.
                    if cid and holder is not None and call["id"] != cid:
                        limiter.release(holder["tenant_id"], cid)
                        cid = ""
                        holder = None
                    state["call"] = call
                    try:
                        state["t"] = await admit(call)
                    except _Reject:
                        # admit marks the row ended before raising. A continuation that never joined holds no
                        # slot; leaving it ended spends a resume and the next socket mints a blank call.
                        fresh = await to_thread(store.get_call_any, call["id"])
                        if fresh is not None and never_joined_continuation(fresh, sip_meta(fresh)):
                            await to_thread(restore_continuation, fresh)
                        raise
                    return await to_thread(runtime.build_session, call, transport)
                # Outbound originate already inserted this row and holds its slot. Bind-before-audio
                # names the same channel. Adopting it keeps voicemail, from, and to; a second row would
                # be a different call. A brand-new uuid with no reservation still raises below.
                adopted = None
                if ari_id:
                    bound_id = app.state.sip_by_channel.get(ari_id)
                    if bound_id:
                        found = store.get_call_any(bound_id)
                        if found and found.get("medium") == "sip" and found.get("status") == "created":
                            adopted = found
                    if adopted is None:
                        adopted = store.find_sip_call(call_sid=ari_id)
                if adopted is None and uid:
                    adopted = store.find_sip_call(audio_uuid=uid)
                if adopted is not None:
                    if cid and holder is not None and adopted["id"] != cid:
                        limiter.release(holder["tenant_id"], cid)
                        cid = ""
                        holder = None
                    patch = {"channel": "sip", "audioSocketUuid": uid}
                    if ari_id:
                        patch["callSid"] = ari_id
                    store.merge_call_metadata(adopted["id"], patch)
                    call = store.get_call_any(adopted["id"])
                    if call is None:
                        writer.close()
                        raise EOFError("sip call is gone")
                    bound = str((call.get("metadata") or {}).get("callSid") or "")
                    if bound:
                        app.state.sip_by_channel[bound] = call["id"]
                    if ari_id:
                        transport.call_sid = ari_id
                    state["call"] = call
                    try:
                        state["t"] = await admit(call)
                    except _Reject:
                        fresh = await to_thread(store.get_call_any, call["id"])
                        if fresh is not None and never_joined_continuation(fresh, sip_meta(fresh)):
                            await to_thread(restore_continuation, fresh)
                        raise
                    return await to_thread(runtime.build_session, call, transport)
                if not cid or holder is None:
                    writer.close()
                    raise EOFError("node full")
                setup = sip_setup_for(uid) if uid else None
                chosen = None
                dialed = ""
                if setup:
                    dialed = setup.get("exten") or setup.get("callee") or ""
                    row = store.get_number(dialed) if dialed else None
                    if row:
                        chosen = store.get_agent(row["tenant_id"], row["agent_id"])
                        if chosen is None:            # number points at a deleted agent: do not silently use the default
                            writer.close()
                            raise EOFError("mapped agent is gone")
                if chosen is None:
                    chosen = default
                if chosen["tenant_id"] != holder["tenant_id"]:
                    # The placeholder holds this process's only slot. Free it, then take the mapped tenant's.
                    limiter.release(holder["tenant_id"], cid)
                    cid, decision = reserve_slot(chosen["tenant_id"])
                    if decision != "ok":
                        cid = ""
                        writer.close()
                        raise EOFError("node full")
                    holder = chosen
                # build_session reads this dict, not a fresh store row, so the caller id has to be on it
                # before the session is built ({{caller}} / CallContext.caller).
                # No ARI channel: callSid stays the socket uuid so /sip/recover can find this leg.
                # audioSocketUuid is the same id, which is what a cold transfer looks up.
                meta = {"callSid": ari_id or uid, "audioSocketUuid": uid, "channel": "sip", "from": "", "to": ""}
                if setup:
                    meta["from"] = setup.get("caller") or ""
                    meta["to"] = dialed
                call = await to_thread(create_reserved, cid, chosen["tenant_id"], chosen["id"], "sip", chosen["config"], meta)
                if ari_id:
                    app.state.sip_by_channel[ari_id] = call["id"]
                    transport.call_sid = ari_id
                state["call"] = call
                state["t"] = await admit(call)
                return await to_thread(runtime.build_session, call, transport)

            tr = AudioSocketTransport(reader, writer, factory, sample_rate=settings.audiosocket_rate)
            tr.before_end = lambda: (
                to_thread(store.update_call, state["call"]["id"], end_reason="agent_hangup") if state.get("call") else None)
            try:
                await tr.run(uuid_timeout=AUDIOSOCKET_UUID_TIMEOUT_S)
            except _Reject:
                pass
            except EOFError:
                pass
            except Exception:  # noqa: BLE001
                # admit() can raise after the row exists. Cleanup is the finally below; do not
                # leave this as an unhandled client_connected_cb error with the row still created.
                METRICS.inc("vox_errors_total", where="audiosocket")
        finally:
            app.state.audiosocket_writers.discard(writer)
            if "t" in state:
                await finish(state["call"], state["t"], "agent_hangup" if tr is not None and tr.ended_by_us else "stream_closed")
            elif cid and holder is not None:
                def _release_placeholder() -> None:
                    # Re-read. A _Reject path (draining, node full) already ended the row; writing
                    # admit_failed over that dict would erase the real reason. A created/active row
                    # is one admit never finished, and it still holds the slot.
                    cur = store.get_call_any(state["call"]["id"]) if state.get("call") else None
                    if cur and cur["status"] in ("created", "active"):
                        release_open(cur, "admit_failed")
                    else:
                        limiter.release(holder["tenant_id"], cid)
                        # admit() records a failed active-status write as "error" so a browser websocket
                        # still ends with that reason. This SIP leg never started, so the reason is admit_failed.
                        # draining, node_full, and concurrency_limit are left as admit() wrote them.
                        if cur and cur.get("end_reason") == "error":
                            try:
                                store.update_call(cur["id"], end_reason="admit_failed")
                            except Exception:  # noqa: BLE001
                                pass
                    if cur or state.get("call"):
                        drop_sip_setup(cur or state.get("call"))

                await to_thread(_release_placeholder)
            _abort_writer(writer)

    @app.on_event("startup")
    async def _start_audiosocket():
        if settings.audiosocket_port and settings.audiosocket_agent:
            import asyncio
            app.state.audiosocket = await asyncio.start_server(audiosocket_call, settings.audiosocket_host,
                                                               settings.audiosocket_port)

    @app.on_event("shutdown")
    async def _stop_audiosocket():
        srv = getattr(app.state, "audiosocket", None)
        if srv:
            srv.close()
        # Accepted peers blocked in the UUID read are not listening to close(); abort them so
        # the handler's finally releases the slot instead of waiting out the grace period.
        for w in list(getattr(app.state, "audiosocket_writers", ())):
            _abort_writer(w)
        if srv:
            await srv.wait_closed()

    @app.on_event("shutdown")
    async def _stop_message_log():
        await to_thread(runtime.message_log.close)

    @app.post("/twilio/resume/{call_id}")
    async def twilio_resume(call_id: str, request: Request):
        """Twilio lands here when a call's media stream has ended but the caller is still on the line. If the agent
        ended the call, hang up. Otherwise the stream or the node serving it was lost: reconnect the caller to a
        healthy node with the conversation so far, at most MAX_RESUMES times per call."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = await to_thread(store.get_call_any, call_id)
        if not call or call["medium"] != "twilio" or call.get("end_reason") in ("agent_hangup", "transfer", "no_credit"):
            return Response(HANGUP_TWIML, media_type="text/xml")
        # The call is already over. Reconnecting would reserve another leg nobody answers.
        if form.get("CallStatus", "") in TERMINAL_CALL_STATUS:
            status_name = form["CallStatus"]

            def _end_terminal() -> None:
                cur = store.get_call_any(call_id) or call
                if cur.get("status") in ("created", "active"):
                    store.update_call(call_id, status="ended", ended=time.time(), end_reason=status_name)
                    try:
                        limiter.release(cur["tenant_id"], cur["id"])
                    except Exception:  # noqa: BLE001
                        METRICS.inc("vox_errors_total", where="limiter_release")

            await to_thread(_end_terminal)
            return Response(HANGUP_TWIML, media_type="text/xml")
        if app.state.draining:
            # Busy TwiML, not a 503. The snapshot from before this await must not pick the reason.
            def _end_drained() -> None:
                runtime.wait_messages()
                err = runtime.peek_message_error(call_id)
                cur = store.get_call_any(call_id) or call
                if cur.get("end_reason") == "transcript_failed" and cur.get("status") != "active":
                    return
                reason = "transcript_failed" if (err is not None or cur.get("end_reason") == "transcript_failed") else "stream_lost"
                if cur.get("status") in ("created", "active"):
                    store.update_call(call_id, status="ended", ended=time.time(), end_reason=reason)
                    try:
                        limiter.release(cur["tenant_id"], cur["id"])
                    except Exception:  # noqa: BLE001
                        METRICS.inc("vox_errors_total", where="limiter_release")

            await to_thread(_end_drained)
            return Response(busy_twiml(), media_type="text/xml")
        meta = call["metadata"] if isinstance(call["metadata"], dict) else json.loads(call["metadata"] or "{}")
        n = int(meta.get("resumes", 0))
        if n >= MAX_RESUMES:
            def _end_capped() -> None:
                runtime.wait_messages()
                err = runtime.peek_message_error(call_id)
                cur = store.get_call_any(call_id) or call
                if cur.get("end_reason") == "transcript_failed" and cur.get("status") != "active":
                    return
                reason = "transcript_failed" if (err is not None or cur.get("end_reason") == "transcript_failed") else "stream_lost"
                if cur.get("status") in ("created", "active"):
                    store.update_call(call_id, status="ended", ended=time.time(), end_reason=reason)
                    try:
                        limiter.release(cur["tenant_id"], cur["id"])
                    except Exception:  # noqa: BLE001
                        METRICS.inc("vox_errors_total", where="limiter_release")

            await to_thread(_end_capped)
            METRICS.inc("vox_call_resume_refused_total")
            return Response('<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, we are having technical '
                            'difficulties. Please call again shortly. Goodbye.</Say><Hangup/></Response>', media_type="text/xml")
        def _end_lost(reason: str) -> None:
            # The end row must not land while the last lines are still only in the writer.
            # A failed insert is not a clean cut-off. finish may already have consumed the
            # in-memory error and stored transcript_failed; do not replace that.
            runtime.wait_messages()
            err = runtime.peek_message_error(call_id)
            cur = store.get_call_any(call_id) or call
            if cur.get("end_reason") == "transcript_failed" and cur.get("status") != "active":
                return
            if err is not None or cur.get("end_reason") == "transcript_failed":
                reason = "transcript_failed"
            if cur.get("status") in ("created", "active"):
                store.update_call(call_id, status="ended", ended=time.time(), end_reason=reason)
                try:
                    limiter.release(cur["tenant_id"], cur["id"])
                except Exception:  # noqa: BLE001
                    METRICS.inc("vox_errors_total", where="limiter_release")

        def _handoff():
            # Free the dead leg and hold the replacement before returning. A gap where
            # local_active drops to 0 lets another join take the only slot.
            # resumeReason "transfer" belongs only to the call transfer-result created. Copying it would make
            # this dropped stream keep the "not available" greeting instead of the cut-off line.
            _end_lost("stream_lost")
            cid, decision = reserve_slot(call["tenant_id"])
            if decision != "ok":
                return None
            carried = {k: v for k, v in meta.items() if k != "resumeReason"}
            return create_reserved(cid, call["tenant_id"], call["agent_id"], "twilio", call["config"],
                                   {**carried, "resumedFrom": call_id, "resumes": n + 1})

        nxt = await to_thread(_handoff)
        if nxt is None:
            return Response(busy_twiml(), media_type="text/xml")   # no new row while this process is full
        resumed_sid = str(form.get("CallSid", "") or (nxt["metadata"] or {}).get("callSid") or "")
        if not await bind_status_callback(nxt["id"], resumed_sid):
            return _drop_unbound(nxt)
        METRICS.inc("vox_call_resumes_total")
        await to_thread(store.audit, call["tenant_id"], "system", "call.resume", nxt["id"], {"from": call_id, "attempt": n + 1})
        return Response(stream_twiml(nxt), media_type="text/xml")

    SIP_OURS = ("agent_hangup", "transfer", "no_credit")
    SIP_LOST = ("stream_closed", "stream_lost")

    def sip_meta(call: dict) -> dict:
        meta = call.get("metadata") or {}
        return json.loads(meta or "{}") if isinstance(meta, str) else meta

    def never_joined_continuation(call: dict, meta: dict) -> bool:
        """Continuation admit rejected before join. It holds no slot and is not a used resume."""
        if not meta.get("resumedFrom") or call.get("joined"):
            return False
        if call["status"] == "created":
            return True
        return call.get("end_reason") in ("concurrency_limit", "draining")

    def restore_continuation(call: dict) -> dict:
        if call["status"] == "created" and not call.get("ended") and not call.get("end_reason"):
            return call
        store.update_call(call["id"], status="created", ended=None, end_reason=None)
        return store.get_call_any(call["id"])

    def end_dead_sip(call: dict, end_reason: Optional[str] = None) -> None:
        """Hangup of a row this process is not running. finish still owns a live socket."""
        if call["id"] in runtime.live or call["status"] not in ("created", "active"):
            return
        fields: dict[str, Any] = {"status": "ended", "ended": time.time()}
        if end_reason is not None:
            fields["end_reason"] = end_reason
        store.update_call(call["id"], **fields)
        limiter.release(call["tenant_id"], call["id"])

    def sip_resume(call: dict) -> dict:
        """What POST /sip/recover and the next AudioSocket do with the newest row for a uuid.

        action is hangup (no new row), busy (this process or another node still has the slot), or continue
        (call is the row to admit). A never-joined continuation is returned as-is.
        A hangup of a dead created/active row ends it so the slot is not kept.
        """
        # runtime.live is only this process. A Redis member another node still refreshes is that node's call.
        if limiter.held_elsewhere(call["tenant_id"], call["id"]):
            return {"action": "busy"}
        if call.get("end_reason") in SIP_OURS:
            end_dead_sip(call)                                 # keep the reason finish or mark_hangup stored
            return {"action": "hangup"}
        # Ending a call this process is still running frees the slot under the session and the
        # next socket opens a second call on the same caller.
        if call["id"] in runtime.live:
            return {"action": "busy"}
        meta = sip_meta(call)
        if never_joined_continuation(call, meta):
            return {"action": "continue", "call": restore_continuation(call)}
        n = int(meta.get("resumes", 0))
        if n >= MAX_RESUMES:
            METRICS.inc("vox_call_resume_refused_total")
            # stream_lost, same as a resume that is still allowed, so the next socket does not mint a blank call
            end_dead_sip(call, "stream_lost")
            return {"action": "hangup", "reason": "resume cap"}
        if call["status"] in ("created", "active"):
            store.update_call(call["id"], status="ended", ended=time.time(), end_reason="stream_lost")
            limiter.release(call["tenant_id"], call["id"])
        nxt = store.create_call(call["tenant_id"], call["agent_id"], "sip", call["config"],
                                {**meta, "resumedFrom": call["id"], "resumes": n + 1})
        METRICS.inc("vox_call_resumes_total")
        store.audit(call["tenant_id"], "system", "call.resume", nxt["id"],
                    {"from": call["id"], "attempt": n + 1, "channel": "sip"})
        return {"action": "continue", "call": nxt}

    def sip_socket_reuses(call: dict) -> bool:
        """True when a new AudioSocket must not mint a blank row for this uuid."""
        if call["id"] in runtime.live or call.get("end_reason") in SIP_OURS:
            return True
        if call.get("end_reason") in SIP_LOST or call["status"] == "active":
            return True
        return never_joined_continuation(call, sip_meta(call))

    @app.post("/sip/recover")
    async def sip_recover(request: Request, x_vox_sip_token: str = Header(default="")):
        """Lost AudioSocket call. Body `uuid` is the AudioSocket id (metadata callSid), not our call id:
        a re-dial does not know the id. Same history rules as /twilio/resume, but JSON instead of TwiML.
        When the socket EOF was not our hang-up, finish has already stored end_reason=stream_closed."""
        if not settings.sip_token:
            raise HTTPException(404, "not found")
        if not hmac.compare_digest(x_vox_sip_token or "", settings.sip_token):
            raise HTTPException(403, "bad sip token")
        if app.state.draining:
            raise HTTPException(503, "node is draining")
        if "json" in request.headers.get("content-type", ""):
            try:
                data = await request.json()
            except Exception:  # noqa: BLE001
                data = {}
            uid = str(data.get("uuid", "")).strip() if isinstance(data, dict) else ""
        else:
            form = {k: str(v) for k, v in (await request.form()).items()}
            uid = form.get("uuid", "").strip()
        if not uid:
            raise HTTPException(422, "uuid is required")
        call = store.find_call_by_sid(uid, medium="sip")
        if not call:
            raise HTTPException(404, "call not found")
        decision = sip_resume(call)
        if decision["action"] == "hangup":
            body = {"action": "hangup"}
            if decision.get("reason"):
                body["reason"] = decision["reason"]
            return body
        if decision["action"] == "busy":
            raise HTTPException(409, "call is still connected")
        nxt = decision["call"]
        return {"callId": nxt["id"], "agentId": nxt["agent_id"]}

    # ----------------------------------------------------- outbound calls
    def twilio_check(request: Request, form: dict) -> None:
        if settings.allow_unsigned_twilio:
            return
        if not settings.twilio_auth_token:
            raise HTTPException(403, "twilio auth token not configured")
        url = settings.public_base_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        if not security.validate_twilio(settings.twilio_auth_token, url, form, request.headers.get("X-Twilio-Signature", "")):
            raise HTTPException(403, "bad twilio signature")

    def require_sip_token(got: str) -> None:
        """Empty VOX_SIP_TOKEN hides the route (404). A wrong token is 403. compare_digest is false for unequal lengths."""
        if not settings.sip_token:
            raise HTTPException(404, "not found")
        if not hmac.compare_digest(got, settings.sip_token):
            raise HTTPException(403, "bad sip token")

    async def sip_fields(request: Request) -> dict:
        """JSON or form body. Asterisk CURL posts urlencoded fields and may omit a content type."""
        ctype = (request.headers.get("content-type") or "").lower()
        if "json" in ctype:
            try:
                body = await request.json()
            except Exception:  # noqa: BLE001
                body = {}
            return body if isinstance(body, dict) else {}
        try:
            form = {k: str(v) for k, v in (await request.form()).items()}
        except Exception:  # noqa: BLE001
            form = {}
        if form:
            return form
        from urllib.parse import parse_qs
        raw = (await request.body()).decode("utf-8", "ignore")
        parsed = {k: v[-1] for k, v in parse_qs(raw, keep_blank_values=True).items()}
        return parsed or {k: v for k, v in request.query_params.items()}

    def clean_channel_id(value: object) -> str:
        channel_id = str(value or "").strip()
        if not channel_id or len(channel_id) > 200 or any(ord(c) < 32 or ord(c) == 127 for c in channel_id):
            raise HTTPException(422, "channelId is required")
        return channel_id

    async def sip_machine(call_id: str) -> str:
        """Act on a stored machine_end result once. A live session is spoken to and hung up; otherwise the ARI
        channel (metadata callSid, never the AudioSocket UUID) is deleted so the originated call does not stay up.
        A hang-up is finished only after DELETE returns 2xx or 404. Concurrent callers wait on that attempt."""
        done = app.state.sip_machine_done.get(call_id)
        if done:
            return done
        inflight = app.state.sip_machine_wait.get(call_id)
        if inflight is not None:
            inflight["waiters"] += 1
            await inflight["event"].wait()
            if inflight.get("error"):
                raise HTTPException(502, inflight["error"])
            return inflight.get("result") or ""
        cur = store.get_call_any(call_id)
        if not cur:
            return ""
        meta = cur.get("metadata") or {}
        answered = str(meta.get("answeredBy") or "")
        if not answered.startswith("machine_end"):
            return ""
        msg = meta.get("voicemail")
        live = runtime.live.get(call_id)
        if live and msg:
            app.state.sip_machine_done[call_id] = "voicemail"
            sess, transport = live

            async def go():
                try:
                    await sess.say_and_wait(msg)
                finally:
                    # A TTS failure must still drop the line; otherwise the machine stays connected.
                    await transport.end()

            import asyncio
            asyncio.get_running_loop().create_task(go())
            return "voicemail"
        if live or cur["status"] != "created":
            return ""
        channel = str(meta.get("callSid") or "")
        if not channel or runtime.http is None:
            return ""
        ctl = control_from_env(runtime.http, settings.public_base_url, "sip")
        if not isinstance(ctl, AsteriskCallControl):
            return ""
        # Mark ended before the DELETE await so a concurrent AudioSocket cannot adopt this row.
        # Do not publish "hangup" until DELETE returns 2xx or 404. A timeout, connect error, or
        # HTTP 5xx drops the in-flight mark and reopens the row this attempt ended.
        import asyncio
        inflight = {"event": asyncio.Event(), "result": "", "error": "", "waiters": 0}
        app.state.sip_machine_wait[call_id] = inflight
        reason = answered[:80]
        we_ended = cur["status"] == "created"
        try:
            if we_ended:
                store.update_call(call_id, status="ended", ended=time.time(), end_reason=reason)
            try:
                await ctl.hangup(channel)
            except (RuntimeError, httpx.TransportError) as e:
                if we_ended:
                    latest = store.get_call_any(call_id)
                    if latest and latest.get("status") == "ended" and (latest.get("end_reason") or "") == reason:
                        store.update_call(call_id, status="created", ended=None, end_reason=None)
                inflight["error"] = str(e) or type(e).__name__
                inflight["event"].set()
                raise HTTPException(502, inflight["error"]) from e
            app.state.sip_machine_done[call_id] = "hangup"
            # The audio leg never joined, so finish() will not release the outbound reservation.
            # A timed-out DELETE reopens the row above and keeps the slot: the call may still be up.
            try:
                limiter.release(cur["tenant_id"], call_id)
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="limiter_release")
            inflight["result"] = "hangup"
            inflight["event"].set()
            return "hangup"
        finally:
            if not inflight["event"].is_set():
                inflight["error"] = inflight.get("error") or "machine hangup failed"
                inflight["event"].set()
            if app.state.sip_machine_wait.get(call_id) is inflight:
                app.state.sip_machine_wait.pop(call_id, None)

    @app.post("/v1/calls/outbound")
    async def outbound(body: OutboundIn, t: dict = Depends(tenant_write)):
        """Places a phone call. "twilio" (default) uses your Twilio account; "sip" originates through Asterisk ARI."""
        if app.state.draining:
            raise HTTPException(503, "node is draining")
        channel = (body.channel or "twilio").strip().lower()
        if channel not in ("twilio", "sip"):
            raise HTTPException(422, "channel must be 'twilio' or 'sip'")
        if channel == "twilio" and not (settings.twilio_account_sid and settings.twilio_auth_token):
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
        agent = await to_thread(store.get_agent, t["id"], body.agentId)
        if not agent:
            raise HTTPException(404, "agent not found")
        # Refuse sip before any row. A missing ARI url must not leave a call behind.
        if channel == "sip":
            control = control_from_env(runtime.http, settings.public_base_url, "sip")
            if not isinstance(control, AsteriskCallControl):
                raise HTTPException(503, "sip outbound needs ASTERISK_ARI_URL")
        cfg = dict(agent["config"])
        if body.record is not None:
            cfg["record"] = body.record
        given_vars = call_variables(cfg, body.variables, strict=True)
        meta = caller_metadata(body.metadata)
        meta["direction"] = "outbound"
        meta["from"] = caller_id
        meta["to"] = body.to
        if given_vars:
            meta["variables"] = given_vars
        if body.voicemailMessage:
            meta["voicemail"] = body.voicemailMessage
        tenant_id = t["id"]
        node_draining = object()
        medium = "sip" if channel == "sip" else "twilio"

        def _admit():
            # hold() sleeps while another node inserts; that wait must not block this event loop.
            # Drain can flip during the wait, and the lease can disappear. Re-check both before the insert.
            # Reserve before dialing. A full node must not place a call the far end will connect to a 1013.
            with outbound_cap.hold(tenant_id, settings.outbound_per_day) as allowed:
                if not allowed:                        # over the cap, or this holder's lease is gone
                    return None
                if app.state.draining:                 # preStop waits on active_calls, which does not include this dial
                    return node_draining
                cid, decision = reserve_slot(tenant_id)
                if decision != "ok":
                    return decision                    # "node" or "tenant"; no row, cap not consumed
                return create_reserved(cid, tenant_id, body.agentId, medium, cfg, meta)

        call = await to_thread(_admit)
        if call is node_draining:
            raise HTTPException(503, "node is draining")
        if call is None:
            raise HTTPException(429, f"outbound limit reached ({settings.outbound_per_day} per 24 h)")
        if call in ("node", "tenant"):
            raise HTTPException(503, "node full" if call == "node" else "concurrency limit")

        async def _release_failed(reason: str) -> None:
            try:
                limiter.release(t["id"], call["id"])
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="limiter_release")
            await to_thread(store.update_call, call["id"], status="ended", ended=time.time(), end_reason=reason)

        if channel == "sip":
            # appArgs is how the Stasis app learns the call id. It POSTs AnsweredBy to /sip/amd/{id}.
            # channelId is stored before the request so a lost response can still DELETE that channel,
            # and an AnsweredBy that arrives during originate merges instead of replacing this document.
            app_args = f"amd,{call['id']}" if body.voicemailMessage else None
            channel_id = uuidlib.uuid4().hex
            await to_thread(store.merge_call_metadata, call["id"], {"callSid": channel_id})
            app.state.sip_by_channel[channel_id] = call["id"]
            try:
                r = await control.originate(body.to, caller_id, app_args, channel_id=channel_id)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                try:
                    await control.hangup(channel_id)
                except Exception:  # noqa: BLE001  (best-effort; the row is still a failure)
                    pass
                await _release_failed("asterisk_unreachable")
                raise HTTPException(502, f"could not reach Asterisk: {type(e).__name__}") from e
            except Exception as e:  # noqa: BLE001
                await _release_failed("asterisk_unreachable")
                raise HTTPException(502, f"could not reach Asterisk: {type(e).__name__}") from e
            if r.status_code >= 300:
                await _release_failed("asterisk_rejected")
                try:
                    msg = r.json().get("message", "")
                except ValueError:
                    msg = ""
                raise HTTPException(502, f"Asterisk refused the call (HTTP {r.status_code}): {msg}")
            try:
                payload = r.json()
                sid = payload.get("id", "") if isinstance(payload, dict) else ""
            except ValueError:
                sid = ""
            if not isinstance(sid, str):
                sid = str(sid)
            if not sid:
                sid = channel_id
            if sid:
                await to_thread(store.merge_call_metadata, call["id"], {"callSid": sid})
                app.state.sip_by_channel[sid] = call["id"]
            try:
                await sip_machine(call["id"])
            except HTTPException:
                # The originate succeeded. A machine DELETE that timed out leaves the row created;
                # 502 here would look like Asterisk never placed the call.
                pass
            await to_thread(log, t, "call.outbound", call["id"], to=body.to)
            fresh = await to_thread(store.get_call_any, call["id"])
            return {**public_call(fresh), "callSid": sid}

        twiml = stream_twiml(call)
        data = {"To": body.to, "From": caller_id, "Twiml": twiml,
                "StatusCallback": f"{settings.public_base_url}/twilio/status/{call['id']}",
                "StatusCallbackEvent": "completed"}
        if body.voicemailMessage:
            data.update({"MachineDetection": "DetectMessageEnd", "AsyncAmd": "true",
                         "AsyncAmdStatusCallback": f"{settings.public_base_url}/twilio/amd/{call['id']}"})

        async def adopted_call():
            """Twilio can open /ws/twilio before this POST returns. That leg is live; finish() owns the slot."""
            cur = await to_thread(store.get_call_any, call["id"])
            if not cur or cur.get("status") != "active":
                return None
            meta_now = cur["metadata"] if isinstance(cur.get("metadata"), dict) else {}
            await to_thread(log, t, "call.outbound", call["id"], to=body.to)
            return {**public_call(cur), "callSid": str(meta_now.get("callSid") or "")}

        try:
            r = await runtime.http.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls.json",
                auth=(settings.twilio_account_sid, settings.twilio_auth_token), data=data, timeout=15)
        except Exception as e:  # noqa: BLE001
            live = await adopted_call()
            if live is not None:
                return live
            try:
                limiter.release(t["id"], call["id"])
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="limiter_release")
            await to_thread(store.update_call, call["id"], status="ended", ended=time.time(), end_reason="twilio_unreachable")
            raise HTTPException(502, f"could not reach Twilio: {type(e).__name__}") from e
        if r.status_code >= 300:
            live = await adopted_call()
            if live is not None:
                return live
            try:
                limiter.release(t["id"], call["id"])
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="limiter_release")
            await to_thread(store.update_call, call["id"], status="ended", ended=time.time(), end_reason="twilio_rejected")
            try:
                msg = r.json().get("message", "")
            except ValueError:
                msg = ""
            raise HTTPException(502, f"Twilio refused the call (HTTP {r.status_code}): {msg}")
        try:
            payload = r.json()
            sid = payload.get("sid", "") if isinstance(payload, dict) else ""
            await to_thread(store.update_call, call["id"], metadata={**meta, "callSid": sid})
        except Exception:  # noqa: BLE001
            live = await adopted_call()
            if live is not None:
                return live
            # 2xx we cannot record is the same as a dial that never took a slot. Release first:
            # a second write failure must not leave the node full.
            try:
                limiter.release(t["id"], call["id"])
            except Exception:  # noqa: BLE001
                METRICS.inc("vox_errors_total", where="limiter_release")
            try:
                await to_thread(store.update_call, call["id"], status="ended", ended=time.time(), end_reason="twilio_rejected")
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(502, f"Twilio returned an unreadable call (HTTP {r.status_code})")
        await to_thread(log, t, "call.outbound", call["id"], to=body.to)
        fresh = await to_thread(store.get_call_any, call["id"])
        return {**public_call(fresh), "callSid": sid}

    @app.post("/twilio/status/{call_id}")
    async def twilio_status(call_id: str, request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = await to_thread(store.get_call_any, call_id)
        if not call:
            raise HTTPException(404, "call not found")
        status = form.get("CallStatus", "")
        # The snapshot above goes stale across this await. Only a row still in `created` ends here,
        # so a join that already claimed it is not rewritten back to busy/no-answer.
        if status in TERMINAL_CALL_STATUS:
            ended = await to_thread(store.update_call_if_status, call_id, "created",
                                    status="ended", ended=time.time(), end_reason=status)
            if ended:
                # The socket never adopted the reservation. A join that already claimed `created`
                # did not end, and its slot stays held.
                try:
                    limiter.release(call["tenant_id"], call_id)
                except Exception:  # noqa: BLE001
                    METRICS.inc("vox_errors_total", where="limiter_release")
        return {"ok": True}

    @app.post("/twilio/whisper/{call_id}")
    async def twilio_whisper(call_id: str, request: Request):
        """Played only to the person who answers a warm transfer, before they are connected to the caller."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        if not await to_thread(store.get_call_any, call_id):
            raise HTTPException(404, "call not found")

        def _said():
            runtime.wait_messages()
            err = runtime.peek_message_error(call_id)
            if err is not None:
                return err
            return [m["text"] for m in store.messages(call_id) if m["role"] == "user" and m["text"] != "[inaudible]"][-2:]

        said = await to_thread(_said)
        if isinstance(said, BaseException):
            raise HTTPException(500, f"{type(said).__name__}: {said}")
        clean = lambda x: re.sub(r"[^\w\s.,?!'-]", "", x)[:160]  # noqa: E731
        summary = " Then they said: ".join(clean(x) for x in said) or "They did not leave details"
        from xml.sax.saxutils import escape as xesc
        return Response('<?xml version="1.0" encoding="UTF-8"?><Response><Say>'
                        f'Incoming transfer from the voice assistant. The caller said: {xesc(summary)}. Connecting you now.'
                        '</Say></Response>', media_type="text/xml")

    @app.post("/twilio/transfer-result/{call_id}")
    async def twilio_transfer_result(call_id: str, request: Request, to: str = ""):
        """Twilio's <Dial action>: hang up when the human talked and finished; if nobody answered, bring the agent back."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = await to_thread(store.get_call_any, call_id)
        if not call:
            raise HTTPException(404, "call not found")
        if form.get("DialCallStatus", "") == "completed":
            return Response('<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>', media_type="text/xml")
        # The Dial already replaced this call's media stream. Drop a slot it still holds, then hold
        # the return leg before yielding so it cannot lose the only slot to another join.
        if call["status"] == "active":
            await to_thread(store.update_call, call_id, status="ended", ended=time.time(), end_reason="transfer_return")
            limiter.release(call["tenant_id"], call_id)
        if app.state.draining:
            return Response(busy_twiml(), media_type="text/xml")
        cid, decision = reserve_slot(call["tenant_id"])
        if decision != "ok":
            return Response(busy_twiml(), media_type="text/xml")   # no new row while this process is full
        who = re.sub(r"[^\w -]", "", to)[:40] or "that person"
        # A variant system prompt replaces cfg["systemPrompt"] in build_session, so the notes are also stored as
        # promptSuffix. Append to a suffix only when it is already the tail of this prompt; anything else was not
        # written by a previous failed transfer and must not be carried onto the next call.
        note = f"\nThe caller asked to be transferred to {who}, who did not answer. Offer to help or to take a message."
        cfg = dict(call["config"])
        cfg["greeting"] = f"Sorry, {who} is not available right now. Is there anything else I can help you with?"
        base_prompt = cfg.get("systemPrompt") or ""
        prev = call["metadata"].get("promptSuffix") if isinstance(call.get("metadata"), dict) else None
        if not (isinstance(base_prompt, str) and isinstance(prev, str) and prev and base_prompt.endswith(prev)):
            prev = ""
        cfg["systemPrompt"] = base_prompt + note
        meta = {**call["metadata"], "resumedFrom": call_id, "promptSuffix": prev + note, "resumeReason": "transfer"}
        resumed = await to_thread(create_reserved, cid, call["tenant_id"], call["agent_id"], "twilio", cfg, meta)
        resumed_sid = str(form.get("CallSid", "") or (resumed["metadata"] or {}).get("callSid") or "")
        if not await bind_status_callback(resumed["id"], resumed_sid):
            return _drop_unbound(resumed)
        return Response(stream_twiml(resumed), media_type="text/xml")

    @app.post("/twilio/amd/{call_id}")
    async def twilio_amd(call_id: str, request: Request):
        """Answering-machine detection result. On a machine, play the prepared message and hang up.

        The webhook may hit a node that does not own the media stream. That node publishes on the
        call bus; the owning node's session is subscribed and speaks."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        twilio_check(request, form)
        call = await to_thread(store.get_call_any, call_id)
        if not call:
            raise HTTPException(404, "call not found")
        answered = form.get("AnsweredBy", "")
        msg = call["metadata"].get("voicemail")
        METRICS.inc("vox_amd_total", answered_by=answered or "unknown")
        if msg and answered.startswith("machine_end"):
            live = runtime.live.get(call_id)
            if live:
                sess, transport = live
                if getattr(sess, "_vm_once", False):
                    return {"action": "voicemail", "delivered": True}
                sess._vm_once = True

                async def go():
                    try:
                        await sess.say_and_wait(msg)
                    finally:
                        await transport.end()

                def _retrieved(done) -> None:
                    if done.cancelled():
                        return
                    try:
                        done.result()
                    except Exception:  # noqa: BLE001  (retrieved so the failure is not logged as abandoned)
                        METRICS.inc("vox_errors_total", where="voicemail")

                import asyncio
                asyncio.get_running_loop().create_task(go()).add_done_callback(_retrieved)
                return {"action": "voicemail", "delivered": True}
            bus = getattr(runtime, "bus", None)
            delivered = bool(bus.publish(call_id, {"op": "voicemail", "text": msg})) if bus is not None else False
            return {"action": "voicemail", "delivered": delivered}
        return {"action": "none", "answeredBy": answered}

    @app.post("/sip/amd/{call_id}")
    async def sip_amd(call_id: str, request: Request, x_vox_sip_token: str = Header(default="")):
        """Asterisk (or the Stasis app) posts AnsweredBy here. Same voicemail action as /twilio/amd when the call
        is live. A machine result with no session deletes the originated ARI channel instead of leaving it up."""
        require_sip_token(x_vox_sip_token)
        if not store.get_call_any(call_id):
            raise HTTPException(404, "call not found")
        fields = await sip_fields(request)
        answered = str(fields.get("AnsweredBy") or fields.get("answeredBy") or "").strip()
        if not answered:
            # Missing or invalid JSON is an empty result. Do not erase a machine_end already stored,
            # and do not run the hang-up on a blank value.
            cur = store.get_call_any(call_id)
            if not cur:
                raise HTTPException(404, "call not found")
            kept = str((cur.get("metadata") or {}).get("answeredBy") or "")
            return {"action": "none", "answeredBy": kept}
        # Merge only answeredBy so a callSid stored while this request was in flight is not dropped.
        if store.merge_call_metadata(call_id, {"answeredBy": answered}) is None:
            raise HTTPException(404, "call not found")
        METRICS.inc("vox_amd_total", answered_by=answered)
        if answered.startswith("machine_end"):
            action = await sip_machine(call_id)
            if action:
                return {"action": action, "answeredBy": answered}
        return {"action": "none", "answeredBy": answered}

    @app.post("/sip/bind")
    async def sip_bind(request: Request, x_vox_sip_token: str = Header(default="")):
        """Dialplan posts CHANNEL(uniqueid) and the AudioSocket UUID before AudioSocket() opens. Machine hangup
        deletes that uniqueid. The AudioSocket UUID is not an ARI channel and must not be redirected or deleted."""
        require_sip_token(x_vox_sip_token)
        fields = await sip_fields(request)
        try:
            audio_uuid = str(uuidlib.UUID(str(fields.get("uuid") or fields.get("audioSocketUuid") or "")))
        except ValueError:
            raise HTTPException(422, "uuid must be a UUID") from None
        channel_id = clean_channel_id(fields.get("channelId") or fields.get("uniqueid") or "")
        app.state.sip_by_uuid[audio_uuid] = channel_id
        found = store.find_sip_call(call_sid=channel_id)
        if found:
            store.merge_call_metadata(found["id"],
                                     {"audioSocketUuid": audio_uuid, "callSid": channel_id, "channel": "sip"})
            app.state.sip_by_channel[channel_id] = found["id"]
        return {"ok": True, "uuid": audio_uuid, "channelId": channel_id}

    @app.get("/sip/transfer")
    async def sip_transfer(request: Request, x_vox_sip_token: str = Header(default="")):
        """Dialplan reads the cold-transfer number after AudioSocket returns. Empty body means Hangup."""
        require_sip_token(x_vox_sip_token)
        try:
            audio_uuid = str(uuidlib.UUID(str(request.query_params.get("uuid") or "")))
        except ValueError:
            return PlainTextResponse("")
        found = store.find_sip_by_audio_uuid(audio_uuid)
        number = str(((found or {}).get("metadata") or {}).get("transferTo") or "")
        if not E164.match(number):
            number = ""
        return PlainTextResponse(number)

    return app
