import json
import time

import httpx
import pytest

from api import http_tools, openapi_import, variables
from api.runtime import Runtime
from api.store import Store
from core.session import SessionConfig
from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD, make, silence, speech, tenant
from tests.unit.test_tools import Ctx, client

H = lambda k: {"X-API-Key": k}  # noqa: E731


# ------------------------------------------------------------------ variables
def test_render_fills_known_blanks_unknown_and_ignores_spacing():
    assert variables.render("Hi {{ name }}, re {{deal}}{{nope}}.", {"name": "Dana", "deal": "Acme"}) == "Hi Dana, re Acme."
    assert variables.missing(["Hi {{name}} {{caller}}"], ["x"]) == ["name"]          # built-ins never count as missing


def test_variable_validation():
    assert variables.validate({"n": 5, "s": "x"}) == {"n": "5", "s": "x"}
    for bad in ({"caller": "x"}, {"a b": "x"}, {"k": {"nested": 1}}, {"k": "x" * 501}, {f"v{i}": "x" for i in range(31)}, [1]):
        with pytest.raises(variables.VariableError):
            variables.validate(bad)


def test_call_needs_every_variable_and_values_reach_the_prompt():
    c, store, _ = make()
    k, _ = tenant(c)
    a = c.post("/v1/agents", headers=H(k), json={"name": "a", "systemPrompt": "Call {{customer}} about {{deal}}. From {{caller}}.",
                                                 "greeting": "Hi {{customer}}!", "variables": {"deal": "the renewal"}}).json()
    r = c.post("/v1/calls", headers=H(k), json={"agentId": a["id"]})
    assert r.status_code == 422 and "customer" in r.json()["detail"]
    assert c.post("/v1/calls", headers=H(k), json={"agentId": a["id"], "variables": {"caller": "x", "customer": "y"}}).status_code == 422
    ok = c.post("/v1/calls", headers=H(k), json={"agentId": a["id"], "variables": {"customer": "Dana"}})
    assert ok.status_code == 200 and ok.json()["metadata"]["variables"] == {"customer": "Dana"}
    call = store.get_call_any(ok.json()["id"])

    class T:
        out_rate = 24000
        async def send_audio(self, b): ...
        async def clear(self): ...
        async def end(self): ...
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    sess = rt.build_session(call, T())
    assert sess.history[0]["content"] == "Call Dana about the renewal. From ."          # agent default + call value + built-in
    assert sess.cfg.greeting == "Hi Dana!"


def test_filler_text_on_a_tool_spec_is_passed_to_the_session():
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    tid = store.create_tenant("t")
    tool = {"name": "lookup_order", "description": "Find an order.",
            "parameters": {"type": "object", "properties": {}},
            "http": {"url": "https://api.example.com/orders"},
            "fillerText": "Checking that order."}
    call = store.create_call(tid, None, "webSocket", {"systemPrompt": "Be brief.", "tools": [tool]}, {})

    class T:
        out_rate = 8000
        async def send_audio(self, b): ...
        async def clear(self): ...
        async def end(self): ...

    sess = rt.build_session(call, T())
    assert sess.cfg.tool_fillers == {"lookup_order": "Checking that order."}
    assert sess.cfg.filler_text == "One moment, let me check that."
    assert SessionConfig().filler_delay_s == 0.7 and sess.cfg.filler_delay_s == 0.7


def test_inbound_phone_calls_get_caller_number_and_empty_unknowns():
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "twilio", {"systemPrompt": "Caller {{caller}} dialled {{callee}}. {{missing}}"},
                             {"from": "+15551230000", "to": "+15559990000"})

    class T:
        out_rate = 8000
        async def send_audio(self, b): ...
        async def clear(self): ...
        async def end(self): ...
    assert rt.build_session(call, T()).history[0]["content"] == "Caller +15551230000 dialled +15559990000. "


# ------------------------------------------------------- HTTP tool: path + query
@pytest.fixture
def allow_private(monkeypatch):
    monkeypatch.setenv("VOX_TOOL_ALLOW_PRIVATE", "1")


def deal_tool(**kw):
    d = {"name": "update_deal", "description": "Update a deal.",
         "parameters": {"type": "object", "properties": {"deal_id": {"type": "string"}, "stage": {"type": "string"},
                                                          "notify": {"type": "boolean"}}, "required": ["deal_id"]},
         "http": {"url": "https://crm.example.com/deals/{deal_id}", "method": "PATCH", "query": ["notify"]}}
    d.update(kw)
    return http_tools.validate(d)


async def test_path_params_are_encoded_and_query_is_split_from_the_body(allow_private):
    seen = {}

    def handler(req):
        seen.update(path=req.url.raw_path.decode(), q=dict(req.url.params), body=json.loads(req.content))
        return httpx.Response(200, json={"ok": True})
    run = http_tools.make_http_tool(deal_tool(), Ctx(), client(handler))
    assert await run({"deal_id": "../admin/x?y=1", "stage": "won", "notify": True}) == {"ok": True}
    assert seen["path"].startswith("/deals/..%2Fadmin%2Fx%3Fy%3D1")          # cannot climb out of the path
    assert seen["q"] == {"notify": "true"} and seen["body"] == {"stage": "won"}


async def test_missing_path_value_is_reported_not_requested(allow_private):
    called = []
    run = http_tools.make_http_tool(deal_tool(), Ctx(), client(lambda r: called.append(1) or httpx.Response(200)))
    assert "deal_id" in (await run({"stage": "won"}))["error"] and not called


def test_url_placeholders_must_be_declared():
    with pytest.raises(http_tools.ToolConfigError):
        deal_tool(http={"url": "https://crm.example.com/deals/{nope}", "method": "GET"})


# -------------------------------------------------------------- OpenAPI import
SPEC = {
    "openapi": "3.0.3", "servers": [{"url": "https://crm.example.com/api/"}],
    "components": {"schemas": {
        "Note": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string", "description": "Note body"},
                                                                       "internal": {"type": "boolean", "nullable": True, "x-foo": 1}}},
        "NoteWithAuthor": {"allOf": [{"$ref": "#/components/schemas/Note"},
                                     {"type": "object", "properties": {"author": {"type": "string"}}}]}}},
    "paths": {
        "/leads/{leadId}": {"get": {"operationId": "getLead", "summary": "Fetch a lead",
                                    "parameters": [{"name": "leadId", "in": "path", "required": True, "schema": {"type": "string"}},
                                                   {"name": "expand", "in": "query", "schema": {"type": "string", "enum": ["deals"]}},
                                                   {"name": "X-Trace", "in": "header", "schema": {"type": "string"}}]}},
        "/leads/{leadId}/notes": {"post": {"operationId": "addNote", "description": "Add a note to a lead.",
                                           "parameters": [{"name": "leadId", "in": "path", "required": True, "schema": {"type": "string"}}],
                                           "requestBody": {"required": True, "content": {"application/json": {
                                               "schema": {"$ref": "#/components/schemas/NoteWithAuthor"}}}}}},
        "/upload": {"post": {"operationId": "upload", "requestBody": {"required": True, "content": {
            "multipart/form-data": {"schema": {"type": "object"}}}}}},
        "/bulk": {"post": {"operationId": "bulk", "requestBody": {"content": {"application/json": {
            "schema": {"type": "array", "items": {"type": "string"}}}}}}},
    }}


def test_import_builds_tools_with_path_query_body_and_safe_defaults():
    tools, skipped = openapi_import.import_spec(SPEC, headers={"Authorization": "Bearer k"})
    by = {t["name"]: t for t in tools}
    assert set(by) == {"getLead", "addNote"}
    g, n = by["getLead"], by["addNote"]
    assert g["http"]["url"] == "https://crm.example.com/api/leads/{leadId}" and g["http"]["method"] == "GET"
    assert g["http"]["query"] == ["expand"] and g["parameters"]["required"] == ["leadId"] and "X-Trace" not in g["parameters"]["properties"]
    assert g["confirm"] is False and n["confirm"] is True                    # writes need read-back by default
    assert set(n["parameters"]["properties"]) == {"leadId", "text", "internal", "author"}
    assert n["parameters"]["required"] == ["leadId", "text"]
    assert "x-foo" not in n["parameters"]["properties"]["internal"] and "nullable" not in n["parameters"]["properties"]["internal"]
    assert n["http"]["headers"] == {"Authorization": "Bearer k"} and n["description"] == "Add a note to a lead."
    why = {s["operation"]: s["reason"] for s in skipped}
    assert "application/json" in why["POST /upload"] and "object" in why["POST /bulk"]
    assert openapi_import.import_spec(SPEC, confirm_writes=False)[0][1]["confirm"] is False


def test_import_rejects_unusable_documents():
    with pytest.raises(http_tools.ToolConfigError):
        openapi_import.import_spec({"swagger": "2.0"})
    with pytest.raises(http_tools.ToolConfigError):
        openapi_import.import_spec({"openapi": "3.0.0", "paths": {}})            # no server URL
    t, _ = openapi_import.import_spec({"openapi": "3.0.0", "paths": SPEC["paths"], "components": SPEC["components"]},
                                      base_url="https://x.example.com", include=["getLead"], prefix="crm_")
    assert [x["name"] for x in t] == ["crm_getLead"]


async def test_imported_tool_actually_calls_the_endpoint(allow_private):
    tools, _ = openapi_import.import_spec(SPEC, headers={"Authorization": "Bearer k"}, confirm_writes=False)
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content or b"{}"))
        return httpx.Response(200, json={"id": 1})
    run = http_tools.make_http_tool(next(t for t in tools if t["name"] == "addNote"), Ctx(), client(handler))
    assert await run({"leadId": "L9", "text": "called, wants a demo"}) == {"id": 1}
    assert seen["url"] == "https://crm.example.com/api/leads/L9/notes" and seen["auth"] == "Bearer k"
    assert seen["body"] == {"text": "called, wants a demo"}


def test_import_endpoint_dry_run_save_overwrite_and_agent_use():
    c, store, _ = make()
    k, _ = tenant(c)
    r = c.post("/v1/tools/import-openapi", headers=H(k), json={"spec": SPEC, "dryRun": True})
    assert r.status_code == 200 and r.json()["imported"] == ["getLead", "addNote"] and c.get("/v1/tools", headers=H(k)).json() == []
    r = c.post("/v1/tools/import-openapi", headers=H(k), json={"spec": SPEC, "headers": {"Authorization": "Bearer k"}})
    assert r.json()["imported"] == ["getLead", "addNote"]
    assert all(t["http"]["headers"] == {"Authorization": "***"} for t in r.json()["tools"])       # credentials write-only
    again = c.post("/v1/tools/import-openapi", headers=H(k), json={"spec": SPEC}).json()
    assert again["imported"] == [] and any("exists" in s["reason"] for s in again["skipped"])
    assert c.post("/v1/tools/import-openapi", headers=H(k), json={"spec": SPEC, "overwrite": True}).json()["imported"]
    a = c.post("/v1/agents", headers=H(k), json={"name": "a", "tools": [{"library": "getLead"}, {"library": "addNote"}, "hangUp"]})
    assert a.status_code == 200
    assert c.post("/v1/tools/import-openapi", headers=H(k), json={"spec": {"swagger": "2.0"}}).status_code == 422
    viewer = c.post("/v1/keys", headers=H(k), json={"role": "viewer"}).json()["apiKey"]
    assert c.post("/v1/tools/import-openapi", headers=H(viewer), json={"spec": SPEC}).status_code == 403


# ------------------------------------------------------- richer call.ended webhook
def test_call_ended_webhook_carries_transcript_and_duration():
    events = []

    class D:
        def spawn(self, url, secret, event):
            events.append(event)
    c, store, _ = make(dispatcher=D())
    k, _ = tenant(c)
    c.put("/v1/webhook", headers=H(k), json={"url": "https://hook.example.com/x"})
    call = c.post("/v1/calls", headers=H(k), json={"systemPrompt": "p", "metadata": {"dealId": "D-1"}}).json()
    with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")) as ws:
        for f in silence(5) + speech(15) + silence(40):
            ws.send_bytes(f)
        deadline = time.time() + 5
        while time.time() < deadline:
            if ws.receive().get("bytes"):
                break
    ended = [e for e in events if e["type"] == "call.ended"]
    assert len(ended) == 1
    e = ended[0]
    assert e["call"]["metadata"]["dealId"] == "D-1" and e["durationSeconds"] >= 0
    assert any(m["role"] == "agent" for m in e["transcript"]) and all({"role", "text", "t"} <= set(m) for m in e["transcript"])
