import json

import httpx
import pytest

from api import builtin_tools, http_tools, tools as toolmod
from api.http_tools import ToolConfigError
from core.session import ToolResult


def good(**over):
    d = {"name": "lookup_order", "description": "Find an order.",
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
         "http": {"url": "https://api.example.com/orders", "method": "post", "headers": {"Authorization": "Bearer s3cret"}},
         "static": {"tenant": "acme"}, "automatic": ["callId"]}
    d.update(over)
    return d


class Ctx:
    call_id, caller, callee, call_sid, twilio = "call_1", "+15550001", "+15550002", "CA123", {"sid": "AC1", "token": "tok"}

    def __init__(self):
        self.ended = 0

    async def end(self):
        self.ended += 1


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ----------------------------------------------------------------- validation
def test_valid_definition_is_normalized():
    d = http_tools.validate(good())
    assert d["http"]["method"] == "POST" and d["http"]["timeoutS"] == 8


@pytest.mark.parametrize("bad", [
    good(name="has space"), good(description=""), good(http={"url": "ftp://x.com"}),
    good(http={"url": "https://u:p@x.com"}), good(http={"url": "https://x.com", "method": "TRACE"}),
    good(http={"url": "https://x.com", "timeoutS": 99}), good(static={"order_id": "x"}),
    good(automatic=["password"]), good(parameters={"type": "array"}), good(http=None)])
def test_invalid_definitions_are_rejected_at_creation(bad):
    with pytest.raises(ToolConfigError):
        http_tools.validate(bad)


def test_filler_text_is_stored_and_hidden_from_the_model():
    d = http_tools.validate(good(fillerText="Checking that order."))
    assert d["fillerText"] == "Checking that order."
    shown = json.dumps(http_tools.schema(d))
    assert "fillerText" not in shown and "Checking that order." not in shown
    assert "fillerText" not in http_tools.validate(good())
    assert toolmod.filler_texts([good(fillerText="Checking that order.")]) == {"lookup_order": "Checking that order."}
    # Explicit blank disables that tool. Omitting the field still means the session default.
    # JSON null is the same disable, not a rejected tool.
    assert http_tools.validate(good(fillerText=""))["fillerText"] == ""
    assert http_tools.validate(good(fillerText="   "))["fillerText"] == ""
    assert http_tools.validate(good(fillerText=None))["fillerText"] == ""
    assert toolmod.filler_texts([good(fillerText="")]) == {"lookup_order": ""}
    assert toolmod.filler_texts([good(fillerText="  ")]) == {"lookup_order": ""}
    assert toolmod.filler_texts([good(fillerText=None)]) == {"lookup_order": ""}
    assert toolmod.filler_texts([good()]) == {}
    assert http_tools.validate(good(fillerText="x" * 200))["fillerText"] == "x" * 200
    with pytest.raises(ToolConfigError):
        http_tools.validate(good(fillerText="x" * 201))
    with pytest.raises(ToolConfigError):
        http_tools.validate(good(fillerText=12))


def test_model_sees_only_name_description_and_parameters():
    s = json.dumps(http_tools.schema(http_tools.validate(good())))
    for secret in ("s3cret", "api.example.com", "acme", "Authorization"):
        assert secret not in s


def test_api_never_returns_header_values():
    out = json.dumps(toolmod.redact_specs([good(), "hangUp"]))
    assert "s3cret" not in out and "***" in out


def test_large_tool_schemas_warn_but_are_not_rejected():
    fat = []
    for i in range(8):
        fat.append({"name": f"lookup_{i}", "description": "d" * 1000,
                    "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "x" * 200}}},
                    "http": {"url": f"https://api.example.com/t{i}"}})
    specs = toolmod.validate_specs(fat)          # over the soft cap, still valid
    schemas, _ = toolmod.build(specs)
    assert toolmod.schema_tokens(schemas) > 2000
    warning = toolmod.warn_schema_tokens(schemas)
    assert warning and "2000" in warning
    assert toolmod.warn_schema_tokens(toolmod.build(["hangUp"])[0]) is None


def test_spec_list_rules():
    with pytest.raises(ToolConfigError):
        toolmod.validate_specs(["nope"])
    with pytest.raises(ToolConfigError):
        toolmod.validate_specs([good(), good()])
    assert [d.get("builtin") for d in toolmod.validate_specs(["hangUp"])] == ["hangUp"]


# ------------------------------------------------------------------ SSRF guard
@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://10.0.0.5/x", "http://169.254.169.254/latest/meta-data",
                                 "http://192.168.1.1/", "http://[::1]/", "http://0.0.0.0/"])
def test_internal_addresses_are_blocked(url):
    with pytest.raises(ToolConfigError):
        http_tools.assert_public(url)


def test_public_literal_ok_and_override(monkeypatch):
    http_tools.assert_public("http://93.184.216.34/x")
    monkeypatch.setenv("VOX_TOOL_ALLOW_PRIVATE", "1")
    http_tools.assert_public("http://127.0.0.1/x")


async def test_tool_to_internal_address_returns_blocked_error():
    run = http_tools.make_http_tool(http_tools.validate(good(http={"url": "http://169.254.169.254/x"})), Ctx(),
                                    client(lambda r: httpx.Response(200)))
    assert "blocked" in (await run({"order_id": "1"}))["error"]


# --------------------------------------------------------------- HTTP tool run
@pytest.fixture
def allow_private(monkeypatch):
    monkeypatch.setenv("VOX_TOOL_ALLOW_PRIVATE", "1")     # tests use example.com names with a mock transport


async def test_post_merges_declared_static_and_automatic_and_drops_unknown(allow_private):
    seen = {}

    def handler(req: httpx.Request):
        seen["body"], seen["auth"], seen["method"] = json.loads(req.content), req.headers["authorization"], req.method
        return httpx.Response(200, json={"status": "shipped"})

    run = http_tools.make_http_tool(http_tools.validate(good()), Ctx(), client(handler))
    out = await run({"order_id": "A-7", "tenant": "evil", "extra": 1})      # model tries to override static
    assert out == {"status": "shipped"}
    assert seen["body"] == {"order_id": "A-7", "tenant": "acme", "callId": "call_1"}
    assert seen["auth"] == "Bearer s3cret" and seen["method"] == "POST"


async def test_get_sends_query_params(allow_private):
    seen = {}

    def handler(req):
        seen["q"] = dict(req.url.params)
        return httpx.Response(200, text="plain")

    d = http_tools.validate(good(http={"url": "https://api.example.com/o", "method": "GET"}))
    assert await http_tools.make_http_tool(d, Ctx(), client(handler))({"order_id": "9"}) == "plain"
    assert seen["q"] == {"order_id": "9", "tenant": "acme", "callId": "call_1"}


async def test_http_error_and_timeout_come_back_as_data_for_the_model(allow_private):
    d = http_tools.validate(good())
    out = await http_tools.make_http_tool(d, Ctx(), client(lambda r: httpx.Response(404, json={"m": "none"})))({})
    assert out == {"error": "HTTP 404", "body": {"m": "none"}}

    def boom(req):
        raise httpx.ReadTimeout("slow")

    out = await http_tools.make_http_tool(d, Ctx(), client(boom))({})
    assert out == {"error": "tool request failed: ReadTimeout"}


async def test_redirects_are_not_followed_and_long_bodies_truncated(allow_private):
    d = http_tools.validate(good())
    r1 = await http_tools.make_http_tool(d, Ctx(), client(lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/"})))({})
    assert r1["error"] == "HTTP 302"
    r2 = await http_tools.make_http_tool(d, Ctx(), client(lambda r: httpx.Response(200, text="x" * 10000)))({})
    assert len(r2) == http_tools.MAX_RESPONSE_CHARS


@pytest.mark.parametrize("header", ["X-Vox-Response-Type", "X-Ultravox-Response-Type"])
async def test_hang_up_response_header_ends_the_call_after_speech(header, allow_private):
    ctx = Ctx()
    run = http_tools.make_http_tool(http_tools.validate(good()), ctx,
                                    client(lambda r: httpx.Response(200, json={"done": 1}, headers={header: "hang-up"})))
    out = await run({})
    assert isinstance(out, ToolResult) and out.content == {"done": 1} and out.strict
    await out.after()
    assert ctx.ended == 1


# --------------------------------------------------------------------- built-ins
async def test_hangup_builtin_respects_strict_flag():
    ctx = Ctx()
    out = await builtin_tools.make(builtin_tools.validate({"builtin": "hangUp", "strict": False}), ctx, None)({"reason": "done"})
    assert isinstance(out, ToolResult) and out.strict is False and out.after == ctx.end


def test_transfer_validation_and_schema_hides_numbers():
    with pytest.raises(ToolConfigError):
        builtin_tools.validate({"builtin": "transfer", "targets": {"billing": "555-1234"}})
    d = builtin_tools.validate({"builtin": "transfer", "targets": {"billing": "+15551234567", "sales": "+442071838750"}})
    s = json.dumps(builtin_tools.schema(d))
    assert "+1555" not in s and '"enum": ["billing", "sales"]' in s


async def test_transfer_only_dials_configured_numbers_via_twilio_after_speech():
    calls = []

    def handler(req: httpx.Request):
        calls.append((str(req.url), req.headers["authorization"][:6], req.content.decode()))
        return httpx.Response(200, json={})

    d = builtin_tools.validate({"builtin": "transfer", "targets": {"billing": "+15551234567"}})
    run = builtin_tools.make(d, Ctx(), client(handler))
    assert "error" in await run({"to": "+19998887777"})                 # raw numbers from the model are refused
    out = await run({"to": "billing"})
    assert calls == []                                                   # nothing happens until the agent has spoken
    await out.after()
    url, auth, body = calls[0]
    assert url == "https://api.twilio.com/2010-04-01/Accounts/AC1/Calls/CA123.json" and auth == "Basic "
    assert "%2B15551234567" in body and "Dial" in body


async def test_transfer_unavailable_without_twilio_credentials():
    ctx = Ctx()
    ctx.twilio = None
    d = builtin_tools.validate({"builtin": "transfer", "targets": {"billing": "+15551234567"}})
    assert "error" in await builtin_tools.make(d, ctx, client(lambda r: httpx.Response(200)))({"to": "billing"})


def test_build_returns_matching_schemas_and_functions():
    schemas, fns = toolmod.build(["check_availability", "hangUp", good()], Ctx(), None)
    assert [s["function"]["name"] for s in schemas] == ["check_availability", "hangUp", "lookup_order"]
    assert set(fns) == {"check_availability", "hangUp", "lookup_order"}


@pytest.mark.parametrize("said,want", [("2 pm", "14:00"), ("2:00 pm", "14:00"), ("11:00", "11:00"), ("9 a.m.", "09:00"),
                                       ("12 am", "00:00"), ("12 pm", "12:00"), ("14", "14:00"), ("noonish", "noonish")])
def test_spoken_times_are_normalized_for_the_demo_booker(said, want):
    assert toolmod.norm_time(said) == want


async def test_demo_booker_accepts_how_the_model_actually_says_times():
    _, fns = toolmod.build(["book_slot"])
    assert (await fns["book_slot"]({"day": "Monday", "time": "2 pm"}))["booked"] is True
    assert (await fns["book_slot"]({"day": "monday", "time": "14:00"}))["booked"] is False     # already taken


# ------------------------------------------------------------ confirm before acting (mishearing guard)
def _confirmed_booker():
    ctx = Ctx()
    ctx.turn = 0
    ctx.turn_fn = lambda: ctx.turn
    schemas, fns = toolmod.build([{"tool": "book_slot", "confirm": True}], ctx, None)
    return ctx, schemas[0], fns["book_slot"]


async def test_confirm_tool_does_not_act_on_the_first_call_and_exposes_confirmed_flag():
    ctx, schema, run = _confirmed_booker()
    assert "confirmed" in schema["function"]["parameters"]["properties"]
    out = await run({"day": "tuesday", "time": "11:00"})
    assert out["executed"] is False and out["details"] == {"day": "tuesday", "time": "11:00"}
    assert "read these details back" in out["next_step"].lower()


async def test_confirmed_call_in_the_same_caller_turn_is_refused():
    ctx, _, run = _confirmed_booker()
    await run({"day": "tuesday", "time": "11:00"})
    out = await run({"day": "tuesday", "time": "11:00", "confirmed": True})    # model "confirms" by itself
    assert out["executed"] is False


async def test_confirmation_in_a_later_turn_with_same_details_runs_it_once():
    ctx, _, run = _confirmed_booker()
    await run({"day": "Tuesday ", "time": "11:00"})
    ctx.turn = 1                                                              # the caller spoke again and said yes
    out = await run({"day": "tuesday", "time": "11:00", "confirmed": True})   # case/whitespace differences are fine
    assert out == {"booked": True, "day": "tuesday", "time": "11:00"}
    again = await run({"day": "tuesday", "time": "11:00", "confirmed": True})
    assert again["executed"] is False                                         # one confirmation = one action


async def test_changed_details_after_confirmation_must_be_confirmed_again():
    ctx, _, run = _confirmed_booker()
    await run({"day": "tuesday", "time": "11:00"})
    ctx.turn = 1
    out = await run({"day": "monday", "time": "14:00", "confirmed": True})    # caller said "actually Monday"
    assert out["executed"] is False and out["details"]["day"] == "monday"
    ctx.turn = 2
    assert (await run({"day": "monday", "time": "14:00", "confirmed": True}))["booked"] is True


async def test_http_tool_with_confirm_never_sends_the_confirmed_flag_to_the_endpoint(allow_private):
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"ok": 1})

    ctx = Ctx()
    ctx.turn = 0
    ctx.turn_fn = lambda: ctx.turn
    spec = good(confirm=True)
    _, fns = toolmod.build([spec], ctx, client(handler))
    await fns["lookup_order"]({"order_id": "A1"})
    assert sent == []                                                          # nothing reaches the tenant's API yet
    ctx.turn = 1
    assert await fns["lookup_order"]({"order_id": "A1", "confirmed": True}) == {"ok": 1}
    assert "confirmed" not in sent[0] and sent[0]["order_id"] == "A1"


def test_tool_options_validate():
    assert toolmod.validate_specs([{"tool": "book_slot", "confirm": True}]) == [{"demo": "book_slot", "confirm": True}]
    with pytest.raises(ToolConfigError):
        toolmod.validate_specs([{"tool": "nope"}])
