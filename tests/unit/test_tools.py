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


def test_model_sees_only_name_description_and_parameters():
    s = json.dumps(http_tools.schema(http_tools.validate(good())))
    for secret in ("s3cret", "api.example.com", "acme", "Authorization"):
        assert secret not in s


def test_api_never_returns_header_values():
    out = json.dumps(toolmod.redact_specs([good(), "hangUp"]))
    assert "s3cret" not in out and "***" in out


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


def _guarded_booker(text):
    ctx = toolmod.CallContext()
    ctx.guard_slots = True
    ctx.last_assistant_text = text
    ctx.turn = 0
    ctx.turn_fn = lambda: ctx.turn
    _, fns = toolmod.build([{"tool": "book_slot", "confirm": True}], ctx, None)
    return ctx, fns["book_slot"]


async def test_readback_of_a_different_hour_does_not_book():
    # "2 pm" is contained in "12 pm", and "9:00" in "19:00". Those are different slots.
    ctx, run = _guarded_booker("Monday at 12 pm")
    refused = await run({"day": "monday", "time": "2 pm"})
    assert refused["executed"] is False
    assert refused["next_step"] == toolmod._READBACK_STEP
    ctx.turn = 1
    assert (await run({"day": "monday", "time": "2 pm", "confirmed": True}))["executed"] is False

    ctx2, run2 = _guarded_booker("Monday at 19:00")
    refused = await run2({"day": "monday", "time": "9:00"})
    assert refused["next_step"] == toolmod._READBACK_STEP

    # the same hour, including a trailing period, still passes the guard and then the confirm rules
    ctx3, run3 = _guarded_booker("Monday at 2 pm.")
    first = await run3({"day": "monday", "time": "2 pm"})
    assert first["executed"] is False and "read these details back" in first["next_step"].lower()
    ctx3.turn = 1
    assert (await run3({"day": "monday", "time": "2 pm", "confirmed": True}))["booked"] is True


async def test_slot_guard_refuses_book_slot_when_tuesday_was_not_read_back():
    ctx, run = _guarded_booker("I can check Monday.")
    out = await run({"day": "tuesday", "time": "11:00"})
    assert out == {"executed": False,
                   "next_step": "Read these values back to the caller before calling the tool.",
                   "details": {"day": "tuesday", "time": "11:00"}}
    ctx.turn = 1
    again = await run({"day": "tuesday", "time": "11:00", "confirmed": True})
    assert again["executed"] is False and again["details"]["day"] == "tuesday"


async def test_slot_guard_allows_tuesday_read_back_and_then_the_confirm_rules():
    ctx, run = _guarded_booker("Tuesday at 11")
    # "11" is shorter than 3 characters, so only "tuesday" has to appear in the assistant text
    first = await run({"day": "tuesday", "time": "11"})
    assert first["executed"] is False and "read these details back" in first["next_step"].lower()
    ctx.turn = 1
    out = await run({"day": "tuesday", "time": "11", "confirmed": True})
    assert out == {"booked": True, "day": "tuesday", "time": "11:00"}


async def test_confirm_tool_without_a_state_name_still_checks_slots(allow_private):
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"ok": 1})

    ctx = toolmod.CallContext()
    ctx.guard_slots = True
    ctx.last_assistant_text = "nothing relevant"
    ctx.turn = 0
    ctx.turn_fn = lambda: ctx.turn
    _, fns = toolmod.build([good(confirm=True)], ctx, client(handler))
    refused = await fns["lookup_order"]({"order_id": "ABC"})
    assert refused["executed"] is False and sent == []
    assert refused["next_step"] == "Read these values back to the caller before calling the tool."
    ctx.last_assistant_text = "The order is ABC."
    await fns["lookup_order"]({"order_id": "ABC"})
    assert sent == []
    ctx.turn = 1
    assert await fns["lookup_order"]({"order_id": "ABC", "confirmed": True}) == {"ok": 1}
    assert sent[0]["order_id"] == "ABC"


def test_readback_prompt_is_added_only_when_slots_are_guarded_and_history_feeds_the_guard():
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD, make, tenant
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    tid = store.create_tenant("t")

    class T:
        out_rate = 8000

        async def send_audio(self, pcm, rate=8000):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

    guarded = store.create_call(tid, None, "webSocket",
                                {"systemPrompt": "Be brief.", "tools": [{"tool": "book_slot", "confirm": True}]}, {})
    sess = rt.build_session(guarded, T())
    assert toolmod.READBACK_INSTRUCTION in sess.history[0]["content"]
    plain = store.create_call(tid, None, "webSocket",
                              {"systemPrompt": "Be brief.", "tools": ["check_availability"]}, {})
    assert toolmod.READBACK_INSTRUCTION not in rt.build_session(plain, T()).history[0]["content"]
    flagged = store.create_call(tid, None, "webSocket",
                                {"systemPrompt": "Be brief.", "confirmSlots": True, "tools": ["book_slot"]}, {})
    sess2 = rt.build_session(flagged, T())
    assert toolmod.READBACK_INSTRUCTION in sess2.history[0]["content"]
    sess2.history.append({"role": "assistant", "content": "Monday works."})

    c, _, _ = make()
    key, _ = tenant(c)
    saved = c.post("/v1/agents", json={"name": "n", "confirmSlots": True, "tools": ["book_slot"]},
                   headers={"X-API-Key": key})
    assert saved.status_code == 200 and saved.json()["config"]["confirmSlots"] is True
    off = c.post("/v1/agents", json={"name": "plain"}, headers={"X-API-Key": key})
    assert "confirmSlots" not in off.json()["config"]


async def test_runtime_readback_uses_the_assistant_history():
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    tid = store.create_tenant("t")

    class T:
        out_rate = 8000

        async def send_audio(self, pcm, rate=8000):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

    call = store.create_call(tid, None, "webSocket",
                             {"systemPrompt": "Be brief.", "confirmSlots": True, "tools": ["book_slot"]}, {})
    sess = rt.build_session(call, T())
    sess.history.append({"role": "assistant", "content": "Monday works."})
    refused = await sess.tools["book_slot"]({"day": "tuesday", "time": "11:00"})
    assert refused["executed"] is False and "tuesday" in json.dumps(refused["details"]).lower()
    sess.history.append({"role": "assistant", "content": "Tuesday at 11"})
    booked = await sess.tools["book_slot"]({"day": "tuesday", "time": "11"})
    assert booked["booked"] is True
