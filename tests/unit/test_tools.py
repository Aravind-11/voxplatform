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


async def test_readback_ignores_an_older_assistant_message_and_the_caller_transcript():
    """Only the latest finished assistant message counts, and a later call still has the slot."""
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
    assert sess._gen is None
    book = {"day": "tuesday", "time": "11:00"}
    refusal = {"executed": False, "next_step": toolmod._READBACK_STEP, "details": dict(book)}
    sess.history += [{"role": "assistant", "content": "Tuesday at 11:00."},
                     {"role": "assistant", "content": "Monday works."}]
    assert await sess.tools["book_slot"](dict(book)) == refusal
    del sess.history[1:]
    sess.history += [{"role": "assistant", "content": "Monday works."},
                     {"role": "user", "content": "Tuesday at 11:00."}]
    assert await sess.tools["book_slot"](dict(book)) == refusal
    sess.history.append({"role": "assistant", "content": "Tuesday at 11:00."})
    assert await sess.tools["book_slot"](dict(book)) == {"booked": True, "day": "tuesday", "time": "11:00"}


async def test_resolved_library_confirm_turns_the_readback_guard_on(allow_private):
    from api.runtime import Runtime
    from api.store import Store
    from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD
    sent = []

    def handler(req):
        sent.append(str(req.url))
        return httpx.Response(200, json={"ok": 1})

    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store)
    rt.http = client(handler)
    tid = store.create_tenant("t")
    spec = good(confirm=True)
    store.put_tool(tid, spec["name"], spec)
    call = store.create_call(tid, None, "webSocket",
                             {"systemPrompt": "Be brief.", "tools": [{"library": spec["name"]}]}, {})

    class T:
        out_rate = 8000

        async def send_audio(self, pcm, rate=8000):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

    sess = rt.build_session(call, T())
    assert toolmod.READBACK_INSTRUCTION in sess.history[0]["content"]
    out = await sess.tools[spec["name"]]({"order_id": "ABC-123"})
    assert out["executed"] is False and out["next_step"] == toolmod._READBACK_STEP and sent == []


@pytest.mark.parametrize("said,value", [
    ("9:10 am", "10 am"),
    ("9:10:00", "10:00"),
    ("9:10:30", "10:30"),
    ("1:12:30", "12:30"),
    ("100.50", "100"),
    ("$20.99", "$20"),
    ("1.000", "1,000"),
    ("1,000", "1.000"),
    ("$1.000", "$1,000"),
    ("$1,000", "$1.000"),
    ("1/10/2", "10/2"),
    ("anne-marie", "anne"),
    ("twenty-one", "twenty"),
    ("two-thirty", "two"),
    # A sign, accounting parentheses, or a percent is a different amount from the bare number.
    ("100.50", "-100.50"),
    ("100.50", "100.50-"),
    ("20", "\u221220"),
    ("20", "--20"),
    ("100", "(100)"),
    ("100", "(-100)"),
    ("15", "15%"),
    ("15", "+15"),
])
async def test_readback_does_not_match_a_different_slot_hidden_by_punctuation(said, value):
    _, run = _guarded_booker(said)
    out = await run({"day": "x", "time": value})
    assert out["executed"] is False and out["next_step"] == toolmod._READBACK_STEP


async def test_clock_time_with_a_trailing_period_still_passes_the_guard():
    _, run = _guarded_booker("Tuesday at 11:00.")
    first = await run({"day": "tuesday", "time": "11:00"})
    assert first["executed"] is False and "read these details back" in first["next_step"].lower()


def test_currency_edge_still_matches_the_bare_amount():
    # Shorter than the guard's length cutoff, so this locks the tokenizer itself.
    assert toolmod._slot_was_said("20", "$20") and toolmod._slot_was_said("$20", "20")
    assert toolmod._slot_was_said("100", "£100") and toolmod._slot_was_said("100", "€100.")


async def test_slot_guard_refuses_plain_book_slot_when_the_readback_lookup_raises():
    # A lookup error is empty text, not a skip: tuesday 11:00 must still be free afterwards.
    ctx = toolmod.CallContext()
    ctx.guard_slots = True

    def lookup():
        raise RuntimeError("history unavailable")

    ctx.last_assistant_text = lookup
    _, fns = toolmod.build(["book_slot"], ctx, None)
    out = await fns["book_slot"]({"day": "tuesday", "time": "11:00"})
    assert out == {"executed": False, "next_step": toolmod._READBACK_STEP,
                   "details": {"day": "tuesday", "time": "11:00"}}
    ctx.last_assistant_text = "Tuesday at 11:00"
    assert await fns["book_slot"]({"day": "tuesday", "time": "11:00"}) == {
        "booked": True, "day": "tuesday", "time": "11:00"}


async def test_short_slot_need_not_appear_in_the_read_back():
    # "11" is under the length cutoff and the characters 11 are not in the read-back.
    ctx = toolmod.CallContext()
    ctx.guard_slots = True
    ctx.last_assistant_text = "Tuesday."
    _, fns = toolmod.build(["book_slot"], ctx, None)
    assert await fns["book_slot"]({"day": "tuesday", "time": "11"}) == {
        "booked": True, "day": "tuesday", "time": "11:00"}


async def test_slot_guard_requires_every_long_string_not_only_the_last():
    # 11:00 was said; tuesday was not. Checking only the last argument would book.
    ctx = toolmod.CallContext()
    ctx.guard_slots = True
    ctx.last_assistant_text = "The time is 11:00."
    _, fns = toolmod.build(["book_slot"], ctx, None)
    out = await fns["book_slot"]({"day": "tuesday", "time": "11:00"})
    assert out == {"executed": False, "next_step": toolmod._READBACK_STEP,
                   "details": {"day": "tuesday", "time": "11:00"}}
    ctx.last_assistant_text = "Tuesday at 11:00"
    assert await fns["book_slot"]({"day": "tuesday", "time": "11:00"}) == {
        "booked": True, "day": "tuesday", "time": "11:00"}


async def test_slot_guard_covers_named_state_changes_but_not_reads_or_hangup(allow_private):
    sent = []

    def handler(req):
        sent.append(str(req.url))
        return httpx.Response(200, json={"ok": 1})

    def spec(name):
        return {"name": name, "description": "Change a record.",
                "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
                "http": {"url": f"https://api.example.com/{name}", "method": "POST"}}

    ctx = toolmod.CallContext()
    ctx.guard_slots = True
    ctx.last_assistant_text = ""
    specs = [spec(name) for name in ("create_order", "update_order", "delete_order", "purchase_order")]
    specs.append({"builtin": "transfer", "targets": {"billing": "+15551234567"}})
    specs += ["hangUp", "check_availability"]
    _, fns = toolmod.build(specs, ctx, client(handler))
    for name in ("create_order", "update_order", "delete_order", "purchase_order"):
        out = await fns[name]({"item": "abc-1"})
        assert out == {"executed": False, "next_step": toolmod._READBACK_STEP, "details": {"item": "abc-1"}}
    transferred = await fns["transfer"]({"to": "billing"})
    assert transferred["executed"] is False and transferred["next_step"] == toolmod._READBACK_STEP
    assert sent == []
    hung = await fns["hangUp"]({"reason": "caller-done"})
    assert isinstance(hung, ToolResult) and hung.content["ok"] is True
    assert await fns["check_availability"]({"day": "tuesday"}) == {"day": "tuesday", "free": ["11:00"]}


async def test_same_turn_retry_cannot_use_the_proposal_as_its_own_read_back():
    """The proposal being spoken is not a prior read-back; the next caller turn may use it once it has finished."""
    import asyncio

    import numpy as np

    from api.runtime import Runtime
    from api.store import Store
    from core.session import ToolCall
    from tests.unit.test_api import FakeVAD

    book = {"day": "tuesday", "time": "11:00"}
    refusal = {"executed": False, "next_step": toolmod._READBACK_STEP, "details": dict(book)}

    class LLM:
        def __init__(self):
            self.sess = None
            self.n = 0

        async def transcribe(self, audio):
            return "tuesday at eleven"

        async def stream(self, history, audio, followups, tools):
            n = self.n
            self.n += 1
            if n == 0:
                await asyncio.sleep(0)
                yield "Booking Tuesday at 11:00. "
                yield ToolCall("b1", "book_slot", dict(book))
            elif n == 1:
                await self._until(self._proposal_visible)
                yield ToolCall("b2", "book_slot", dict(book))
            elif n == 2:
                await asyncio.sleep(0)
                yield "I have noted that."
            elif n == 3:
                await asyncio.sleep(0)
                yield "Okay. "
                await self._until(self._okay_visible)
                yield ToolCall("b3", "book_slot", dict(book))
            elif n == 4:
                await asyncio.sleep(0)
                yield "Done."
            else:
                raise AssertionError(f"unexpected model call {n}")

        def _proposal_visible(self):
            text = str(self.sess.history[-1].get("content") or "")
            return "Tuesday" in text and "11:00" in text

        def _okay_visible(self):
            return str(self.sess.history[-1].get("content") or "").startswith("Okay")

        async def _until(self, pred):
            for _ in range(2000):
                if self.sess is not None and pred():
                    return
                await asyncio.sleep(0)
            raise AssertionError(self.sess.history if self.sess else None)

    class TTS:
        sample_rate = 8000

        async def synthesize(self, text):
            await asyncio.sleep(0)
            yield np.zeros(0, dtype=np.int16)

    class T:
        out_rate = 8000

        async def send_audio(self, pcm, rate=8000):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

    store = Store()
    llm = LLM()
    rt = Runtime(llm, lambda v: TTS(), FakeVAD, store)
    tid = store.create_tenant("t")
    call = store.create_call(tid, None, "webSocket",
                             {"systemPrompt": "Be brief.", "confirmSlots": True, "tools": ["book_slot"]}, {})
    sess = rt.build_session(call, T())
    llm.sess = sess
    seen = []
    inner = sess.tools["book_slot"]

    async def spy(args):
        out = await inner(args)
        seen.append(out)
        return out

    sess.tools["book_slot"] = spy
    speech = np.full(512, 5000, dtype=np.int16)
    silence = np.zeros(512, dtype=np.int16)

    async def one_turn():
        for _ in range(15):
            await sess.feed(speech)
            await asyncio.sleep(0)
        for _ in range(25):
            await sess.feed(silence)
            await asyncio.sleep(0)

    async def finished(n_seen):
        for _ in range(2000):
            gen = sess._gen
            if len(seen) >= n_seen and gen is not None and gen.tasks and all(t.done() for t in gen.tasks):
                return
            await asyncio.sleep(0)
        raise AssertionError({"seen": seen, "turns": sess.stats["turns"], "history": sess.history})

    await one_turn()
    await finished(2)
    assert sess.stats["turns"] == 1
    assert seen == [refusal, refusal]
    await one_turn()
    await finished(3)
    assert sess.stats["turns"] == 2
    assert seen[2] == {"booked": True, "day": "tuesday", "time": "11:00"}
    await sess.close()
