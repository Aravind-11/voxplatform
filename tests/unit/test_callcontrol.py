"""Asterisk ARI call control. No real Asterisk: a fake HTTP client records the requests."""
import httpx

from api import tools as toolmod
from api.callcontrol import AsteriskCallControl, TwilioCallControl
from api.runtime import Runtime
from api.store import Store
from api.tools import CallContext
from test_api import FakeLLM, FakeTTS, FakeVAD


class _Rec:
    def __init__(self):
        self.seen = []

    async def post(self, url, **kw):
        self.seen.append(("POST", str(url), kw))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    async def delete(self, url, **kw):
        self.seen.append(("DELETE", str(url), kw))
        return httpx.Response(204, request=httpx.Request("DELETE", url))


async def test_asterisk_transfer_redirects_after_speech_and_hangup_deletes_the_channel():
    client = _Rec()
    ctl = AsteriskCallControl(client, "http://asterisk:8088/", "ari", "secret")
    ctx = CallContext(call_id="call_1", call_sid="chan-9", call_control=ctl)
    _, fns = toolmod.build([{"builtin": "transfer", "targets": {"billing": "+15551230001"}}], ctx, None)
    result = await fns["transfer"]({"to": "billing"})
    assert client.seen == []                                          # nothing is posted until the goodbye is spoken
    await result.after()
    method, url, kw = client.seen[0]
    assert method == "POST" and url == "http://asterisk:8088/ari/channels/chan-9/redirect"
    assert "twilio" not in url.lower()
    assert kw["params"]["endpoint"] == "PJSIP/+15551230001" and kw["auth"] == ("ari", "secret")
    await ctl.hangup("chan-9")
    method, url, kw = client.seen[1]
    assert method == "DELETE" and url == "http://asterisk:8088/ari/channels/chan-9"
    assert "twilio" not in url.lower() and kw["auth"] == ("ari", "secret")


async def test_twilio_hangup_marks_the_call_completed():
    client = _Rec()
    ctl = TwilioCallControl(client, "ACx", "t", "https://vox.example.com")
    await ctl.hangup("CA9")
    method, url, kw = client.seen[0]
    assert method == "POST" and url.endswith("/Accounts/ACx/Calls/CA9.json")
    assert kw["data"] == {"Status": "completed"}


def _phone():
    class T:
        out_rate = 8000
        call_sid = "chan-9"

        async def end(self):
            return None

        async def send_audio(self, *a):
            return None

        def clear(self):
            return None
    return T()


async def test_runtime_uses_asterisk_control_when_ari_url_is_set(monkeypatch):
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(204)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, Store(), http=http)
    call = {"id": "c1", "tenant_id": "t1", "metadata": {}, "config": {
        "systemPrompt": "hi", "tools": [{"builtin": "transfer", "targets": {"billing": "+15551230001"}}]}}
    sess = rt.build_session(call, _phone())
    result = await sess.tools["transfer"]({"to": "billing"})
    assert seen == []
    await result.after()
    assert seen and seen[0].startswith("http://asterisk:8088/ari/channels/chan-9/redirect")
    assert "twilio.com" not in seen[0]


async def test_runtime_without_ari_still_requires_twilio_credentials(monkeypatch):
    monkeypatch.delenv("ASTERISK_ARI_URL", raising=False)
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, Store(), http=http)
    call = {"id": "c1", "tenant_id": "t1", "metadata": {}, "config": {
        "systemPrompt": "hi", "tools": [{"builtin": "transfer", "targets": {"billing": "+15551230001"}}]}}
    sess = rt.build_session(call, _phone())
    err = await sess.tools["transfer"]({"to": "billing"})
    assert "Twilio credentials" in err["error"]


def _spec():
    return {"systemPrompt": "hi", "tools": [{"builtin": "transfer", "targets": {"billing": "+15551230001"}}]}


async def test_ari_channel_id_cannot_redirect_or_hang_up_a_different_resource():
    """A channel id is one path segment. Slashes, '?', and '..' must not select another ARI route."""
    from urllib.parse import parse_qs, urlsplit
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.raw_path.decode())
        return httpx.Response(204)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctl = AsteriskCallControl(client, "http://asterisk:8088", "ari", "secret")
    await ctl.transfer("1730000000.42", "+15551230001")
    await ctl.hangup("1730000000.42")
    await ctl.transfer("../channels/victim", "+15551230001")
    await ctl.hangup("..")
    await ctl.transfer("chan-9?endpoint=PJSIP/evil&context=from-internal", "+15551230001")
    await ctl.hangup("PJSIP/alice-0000001a")
    assert seen[0].startswith("/ari/channels/1730000000.42/redirect?")
    assert seen[1] == "/ari/channels/1730000000.42"
    assert "/victim" not in seen[2] and seen[2].split("?")[0].endswith("/redirect")
    assert seen[3] not in ("/ari", "/ari/channels", "/ari/channels/..")
    injected = parse_qs(urlsplit("http://asterisk" + seen[4]).query)
    assert set(injected) == {"endpoint"} and injected["endpoint"] == ["PJSIP/+15551230001"]
    assert seen[4].split("?")[0].endswith("/redirect")
    assert "/ari/channels/PJSIP/" not in seen[5] and "PJSIP%2Falice-0000001a" in seen[5]


async def test_transfer_without_a_channel_is_an_error_not_success():
    client = _Rec()
    spec = [{"builtin": "transfer", "targets": {"billing": "+15551230001"}}]
    ctl = AsteriskCallControl(client, "http://asterisk:8088", "ari", "s")
    _, fns = toolmod.build(spec, CallContext(call_id="c", call_sid="", call_control=ctl), None)
    result = await fns["transfer"]({"to": "billing"})
    assert isinstance(result, dict) and "error" in result and client.seen == []
    tw = TwilioCallControl(client, "AC", "t", "https://vox.example.com")
    _, fns2 = toolmod.build(spec, CallContext(call_id="c", call_sid="", call_control=tw,
                                              twilio={"sid": "AC", "token": "t"}), client)
    result2 = await fns2["transfer"]({"to": "billing"})
    assert isinstance(result2, dict) and "error" in result2 and client.seen == []


async def test_twilio_call_stays_on_twilio_when_ari_is_also_configured(monkeypatch):
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACx")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "t")
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, json={})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, Store(), http=http, public_url="https://vox.example.com")
    call = {"id": "c1", "tenant_id": "t1", "medium": "twilio", "metadata": {}, "config": _spec()}
    sess = rt.build_session(call, _phone())
    result = await sess.tools["transfer"]({"to": "billing"})
    assert seen == []
    await result.after()
    assert seen and "api.twilio.com" in seen[0] and "asterisk" not in seen[0]


async def test_sip_call_does_not_transfer_through_twilio_or_asterisk_on_a_browser_call(monkeypatch):
    monkeypatch.delenv("ASTERISK_ARI_URL", raising=False)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACx")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "t")
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, json={})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, Store(), http=http, public_url="https://vox.example.com")
    sip = {"id": "c1", "tenant_id": "t1", "medium": "sip", "metadata": {}, "config": _spec()}
    err = await rt.build_session(sip, _phone()).tools["transfer"]({"to": "billing"})
    assert isinstance(err, dict) and "error" in err and seen == []

    class _NoChannel:
        out_rate = 8000

        async def end(self):
            return None

        async def send_audio(self, *a):
            return None

        def clear(self):
            return None

    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    browser = {"id": "c2", "tenant_id": "t1", "medium": "webSocket", "metadata": {}, "config": _spec()}
    err = await rt.build_session(browser, _NoChannel()).tools["transfer"]({"to": "billing"})
    assert isinstance(err, dict) and "error" in err and seen == []


def _sip_phone(audio_uuid: str):
    class T:
        out_rate = 8000
        call_sid = audio_uuid

        async def end(self):
            return None

        async def send_audio(self, *a):
            return None

        def clear(self):
            return None
    return T()


def _ari_runtime(monkeypatch):
    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACx")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "t")
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(204)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, Store(), http=http, public_url="https://vox.example.com")
    return rt, seen


async def test_sip_transfer_stores_the_target_instead_of_a_409_redirect(monkeypatch):
    """AudioSocket is not Stasis, so ARI redirect is 409. Do not report success for that redirect.
    Store the number, drop the audio leg, and that number is what the dialplan dials. No store, no success payload."""
    from pathlib import Path
    from api.app import Settings, create_app
    from api import security
    from core.session import ToolResult
    audio = "11111111-1111-1111-1111-111111111111"
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url)))
        return httpx.Response(409, json={"message": "Channel not in Stasis application"})

    monkeypatch.setenv("ASTERISK_ARI_URL", "http://asterisk:8088")
    monkeypatch.setenv("ASTERISK_ARI_USER", "ari")
    monkeypatch.setenv("ASTERISK_ARI_PASSWORD", "pw")
    # A SIP transfer uses the dialplan, not Twilio. Requiring those credentials must fail this test.
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    store = Store()
    rt = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store, http=http, public_url="https://vox.example.com")
    app = create_app(store, rt, security.ConcurrencyLimiter(2), None,
                     Settings(sip_token="s3cret", public_base_url="https://vox.example.com"))
    row = store.create_call("t1", None, "sip", _spec(), {"callSid": "1730000000.42", "audioSocketUuid": audio})
    ended = []
    events = []

    class Phone:
        out_rate = 8000
        call_sid = audio

        async def end(self):
            ended.append("end")

        async def send_audio(self, *a):
            return None

        def clear(self):
            return None

    # Not in the store: redirect would 409, and the model must not be told the transfer started.
    ghost = {"id": "missing", "tenant_id": "t1", "medium": "sip", "metadata": {"callSid": "1730000000.42",
             "audioSocketUuid": audio}, "config": _spec()}
    err = await rt.build_session(ghost, Phone()).tools["transfer"]({"to": "billing"})
    assert isinstance(err, dict) and "error" in err and not isinstance(err, ToolResult)
    assert ended == [] and seen == []

    # No AudioSocket UUID: nothing for the dialplan to look up, so this is not success either.
    no_audio = store.create_call("t1", None, "sip", _spec(), {"callSid": "1730000000.99"})
    err = await rt.build_session(no_audio, Phone()).tools["transfer"]({"to": "billing"})
    assert isinstance(err, dict) and "error" in err and ended == [] and seen == []
    assert "transferTo" not in (store.get_call_any(no_audio["id"])["metadata"] or {})

    sess = rt.build_session(row, Phone())
    result = await sess.tools["transfer"]({"to": "billing"})
    assert isinstance(result, ToolResult) and result.content == {"transferring": "billing"}
    assert seen == [] and ended == []
    # The number is stored before the deferred hangup. A write that lives only in after() is too late:
    # the model has already been told the transfer started, and the dialplan would GET nothing.
    assert store.get_call_any(row["id"])["metadata"]["transferTo"] == "+15551230001"
    sess.on_event = events.append
    await sess._run_action(result)
    assert ended == ["end"] and seen == []
    assert not any(e.get("type") == "error" for e in events)
    assert any(e.get("type") == "call_action_done" for e in events)
    meta = store.get_call_any(row["id"])["metadata"]
    assert meta["transferTo"] == "+15551230001" and meta["audioSocketUuid"] == audio
    assert "1730000000.42" == meta["callSid"]

    dialplan = (Path(__file__).resolve().parents[2] / "deploy/sip/extensions.conf").read_text()
    after = dialplan.split("AudioSocket(", 1)[1]
    assert "/sip/transfer?uuid=${AS_UUID}" in after
    assert 'GotoIf($["${VOX_XFER}" = ""]?done)' in after
    assert "GotoIf($[${LEN(${VOX_XFER})} < 8]?done)" in after
    assert "Dial(PJSIP/${VOX_XFER})" in after and "@voxsun" not in after and "/redirect" not in after
    assert "n(done),Hangup()" in after
    # CURLOPT(httpheader) is cumulative and stays set across AudioSocket. Both CURLs must send the token.
    # A header that is missing, moved after a CURL, or cleared before the transfer GET fails closed (403).
    token_line = "Set(CURLOPT(httpheader)=X-Vox-Sip-Token: ${ENV(VOX_SIP_TOKEN)})"
    type_line = "Set(CURLOPT(httpheader)=Content-Type: application/x-www-form-urlencoded)"
    assert token_line in dialplan and type_line in dialplan
    i_token, i_type = dialplan.index(token_line), dialplan.index(type_line)
    i_bind = dialplan.index("CURL(http://api1:8080/sip/bind,")
    i_audio = dialplan.index("AudioSocket(")
    i_xfer = dialplan.index("/sip/transfer?uuid=${AS_UUID}")
    assert max(i_token, i_type) < i_bind < i_audio < i_xfer
    assert dialplan[min(i_token, i_type):i_xfer].count("CURLOPT(httpheader)") == 2
    tools = (Path(__file__).resolve().parents[2] / "docs/TOOLS.md").read_text()
    assert "A SIP transfer needs `ASTERISK_ARI_URL` and does not need `TWILIO_ACCOUNT_SID` or `TWILIO_AUTH_TOKEN`." in tools

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.get("/sip/transfer", params={"uuid": audio})
        assert denied.status_code == 403
        res = await client.get("/sip/transfer", params={"uuid": audio}, headers={"X-Vox-Sip-Token": "s3cret"})
        assert res.status_code == 200 and res.text == "+15551230001"
        assert f"PJSIP/{res.text}" == "PJSIP/+15551230001" and "@voxsun" not in f"PJSIP/{res.text}"
        upper = await client.get("/sip/transfer", params={"uuid": audio.upper()}, headers={"X-Vox-Sip-Token": "s3cret"})
        assert upper.text == "+15551230001"
        store.update_call(row["id"], status="ended", ended=1.0, end_reason="agent_hangup")
        still = await client.get("/sip/transfer", params={"uuid": audio}, headers={"X-Vox-Sip-Token": "s3cret"})
        assert still.text == "+15551230001"
        other = "22222222-2222-2222-2222-222222222222"
        store.create_call("t1", None, "sip", {}, {"audioSocketUuid": other})
        empty = await client.get("/sip/transfer", params={"uuid": other}, headers={"X-Vox-Sip-Token": "s3cret"})
        assert empty.status_code == 200 and empty.text == ""


async def test_sip_warm_transfer_does_not_redirect(monkeypatch):
    audio = "22222222-2222-2222-2222-222222222222"
    rt, seen = _ari_runtime(monkeypatch)
    call = {"id": "c1", "tenant_id": "t1", "medium": "sip", "metadata": {"callSid": "1730000000.42"},
            "config": {"systemPrompt": "hi", "tools": [
                {"builtin": "transfer", "mode": "warm", "targets": {"billing": "+15551230001"}}]}}
    result = await rt.build_session(call, _sip_phone(audio)).tools["transfer"]({"to": "billing"})
    if not isinstance(result, dict):
        await result.after()
    assert isinstance(result, dict) and "warm" in result["error"] and seen == []
    client = _Rec()
    ctl = AsteriskCallControl(client, "http://asterisk:8088", "ari", "s")
    try:
        await ctl.transfer("1730000000.42", "+15551230001", mode="warm")
        raised = False
    except RuntimeError:
        raised = True
    assert raised and client.seen == []
