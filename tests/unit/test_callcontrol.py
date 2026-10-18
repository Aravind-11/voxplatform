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
