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
