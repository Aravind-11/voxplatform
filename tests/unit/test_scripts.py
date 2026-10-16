import importlib.util
import os

import httpx
import pytest

spec = importlib.util.spec_from_file_location("twh", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "twilio_set_webhook.py"))
twh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twh)


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_set_voice_url_looks_up_number_then_updates_it():
    calls = []

    def h(req: httpx.Request):
        calls.append((req.method, req.url.path, req.url.params.get("PhoneNumber"), req.content.decode()))
        if req.method == "GET":
            return httpx.Response(200, json={"incoming_phone_numbers": [{"sid": "PN1"}]})
        return httpx.Response(200, json={})
    assert twh.set_voice_url("ACx", "tok", "+16056278460", "https://h/twilio/voice/agt_1", client(h)) == "PN1"
    assert calls[0][:3] == ("GET", "/2010-04-01/Accounts/ACx/IncomingPhoneNumbers.json", "+16056278460")
    assert calls[1][1].endswith("/IncomingPhoneNumbers/PN1.json") and "VoiceUrl=https%3A%2F%2Fh%2Ftwilio%2Fvoice%2Fagt_1" in calls[1][3]
    assert "VoiceMethod=POST" in calls[1][3]


def test_set_voice_url_reports_unknown_number_and_http_errors():
    with pytest.raises(RuntimeError, match="not a number"):
        twh.set_voice_url("ACx", "t", "+1555", "u", client(lambda r: httpx.Response(200, json={"incoming_phone_numbers": []})))
    with pytest.raises(RuntimeError, match="401"):
        twh.set_voice_url("ACx", "t", "+1555", "u", client(lambda r: httpx.Response(401, text="bad auth")))


def test_drain_hook_waits_for_calls_to_finish_and_reports_leftovers():
    import io
    import json
    import urllib.error
    spec2 = importlib.util.spec_from_file_location("drain", os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "drain.py"))
    drain = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(drain)
    counts = [3, 2, 0]
    posted, slept = [], []

    def urlopen(req, timeout=0):
        if not isinstance(req, str) and getattr(req, "method", "") == "POST":
            posted.append(req.headers["X-admin-key"])
            return io.BytesIO(b"{}")
        body = json.dumps({"active_calls": counts.pop(0)}).encode()
        raise urllib.error.HTTPError("u", 503, "draining", {}, io.BytesIO(body))     # healthz is 503 while draining
    assert drain.drain("http://x", "adm", urlopen=urlopen, sleep=slept.append) == 0
    assert posted == ["adm"] and slept == [5, 5]
    t = [0.0]
    def urlopen2(req, timeout=0):
        if getattr(req, "method", "") == "POST":
            return io.BytesIO(b"{}")
        return io.BytesIO(b'{"active_calls": 4}')
    def sleep(s):
        t[0] += s
    assert drain.drain("http://x", "adm", max_s=12, urlopen=urlopen2, sleep=sleep, now=lambda: t[0]) == 4   # gave up with 4 live
