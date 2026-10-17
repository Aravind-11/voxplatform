"""PII redaction: the helper, stored transcripts, and the call.ended webhook."""
import json

from core.redact import redact_text
from tests.unit.test_api import make, tenant

H = lambda k: {"X-API-Key": k}  # noqa: E731

PII = "Card 4111 1111 1111 1111, email ada@example.com, phone +14155551212 or (415) 555-1212."


def test_redact_text_covers_card_ssn_email_phone_address_and_keeps_short_numbers():
    cards = redact_text("4111 1111 1111 1111 and 4111-1111-1111-1111")
    assert cards == "[REDACTED:card] and [REDACTED:card]"
    assert redact_text("ssn 123-45-6789 end") == "ssn [REDACTED:ssn] end"
    assert redact_text("mail Ada.Lovelace@Example.com today") == "mail [REDACTED:email] today"
    phones = redact_text("call +14155551212 or (415) 555-1212")
    assert phones == "call [REDACTED:phone] or [REDACTED:phone]"
    assert redact_text("ship to 1600 Pennsylvania Avenue and 10 King St.") == (
        "ship to [REDACTED:address] and [REDACTED:address]")
    # 13-digit Luhn is a card; 13- and 14-digit failures and a 12-digit order id are not. 15+ digits are.
    assert redact_text("4222222222222") == "[REDACTED:card]"
    assert redact_text("1234567890123") == "1234567890123"
    assert redact_text("12345678901234") == "12345678901234"
    assert redact_text("123456789012345") == "[REDACTED:card]"
    assert redact_text("12345678901234567890") == "12345678901234567890"
    kept = "meet in 1999 at 11:00 about order 123456789012"
    assert redact_text(kept) == kept
    assert redact_text(redact_text(PII)) == redact_text(PII)


def _transport():
    return type("T", (), {"out_rate": 16000, "call_sid": "",
                          "send_audio": lambda s, *a: None, "clear": lambda s: None,
                          "end": lambda s: None})()


def test_redact_flag_stores_redacted_transcript_and_webhook():
    events = []

    class D:
        def spawn(self, url, secret, event):
            events.append(event)

    c, store, _ = make(dispatcher=D())
    k, _ = tenant(c)
    h = H(k)
    c.put("/v1/webhook", headers=h, json={"url": "https://hook.example/x"})
    agent = c.post("/v1/agents", headers=h, json={"name": "pii", "systemPrompt": "s", "redact": True}).json()
    assert agent["config"]["redact"] is True
    call = c.post("/v1/calls", headers=h, json={"agentId": agent["id"]}).json()
    assert store.get_call_any(call["id"])["config"]["redact"] is True
    sess = c.app.state.runtime.build_session(store.get_call_any(call["id"]), _transport())
    sess.on_event({"type": "user_transcript", "text": PII})
    sess.on_event({"type": "agent_speech", "text": PII})
    stored = " ".join(m["text"] for m in store.messages(call["id"]))
    assert "[REDACTED:card]" in stored and "[REDACTED:email]" in stored and "[REDACTED:phone]" in stored
    assert "4111" not in stored and "ada@example.com" not in stored and "+14155551212" not in stored
    assert "4111 1111 1111 1111" not in json.dumps(store.messages(call["id"]))
    # A row written raw still has to leave the webhook redacted. The stored row is not duplicated.
    store.add_message(call["id"], "user", "ssn 123-45-6789 at 10 King Street", 1.0)
    with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")):
        pass
    ended = [e for e in events if e["type"] == "call.ended"]
    assert len(ended) == 1
    blob = json.dumps(ended[0]["transcript"])
    assert "[REDACTED:card]" in blob and "[REDACTED:ssn]" in blob and "[REDACTED:address]" in blob
    assert "4111" not in blob and "ada@example.com" not in blob and "123-45-6789" not in blob and "King" not in blob
    raw_row = [m for m in store.messages(call["id"]) if "King" in m["text"]]
    assert len(raw_row) == 1                                         # finish does not write a second copy

    plain = c.post("/v1/agents", headers=h, json={"name": "raw", "systemPrompt": "s"}).json()
    assert plain["config"]["redact"] is False
    call2 = c.post("/v1/calls", headers=h, json={"agentId": plain["id"]}).json()
    sess2 = c.app.state.runtime.build_session(store.get_call_any(call2["id"]), _transport())
    sess2.on_event({"type": "user_transcript", "text": PII})
    raw = store.messages(call2["id"])[0]["text"]
    assert "4111 1111 1111 1111" in raw and "ada@example.com" in raw and "+14155551212" in raw
    patched = c.patch(f"/v1/agents/{plain['id']}", headers=h, json={"redact": True}).json()
    assert patched["config"]["redact"] is True
