"""PII redaction: the helper, stored transcripts, and the call.ended webhook."""
import json

import numpy as np

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


def test_years_and_clock_times_stay_while_cards_and_streets_are_redacted():
    assert redact_text("See you at 11:00 Dr. Smith") == "See you at 11:00 Dr. Smith"
    assert redact_text("Meet in 1999 Dr. Adams arrived") == "Meet in 1999 Dr. Adams arrived"
    assert redact_text("since 1999 St. Louis has grown") == "since 1999 St. Louis has grown"
    assert redact_text("2000-01-01-12-00-00") == "2000-01-01-12-00-00"
    assert redact_text("2000 01 01 12 00 00") == "2000 01 01 12 00 00"
    assert redact_text("paid on 1990-01-01-2000-01-01 thanks") == "paid on 1990-01-01-2000-01-01 thanks"
    assert redact_text("4111-1111-1111-1111") == "[REDACTED:card]"
    assert redact_text("4222222222222") == "[REDACTED:card]"
    assert redact_text("1999-1999-1999-199") == "[REDACTED:card]"
    assert redact_text("ship to 10 King St. in 1999 at 11:00") == "ship to [REDACTED:address] in 1999 at 11:00"
    assert redact_text("1999 Oak Dr") == "[REDACTED:address]"
    assert redact_text("10 St") == "[REDACTED:address]"
    # A year plus only function words is a title or a sentence, not a house number.
    assert redact_text("Meet in 1999 and Dr. Adams arrived") == "Meet in 1999 and Dr. Adams arrived"
    assert redact_text("since 1999 the street has changed") == "since 1999 the street has changed"
    assert redact_text("In 2019 the St. Patrick parade started at 11:00.") == (
        "In 2019 the St. Patrick parade started at 11:00.")
    # A card is one number. Clocks, years, zips, and order ids are not glued onto it,
    # and a 16-digit number written as two groups is still a card.
    glued = [
        ("at 11:00 4111 1111 1111 1111", "at 11:00 [REDACTED:card]"),
        ("11:00 1999 123456789012", "11:00 1999 123456789012"),
        ("order 123456789012 4111 1111 1111 1111", "order 123456789012 [REDACTED:card]"),
        ("in 1999 4111 1111 1111 1111", "in 1999 [REDACTED:card]"),
        ("zip 94107 4111 1111 1111 1111", "zip 94107 [REDACTED:card]"),
        ("in 1999, 4111 1111 1111 1111", "in 1999, [REDACTED:card]"),
        ("41111111 11111111", "[REDACTED:card]"),
    ]
    for raw, expected in glued:
        assert redact_text(raw) == expected
        assert redact_text(expected) == expected


def test_uneven_luhn_groups_redact_and_non_luhn_fragments_stay():
    # 5+11, 5+8, and 5+10 are still one card when the joined digits pass Luhn.
    for raw in ("41111 11111111111", "41111-11111111111", "42222 22222222", "37828 2246310005"):
        assert redact_text(raw) == "[REDACTED:card]"
        assert redact_text("[REDACTED:card]") == "[REDACTED:card]"
    # 8+8+4 is 20 digits and fails Luhn. Only a contiguous group or the whole run counts,
    # so a 16-digit prefix must not be redacted and leave the tail in clear.
    tail = "12345678 90123456 7890"
    assert redact_text(tail) == tail
    assert redact_text(redact_text(tail)) == tail
    # A zip or a 1–2 digit fragment beside an order id stays when the span fails Luhn.
    for raw in ("94107 123456789012", "94107-123456789012", "12 123456789012 99"):
        assert redact_text(raw) == raw
        assert redact_text(redact_text(raw)) == raw


def test_glued_card_tail_is_not_stored_or_sent_on_call_ended():
    samples = [
        ("at 11:00 4111 1111 1111 1111", "at 11:00 [REDACTED:card]"),
        ("11:00 1999 123456789012", "11:00 1999 123456789012"),
        ("order 123456789012 4111 1111 1111 1111", "order 123456789012 [REDACTED:card]"),
    ]
    events = []

    class D:
        def spawn(self, url, secret, event):
            events.append(event)

    c, store, _ = make(dispatcher=D())
    k, _ = tenant(c)
    h = H(k)
    c.put("/v1/webhook", headers=h, json={"url": "https://hook.example/x"})
    agent = c.post("/v1/agents", headers=h, json={"name": "pii", "systemPrompt": "s", "redact": True}).json()
    call = c.post("/v1/calls", headers=h, json={"agentId": agent["id"]}).json()
    sess = c.app.state.runtime.build_session(store.get_call_any(call["id"]), _transport())
    for raw, _expected in samples:
        sess.on_event({"type": "user_transcript", "text": raw})
    stored = [m["text"] for m in store.messages(call["id"])]
    assert stored == [expected for _raw, expected in samples]
    with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")):
        pass
    ended = [e for e in events if e["type"] == "call.ended"]
    assert len(ended) == 1
    assert [m["text"] for m in ended[0]["transcript"]] == stored
    blob = json.dumps(ended[0]["transcript"])
    assert "4111 1111 1111 1111" not in blob and "1111 1111 1111" not in blob


def test_uneven_luhn_card_is_not_stored_or_sent_on_call_ended():
    raw = "41111 11111111111"
    events = []

    class D:
        def spawn(self, url, secret, event):
            events.append(event)

    c, store, _ = make(dispatcher=D())
    k, _ = tenant(c)
    h = H(k)
    c.put("/v1/webhook", headers=h, json={"url": "https://hook.example/x"})
    agent = c.post("/v1/agents", headers=h, json={"name": "pii", "systemPrompt": "s", "redact": True}).json()
    call = c.post("/v1/calls", headers=h, json={"agentId": agent["id"]}).json()
    sess = c.app.state.runtime.build_session(store.get_call_any(call["id"]), _transport())
    sess.on_event({"type": "user_transcript", "text": raw})
    stored = [m["text"] for m in store.messages(call["id"])]
    assert stored == ["[REDACTED:card]"]
    with c.websocket_connect(call["joinUrl"].replace("wss://vox.example.com", "")):
        pass
    ended = [e for e in events if e["type"] == "call.ended"]
    assert len(ended) == 1
    assert [m["text"] for m in ended[0]["transcript"]] == ["[REDACTED:card]"]
    blob = json.dumps(ended[0]["transcript"])
    assert "41111" not in blob and "4111111111111111" not in blob


def test_dataset_utterance_is_redacted_only_when_the_flag_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX_DATASET", "1")
    c, store, _ = make()
    k, _ = tenant(c)
    h = H(k)
    audio = np.zeros(160, dtype=np.int16)
    sentence = "card 4111 1111 1111 1111 mailed ada@example.com"
    runtime = c.app.state.runtime

    def index_text(redact: bool):
        record_dir = tmp_path / ("on" if redact else "off")
        runtime.record_dir = str(record_dir)
        body = {"name": "ds", "systemPrompt": "s", "record": True}
        if redact:
            body["redact"] = True
        agent = c.post("/v1/agents", headers=h, json=body).json()
        call = c.post("/v1/calls", headers=h, json={"agentId": agent["id"]}).json()
        cfg = store.get_call_any(call["id"])["config"]
        assert cfg["record"] is True and bool(cfg.get("redact")) is redact
        sess = runtime.build_session(store.get_call_any(call["id"]), _transport())
        sess.on_utterance(audio, sentence, 0.4)
        return json.loads((record_dir / "utterances" / "index.jsonl").read_text())

    redacted = index_text(True)
    assert redacted["text"] == "card [REDACTED:card] mailed [REDACTED:email]"
    assert "4111" not in redacted["text"] and "ada@example.com" not in redacted["text"]
    assert index_text(False)["text"] == sentence
