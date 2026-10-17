import copy

from core.history import estimate_messages, estimate_tokens, fit_history


def test_estimate_tokens_is_about_four_characters():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abc") == 1          # never zero
    assert estimate_tokens("") == 1
    assert estimate_messages([{"role": "user", "content": "abcd" * 4}]) == 4


def test_short_history_is_unchanged_at_the_default_budget():
    msgs = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = fit_history(msgs, 6000)
    assert out == msgs and out is not msgs


def test_over_budget_summarizes_older_turns_and_keeps_the_last_user():
    system = {"role": "system", "content": "You are a helpful bot."}
    msgs = [system]
    for i in range(30):
        msgs.append({"role": "user", "content": f"question {i:02d} " + ("abcd" * 20)})
        msgs.append({"role": "assistant", "content": f"answer {i:02d} " + ("efgh" * 20)})
    original = copy.deepcopy(msgs)
    out = fit_history(msgs, 80, keep_tail=4)
    assert msgs == original
    assert estimate_messages(out) <= 80
    assert out[0]["content"].startswith("You are a helpful")
    assert any(m["role"] == "user" and "question 29" in m["content"] for m in out)
    assert any(m["role"] == "system" and m["content"].startswith("Earlier in this call:") for m in out[1:])
    # the summary is extractive, not a second model call: it starts with the oldest dropped text
    summary = next(m["content"] for m in out if m["content"].startswith("Earlier in this call:"))
    assert "question 00" in summary
    assert len(summary) <= len("Earlier in this call: ") + 400


def test_last_user_is_kept_even_when_it_falls_outside_the_tail_window():
    msgs = [{"role": "system", "content": "sys prompt here"}]
    msgs.append({"role": "user", "content": "LATEST-USER " + ("z" * 40)})
    for i in range(8):
        msgs.append({"role": "assistant", "content": f"later {i} " + ("q" * 80)})
    out = fit_history(msgs, 60, keep_tail=2)
    assert estimate_messages(out) <= 60
    assert any("LATEST-USER" in m.get("content", "") for m in out)
    assert out[0]["content"] == "sys prompt here"
