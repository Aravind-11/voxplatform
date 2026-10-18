import copy
import json

from core.history import estimate_messages, estimate_tokens, fit_history


def _tool_pairs_intact(msgs) -> bool:
    """An assistant tool call and the tool results that follow it must stay together."""
    pending: list[str] = []
    for m in msgs:
        calls = m.get("tool_calls") or []
        if m.get("role") == "assistant" and calls:
            if pending:
                return False
            pending = [c.get("id") for c in calls]
        elif m.get("role") == "tool":
            if not pending or m.get("tool_call_id") != pending[0]:
                return False
            pending = pending[1:]
        elif pending:
            return False
    return not pending


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
    latest = "LATEST-USER " + ("z" * 40)
    msgs = [{"role": "system", "content": "sys prompt here"}]
    msgs.append({"role": "user", "content": latest})
    for i in range(8):
        msgs.append({"role": "assistant", "content": f"later {i} " + ("q" * 80)})
    out = fit_history(msgs, 60, keep_tail=2)
    assert estimate_messages(out) <= 60
    # The text must survive as the user turn. A summary that quotes it is not that turn.
    assert any(m.get("role") == "user" and m.get("content") == latest for m in out)
    assert out[0]["content"] == "sys prompt here"


def _call(id_: str, arguments: str = "{}") -> dict:
    return {"id": id_, "type": "function", "function": {"name": "lookup", "arguments": arguments}}


def test_trimming_does_not_split_a_tool_call_from_its_result():
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "u0 " + "a" * 100})
    msgs.append({"role": "assistant", "content": "a0 " + "b" * 100})
    msgs.append({"role": "assistant", "content": "calling",
                 "tool_calls": [_call("c0"), _call("c1")]})
    msgs.append({"role": "tool", "tool_call_id": "c0", "content": "res " + "z" * 80})
    msgs.append({"role": "tool", "tool_call_id": "c1", "content": "res " + "z" * 80})
    msgs.append({"role": "assistant", "content": "done " + "d" * 80})
    msgs.append({"role": "user", "content": "LATEST " + "e" * 20})
    msgs.append({"role": "assistant", "content": "final " + "f" * 80})
    original = copy.deepcopy(msgs)
    out = fit_history(msgs, 70, keep_tail=8)
    assert msgs == original
    assert _tool_pairs_intact(out)
    assert estimate_messages(out) <= 70
    assert out[0]["content"] == "sys"
    assert any("LATEST" in (m.get("content") or "") for m in out)
    # keep_tail=3 opens on the tool result (the call sits one message earlier).
    window = [{"role": "system", "content": "sys"}]
    for i in range(6):
        window.append({"role": "user", "content": f"q{i} " + "abcd" * 30})
        window.append({"role": "assistant", "content": f"a{i} " + "efgh" * 30})
    window.append({"role": "assistant", "content": None, "tool_calls": [_call("c9")]})
    window.append({"role": "tool", "tool_call_id": "c9", "content": "result " + "z" * 40})
    window.append({"role": "assistant", "content": "after " + "q" * 40})
    window.append({"role": "user", "content": "LATEST"})
    cut = fit_history(window, 200, keep_tail=3)
    assert _tool_pairs_intact(cut)
    assert estimate_messages(cut) <= 200
    assert any(m.get("role") == "user" and m.get("content") == "LATEST" for m in cut)
    assert cut[0]["content"] == "sys"
    call_at = next(i for i, m in enumerate(cut)
                   if m.get("role") == "assistant"
                   and any(c.get("id") == "c9" for c in (m.get("tool_calls") or [])))
    result_at = next(i for i, m in enumerate(cut)
                     if m.get("role") == "tool" and m.get("tool_call_id") == "c9")
    assert call_at < result_at
    assert all(m.get("role") == "tool" for m in cut[call_at + 1:result_at + 1])
    for m in cut:
        if "c9" not in json.dumps(m):
            continue
        is_call = m is cut[call_at]
        is_result = m is cut[result_at]
        assert is_call or is_result


def test_system_prompt_is_kept_whole_when_it_exceeds_the_budget():
    prompt = "S" * 500
    assert estimate_messages([{"role": "system", "content": prompt}]) > 10
    out = fit_history(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "old " + "a" * 80},
        ],
        10,
    )
    assert out[0]["content"] == prompt


def test_tool_call_arguments_count_toward_the_budget():
    args = "x" * 2000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "tool_calls": [_call("c1", args)]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
        {"role": "assistant", "content": "all set"},
        {"role": "user", "content": "thanks"},
    ]
    out = fit_history(msgs, 40)
    assert estimate_messages(out) <= 40
    assert "x" * 100 not in json.dumps(out)
    assert _tool_pairs_intact(out)
    assert out[0]["content"] == "sys"
    assert any(m.get("content") == "thanks" for m in out)
