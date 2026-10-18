"""Token-budget view of a conversation.

The stored transcript stays complete. Only the list handed to the model is trimmed:
older turns become a short extractive note, and the newest user message is never dropped.
An assistant tool call is never separated from the tool results that follow it.
Counts are a length heuristic (about 4 characters per token), not a tokenizer.
Tool-call arguments count as well as message text: they are part of the request.
"""
from __future__ import annotations

import json

SUMMARY_PREFIX = "Earlier in this call: "
_SUMMARY_CHARS = 400


def estimate_tokens(text: str) -> int:
    """Rough token count: about four characters per token, and never zero for a real string."""
    return max(1, len(text) // 4)


def _text(message: dict) -> str:
    content = message.get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _tool_extra(message: dict) -> str:
    """Ids, names, and arguments the model also receives for this message."""
    parts: list[str] = []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for tc in calls:
            if not isinstance(tc, dict):
                parts.append(str(tc))
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args = fn.get("arguments")
            if args is None:
                args_s = ""
            elif isinstance(args, str):
                args_s = args
            else:
                args_s = json.dumps(args, separators=(",", ":"), default=str)
            parts.append(" ".join(p for p in (str(tc.get("id") or ""), str(fn.get("name") or ""), args_s) if p))
    if message.get("role") == "tool" and message.get("tool_call_id"):
        parts.append(str(message["tool_call_id"]))
    return " ".join(p for p in parts if p)


def message_tokens(message: dict) -> int:
    parts: list[str] = []
    text = _text(message)
    if text:
        parts.append(text)
    extra = _tool_extra(message)
    if extra:
        parts.append(extra)
    if not parts:
        return 1
    return estimate_tokens("\n".join(parts))


def estimate_messages(msgs) -> int:
    return sum(message_tokens(m) for m in msgs)


def _copy(message: dict) -> dict:
    return dict(message)


def _last_user_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _has_tool_calls(message: dict) -> bool:
    calls = message.get("tool_calls")
    return message.get("role") == "assistant" and isinstance(calls, list) and len(calls) > 0


def _tool_group_start(messages: list[dict], index: int) -> int:
    """If `index` is inside an assistant tool-call plus its results, return the assistant index."""
    if index <= 0 or index >= len(messages) or messages[index].get("role") != "tool":
        return index
    i = index
    while i > 0 and messages[i].get("role") == "tool":
        i -= 1
    if _has_tool_calls(messages[i]) and all(messages[k].get("role") == "tool" for k in range(i + 1, index + 1)):
        return i
    return index


def _units(tail: list[dict]):
    """Half-open spans. A tool call and the tool messages straight after it are one span."""
    i, n = 0, len(tail)
    while i < n:
        if _has_tool_calls(tail[i]):
            j = i + 1
            while j < n and tail[j].get("role") == "tool":
                j += 1
            yield i, j
            i = j
        else:
            yield i, i + 1
            i += 1


def _trim_to_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    # len // 4 <= budget  =>  len <= budget * 4 + 3
    text = text[: token_budget * 4 + 3]
    while text and estimate_tokens(text) > token_budget:
        text = text[:-1]
    return text


def fit_history(messages, budget: int, keep_tail: int = 6) -> list[dict]:
    """Return a copy that fits `budget`, for the model only.

    The leading system message is always kept. When the list is over budget, turns older than
    the last `keep_tail` messages are replaced by one system line whose text is the first
    ~400 characters of what was dropped (no model call). If that is still over budget, the
    oldest remaining messages are dropped until it fits. An assistant tool call is dropped
    together with its tool results, never just one of them. The latest user message stays.
    """
    msgs = [_copy(m) for m in messages]
    if not msgs:
        return []
    keep_tail = max(0, keep_tail)

    head: list[dict] = []
    rest = msgs
    if rest[0].get("role") == "system":
        head = [rest[0]]
        rest = rest[1:]

    def assemble(summary: dict | None, tail: list[dict]) -> list[dict]:
        out = list(head)
        if summary is not None:
            out.append(summary)
        out.extend(tail)
        return out

    if estimate_messages(assemble(None, rest)) <= budget:
        return assemble(None, rest)

    last_user = _last_user_index(rest)
    tail_start = max(0, len(rest) - keep_tail)
    # The tail window must include the latest user message even when keep_tail is smaller than
    # the gap between that message and the end of the list.
    if last_user is not None and tail_start > last_user:
        tail_start = last_user
    # Opening the window on a tool result would leave the model a result with no call.
    if tail_start < len(rest):
        tail_start = _tool_group_start(rest, tail_start)
    dropped, tail = rest[:tail_start], list(rest[tail_start:])

    summary: dict | None = None
    if dropped:
        blob = " ".join(t for t in (_text(m) for m in dropped) if t)
        summary = {"role": "system", "content": SUMMARY_PREFIX + blob[:_SUMMARY_CHARS]}

    def over() -> bool:
        return estimate_messages(assemble(summary, tail)) > budget

    while over() and tail:
        lu = _last_user_index(tail)
        removed = False
        for start, end in _units(tail):
            if lu is not None and start <= lu < end:
                continue
            del tail[start:end]
            removed = True
            break
        if not removed:
            break

    if summary is not None and over():
        room = budget - estimate_messages(assemble(None, tail))
        trimmed = _trim_to_tokens(summary["content"], room)
        summary = {"role": "system", "content": trimmed} if trimmed else None

    return assemble(summary, tail)
