"""Token-budget view of a conversation.

The stored transcript stays complete. Only the list handed to the model is trimmed:
older turns become a short extractive note, and the newest user message is never dropped.
Counts are a length heuristic (about 4 characters per token), not a tokenizer.
"""
from __future__ import annotations

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


def estimate_messages(msgs) -> int:
    return sum(estimate_tokens(_text(m)) for m in msgs)


def _copy(message: dict) -> dict:
    return dict(message)


def _last_user_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


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
    oldest remaining messages are dropped in pairs until it fits. The latest user message stays.
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
    dropped, tail = rest[:tail_start], list(rest[tail_start:])

    summary: dict | None = None
    if dropped:
        blob = " ".join(t for t in (_text(m) for m in dropped) if t)
        summary = {"role": "system", "content": SUMMARY_PREFIX + blob[:_SUMMARY_CHARS]}

    def over() -> bool:
        return estimate_messages(assemble(summary, tail)) > budget

    while over() and tail:
        lu = _last_user_index(tail)
        droppable = [i for i in range(len(tail)) if i != lu]
        if not droppable:
            break
        del tail[droppable[0]]
        lu = _last_user_index(tail)
        droppable = [i for i in range(len(tail)) if i != lu]
        if droppable and over():
            del tail[droppable[0]]

    if summary is not None and over():
        room = budget - estimate_messages(assemble(None, tail))
        trimmed = _trim_to_tokens(summary["content"], room)
        summary = {"role": "system", "content": trimmed} if trimmed else None

    return assemble(summary, tail)
