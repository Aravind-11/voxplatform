"""Redact payment and contact details before text is stored or put on a webhook.

A card is one number of 13–19 digits: a contiguous run, or several groups joined by
spaces or dashes. It must pass Luhn, or be at least 15 digits and be the whole run
(a proper prefix would leave the rest of the number in clear). A year, zip, or clock
fragment beside an order id under 13 digits is not a card. Years, times like 11:00,
and those order ids stay.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])")
_PHONE = re.compile(r"(?<![0-9])(?:\+[0-9]{8,15}|\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4})(?![0-9])")
# Digits of "11:00" are not a house number and not the start of a card.
_CLOCK = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
_RUN = re.compile(r"\d+(?:[ \-]+\d+)*")
# (?<!\d:) keeps the minutes of a clock ("11:00 Dr.") from looking like a house number.
_ADDRESS = re.compile(
    r"(?<!\d:)\b[0-9]{1,6}\s+(?:[A-Za-z][A-Za-z'.-]*\s+){0,5}"
    r"(?:Boulevard|Street|Avenue|Road|Drive|Lane|Blvd|Ave|St|Rd|Dr|Ln)\b\.?",
    re.IGNORECASE,
)
_BARE_YEAR = re.compile(
    r"(?:19|20)\d{2}\s+(?:Boulevard|Street|Avenue|Road|Drive|Lane|Blvd|Ave|St|Rd|Dr|Ln)\.?",
    re.IGNORECASE,
)
# Closed-class words between a year and a title ("1999 and Dr.", "1999 the street").
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "of", "in", "on", "at", "to",
    "for", "from", "with", "by", "as", "our", "my", "your", "their", "his",
    "her", "its", "this", "that", "these", "those",
})


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _clock_or_date(raw: str) -> bool:
    """Year plus date or clock parts (2000-01-01-12-00-00). A card group is not only years and small parts."""
    parts = [p for p in re.split(r"[ \-]+", raw) if p]
    if len(parts) < 2 or not all(re.fullmatch(r"\d{2}|\d{4}", p) for p in parts):
        return False
    saw_year = saw_small = False
    for p in parts:
        n = int(p)
        if len(p) == 4:
            if not 1900 <= n <= 2099:
                return False
            saw_year = True
        elif n > 59:
            return False
        else:
            saw_small = True
    return saw_year and saw_small


def _separate_numbers(groups: list[str]) -> bool:
    """A year, zip, or 1–2 digit fragment next to an order id is not one card."""
    has_order = has_sticky = False
    for g in groups:
        n = len(g)
        if 7 <= n <= 12:
            has_order = True
        elif n <= 2 or n == 5 or (n == 4 and 1900 <= int(g) <= 2099):
            has_sticky = True
    return has_order and has_sticky


def _join_groups(groups: list[str], seps: list[str], i: int, j: int) -> str:
    parts = [groups[i]]
    for k in range(i, j):
        parts.append(seps[k])
        parts.append(groups[k + 1])
    return "".join(parts)


def _overlaps(a: tuple[int, int, bool], b: tuple[int, int, bool]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _redact_run(raw: str) -> str:
    groups = re.findall(r"\d+", raw)
    seps = re.findall(r"[ \-]+", raw)
    n = len(groups)
    cands: list[tuple[int, int, bool]] = []

    def consider(i: int, j: int) -> None:
        span = groups[i:j + 1]
        digits = "".join(span)
        if not 13 <= len(digits) <= 19 or _separate_numbers(span):
            return
        if _clock_or_date(_join_groups(groups, seps, i, j)):
            return
        luhn = _luhn(digits)
        whole = i == 0 and j == n - 1
        # Non-Luhn 15+ digits count only for one number: a contiguous run, or the whole
        # separator run. A shorter window is a prefix and would leave a tail in clear.
        if luhn or (len(digits) >= 15 and (i == j or whole)):
            cands.append((i, j, luhn))

    for i in range(n):
        consider(i, i)
        total = len(groups[i])
        for j in range(i + 1, n):
            total += len(groups[j])
            if total > 19:
                break
            consider(i, j)

    luhn_spans = [c for c in cands if c[2]]
    if luhn_spans:
        cands = [c for c in cands if c[2] or not any(_overlaps(c, other) for other in luhn_spans)]
    cands.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    chosen: list[tuple[int, int, bool]] = []
    last = -1
    for c in cands:
        if c[0] <= last:
            continue
        chosen.append(c)
        last = c[1]
    if not chosen:
        return raw

    parts: list[str] = []
    idx = 0
    for i, j, _ in chosen:
        while idx < i:
            parts.append(groups[idx])
            parts.append(seps[idx])
            idx += 1
        parts.append("[REDACTED:card]")
        idx = j + 1
        if idx < n:
            parts.append(seps[j])
    while idx < n:
        parts.append(groups[idx])
        if idx < n - 1:
            parts.append(seps[idx])
        idx += 1
    return "".join(parts)


def _redact_cards(text: str) -> str:
    masked = list(text)
    for m in _CLOCK.finditer(text):
        for i in range(m.start(), m.end()):
            if masked[i].isdigit():
                masked[i] = "x"
    masked_s = "".join(masked)
    parts: list[str] = []
    last = 0
    for m in _RUN.finditer(masked_s):
        parts.append(text[last:m.start()])
        parts.append(_redact_run(text[m.start():m.end()]))
        last = m.end()
    parts.append(text[last:])
    return "".join(parts)


def _address(match: re.Match) -> str:
    raw = match.group(0)
    if _BARE_YEAR.fullmatch(raw):
        return raw  # "1999 St." is a year and a title, not a street
    body = raw[:-1] if raw.endswith(".") else raw
    parts = body.split()
    # "1999 and Dr." / "2019 the St." — function words, not "1999 Oak Dr".
    if (len(parts) >= 3 and re.fullmatch(r"(?:19|20)\d{2}", parts[0])
            and all(p.lower() in _FUNCTION_WORDS for p in parts[1:-1])):
        return raw
    return "[REDACTED:address]"


def redact_text(text: str) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[REDACTED:email]", text)
    text = _SSN.sub("[REDACTED:ssn]", text)
    text = _redact_cards(text)
    text = _PHONE.sub("[REDACTED:phone]", text)
    text = _ADDRESS.sub(_address, text)
    return text
