"""Redact payment and contact details before text is stored or put on a webhook.

A digit run is a card only when it is 13–19 digits after spaces and dashes are removed, and it passes Luhn or
is at least 15 digits. Years, clock times, and order ids under 13 digits stay.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])")
_CARD = re.compile(r"(?<![0-9])[0-9](?:[ \-]*[0-9]){12,18}(?![0-9])")
_PHONE = re.compile(r"(?<![0-9])(?:\+[0-9]{8,15}|\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4})(?![0-9])")
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


def _card(match: re.Match) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if 13 <= len(digits) <= 19 and (len(digits) >= 15 or _luhn(digits)) and not _clock_or_date(raw):
        return "[REDACTED:card]"
    return raw


def _address(match: re.Match) -> str:
    if _BARE_YEAR.fullmatch(match.group(0)):
        return match.group(0)   # "1999 St." is a year and a title, not a street
    return "[REDACTED:address]"


def redact_text(text: str) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[REDACTED:email]", text)
    text = _SSN.sub("[REDACTED:ssn]", text)
    text = _CARD.sub(_card, text)
    text = _PHONE.sub("[REDACTED:phone]", text)
    text = _ADDRESS.sub(_address, text)
    return text
