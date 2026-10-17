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
_ADDRESS = re.compile(
    r"\b[0-9]{1,6}\s+(?:[A-Za-z][A-Za-z'.-]*\s+){0,5}"
    r"(?:Boulevard|Street|Avenue|Road|Drive|Lane|Blvd|Ave|St|Rd|Dr|Ln)\b\.?",
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


def _card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and (len(digits) >= 15 or _luhn(digits)):
        return "[REDACTED:card]"
    return match.group(0)


def redact_text(text: str) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[REDACTED:email]", text)
    text = _SSN.sub("[REDACTED:ssn]", text)
    text = _CARD.sub(_card, text)
    text = _PHONE.sub("[REDACTED:phone]", text)
    text = _ADDRESS.sub("[REDACTED:address]", text)
    return text
