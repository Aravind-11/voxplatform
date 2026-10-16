"""Call variables: `{{name}}` placeholders in an agent's system prompt and greeting, filled per call.

A CRM-driven call needs the customer's name, the deal being discussed, and so on. The caller of the API passes
`variables` when it creates a call (or an outbound call); the agent can carry defaults; and the platform always
supplies `caller`, `callee` and `callId`. Precedence: platform < agent defaults < call variables, except that
the platform values cannot be overridden by a request (a caller cannot spoof their own number into the prompt).
"""
from __future__ import annotations

import re
from typing import Iterable

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]{0,63})\s*\}\}")
BUILTIN = ("caller", "callee", "callId")
MAX_VARS, MAX_VALUE = 30, 500


class VariableError(ValueError):
    pass


def names_in(*texts: str) -> set[str]:
    return {m.group(1) for t in texts if t for m in PLACEHOLDER.finditer(t)}


def validate(variables) -> dict[str, str]:
    """Normalizes request variables to {str: str}. Raises VariableError."""
    if variables is None:
        return {}
    if not isinstance(variables, dict):
        raise VariableError("variables must be an object of name -> text")
    if len(variables) > MAX_VARS:
        raise VariableError(f"at most {MAX_VARS} variables")
    out = {}
    for k, v in variables.items():
        if not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", k):
            raise VariableError(f"bad variable name {k!r}")
        if k in BUILTIN:
            raise VariableError(f"{k!r} is set by the platform and cannot be overridden")
        if isinstance(v, (dict, list)) or v is None:
            raise VariableError(f"variable {k!r} must be text or a number")
        s = str(v)
        if len(s) > MAX_VALUE:
            raise VariableError(f"variable {k!r} is longer than {MAX_VALUE} characters")
        out[k] = s
    return out


def missing(texts: Iterable[str], available: Iterable[str]) -> list[str]:
    return sorted(names_in(*texts) - set(available) - set(BUILTIN))


def render(text: str, values: dict[str, str]) -> str:
    """Unknown placeholders become empty text, so a live call never speaks braces."""
    return PLACEHOLDER.sub(lambda m: values.get(m.group(1), ""), text or "")
