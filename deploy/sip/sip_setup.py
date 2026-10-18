#!/usr/bin/env python3
"""POST /internal/sip/setup before AudioSocket connects.

The dialplan passes only digits and an optional leading '+'. The shared token stays in
SIP_SETUP_TOKEN so a secret that contains quotes or '$' is never placed in a shell command.
'+' is form-encoded as %2B (a raw form plus would be read as a space).
"""
from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

_SAFE_NUM = re.compile(r"^[0-9+]*$")
_SAFE_UUID = re.compile(r"^[0-9a-fA-F-]+$")
DEFAULT_URL = "http://api1:8080/internal/sip/setup"


def post_setup(caller: str, exten: str, uuid: str, token: str, url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """Return True only when the API accepts the setup. Unsafe caller, extension, or uuid is not sent."""
    if not _SAFE_NUM.fullmatch(caller or "") or not _SAFE_NUM.fullmatch(exten or ""):
        return False
    if not uuid or not _SAFE_UUID.fullmatch(uuid):
        return False
    body = urllib.parse.urlencode({"uuid": uuid, "caller": caller, "exten": exten}).encode()
    try:
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"X-Vox-Sip-Token": token or ""})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    return True


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        return 2
    caller, exten, uid = argv[0], argv[1], argv[2]
    url = argv[3] if len(argv) == 4 else DEFAULT_URL
    # A missed setup falls back to VOX_AUDIOSOCKET_AGENT. Do not fail the dialplan on a network error.
    post_setup(caller, exten, uid, os.environ.get("SIP_SETUP_TOKEN", ""), url)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
