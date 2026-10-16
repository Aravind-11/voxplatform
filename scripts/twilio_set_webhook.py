"""Points a Twilio phone number's voice webhook at an agent, using the Twilio REST API (no console clicking).
Usage: TWILIO_ACCOUNT_SID=AC... TWILIO_AUTH_TOKEN=... python scripts/twilio_set_webhook.py +16056278460 https://host/twilio/voice/agt_..."""
import os
import sys

import httpx

BASE = "https://api.twilio.com/2010-04-01/Accounts"


def set_voice_url(sid: str, token: str, number: str, url: str, client: httpx.Client | None = None) -> str:
    """Returns the phone number's SID. Raises RuntimeError with Twilio's message on failure."""
    c = client or httpx.Client(timeout=15)
    r = c.get(f"{BASE}/{sid}/IncomingPhoneNumbers.json", params={"PhoneNumber": number}, auth=(sid, token))
    if r.status_code >= 300:
        raise RuntimeError(f"Twilio lookup failed (HTTP {r.status_code}): {r.text[:200]}")
    nums = r.json().get("incoming_phone_numbers", [])
    if not nums:
        raise RuntimeError(f"{number} is not a number on this Twilio account")
    pn = nums[0]["sid"]
    r = c.post(f"{BASE}/{sid}/IncomingPhoneNumbers/{pn}.json", auth=(sid, token),
               data={"VoiceUrl": url, "VoiceMethod": "POST"})
    if r.status_code >= 300:
        raise RuntimeError(f"Twilio update failed (HTTP {r.status_code}): {r.text[:200]}")
    return pn


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sid, token = os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        sys.exit("set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
    print("updated", set_voice_url(sid, token, sys.argv[1], sys.argv[2]))
