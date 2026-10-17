"""Telephony call control. TwilioCallControl is the Calls.json client the transfer tool and the outbound
route already used. AsteriskCallControl talks to ARI when ASTERISK_ARI_URL is set. No real Asterisk here:
callers pass an HTTP client, and tests pass a fake that records the requests."""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol
from xml.sax.saxutils import escape, quoteattr


class CallControl(Protocol):
    async def transfer(self, call_sid: str, to: str, mode: str = "cold", *, call_id: str = "", target: str = "") -> None:
        """Move an answered call. `to` is the E.164 number; `target` is the configured name (warm Twilio query)."""
        ...

    async def originate(self, to: str, from_: str, twiml_or_app: Any = None) -> Any:
        """Place a new call. Twilio: `twiml_or_app` is the Calls.json form. Asterisk: it is ARI appArgs."""
        ...

    async def hangup(self, call_sid: str) -> None:
        ...


class TwilioCallControl:
    """Same TwiML and REST paths as the in-tree Twilio transfer and outbound calls."""

    def __init__(self, client: Any, sid: str, token: str, base_url: str = ""):
        self.client, self.sid, self.token, self.base_url = client, sid, token, base_url

    async def transfer(self, call_sid: str, to: str, mode: str = "cold", *, call_id: str = "", target: str = "") -> None:
        if mode == "warm":
            if not self.base_url or not call_id:
                raise RuntimeError("warm transfer needs the public URL configured")
            # Warm: the person who picks up first hears a short summary (<Number url=...>), and if nobody
            # answers the caller is returned to the agent (the <Dial action=...> callback).
            name = target or to
            twiml = (f'<Response><Dial timeout="25" action={quoteattr(f"{self.base_url}/twilio/transfer-result/{call_id}?to={name}")}>'
                     f'<Number url={quoteattr(f"{self.base_url}/twilio/whisper/{call_id}")}>{escape(to)}</Number></Dial></Response>')
        else:
            twiml = f"<Response><Dial>{escape(to)}</Dial></Response>"
        r = await self.client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Calls/{call_sid}.json",
            auth=(self.sid, self.token), timeout=10, data={"Twiml": twiml})
        if r.status_code >= 300:
            raise RuntimeError(f"twilio transfer failed: HTTP {r.status_code}")

    async def originate(self, to: str, from_: str, twiml_or_app: Any = None) -> Any:
        if isinstance(twiml_or_app, dict):
            data = dict(twiml_or_app)
            data.setdefault("To", to)
            data.setdefault("From", from_)
        else:
            data = {"To": to, "From": from_}
            if isinstance(twiml_or_app, str) and twiml_or_app:
                data["Twiml"] = twiml_or_app
        return await self.client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Calls.json",
            auth=(self.sid, self.token), data=data, timeout=15)

    async def hangup(self, call_sid: str) -> None:
        r = await self.client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Calls/{call_sid}.json",
            auth=(self.sid, self.token), timeout=10, data={"Status": "completed"})
        if r.status_code >= 300:
            raise RuntimeError(f"twilio hangup failed: HTTP {r.status_code}")


class AsteriskCallControl:
    """ARI. `originate` enters Stasis app `vox` (this process does not run that app). `appArgs` of
    `amd,<call id>` tells it to POST AnsweredBy to /sip/amd/<call id>. `transfer` is a blind redirect;
    `mode` is accepted so the tool can pass warm/cold and is not sent to ARI."""

    def __init__(self, client: Any, ari_url: str, user: str = "", password: str = ""):
        self.client = client
        self.ari_url = ari_url.rstrip("/")
        self.user, self.password = user, password

    def _auth(self) -> tuple:
        return (self.user, self.password)

    async def originate(self, to: str, from_: str, twiml_or_app: Any = None) -> Any:
        params = {"endpoint": f"PJSIP/{to}@voxsun", "app": "vox", "callerId": from_ or ""}
        if isinstance(twiml_or_app, str) and twiml_or_app:
            params["appArgs"] = twiml_or_app
        return await self.client.post(f"{self.ari_url}/ari/channels", params=params, auth=self._auth(), timeout=15)

    async def transfer(self, call_sid: str, to: str, mode: str = "cold", *, call_id: str = "", target: str = "") -> None:
        del mode, call_id, target
        r = await self.client.post(
            f"{self.ari_url}/ari/channels/{call_sid}/redirect",
            params={"endpoint": f"PJSIP/{to}"}, auth=self._auth(), timeout=10)
        if r.status_code >= 300:
            raise RuntimeError(f"asterisk transfer failed: HTTP {r.status_code}")

    async def hangup(self, call_sid: str) -> None:
        r = await self.client.delete(f"{self.ari_url}/ari/channels/{call_sid}", auth=self._auth(), timeout=10)
        if r.status_code >= 300:
            raise RuntimeError(f"asterisk hangup failed: HTTP {r.status_code}")


def control_from_env(client: Any, public_url: str = "") -> Optional[CallControl]:
    """ARI wins when ASTERISK_ARI_URL is set; otherwise Twilio when both account sid and token are set."""
    ari = os.environ.get("ASTERISK_ARI_URL", "").strip()
    if ari:
        return AsteriskCallControl(client, ari, os.environ.get("ASTERISK_ARI_USER", ""),
                                   os.environ.get("ASTERISK_ARI_PASSWORD", ""))
    sid, token = os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")
    if sid and token:
        return TwilioCallControl(client, sid, token, public_url)
    return None
