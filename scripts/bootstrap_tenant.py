"""Creates a tenant (account) and a first agent on a running API and prints the keys as JSON.
Usage: VOX_ADMIN_KEY=... python scripts/bootstrap_tenant.py http://127.0.0.1:8080 [--name demo] [--greeting "..."]
The tenant API key is shown only here, once: the server stores just its hash."""
import argparse
import json
import os
import sys
import urllib.request

DEFAULT_GREETING = "Hi, this is Arvindh's voice agent. How can I help you today?"
PROMPT = ("You are a friendly receptionist for a dental clinic. Keep answers to one or two short sentences. "
          "You can check availability and book a slot. Always read the day and time back before booking.")


def call(base: str, path: str, body: dict, headers: dict):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def bootstrap(base: str, admin_key: str, name: str = "demo", greeting: str = DEFAULT_GREETING, concurrency: int = 64) -> dict:
    t = call(base, "/admin/tenants", {"name": name, "concurrencyLimit": concurrency}, {"X-Admin-Key": admin_key})
    a = call(base, "/v1/agents", {"name": "receptionist", "voice": "af_heart", "greeting": greeting, "systemPrompt": PROMPT,
                                  "tools": ["check_availability", {"tool": "book_slot", "confirm": True}, "hangUp"]},
             {"X-API-Key": t["apiKey"]})
    return {"tenantId": t["tenantId"], "apiKey": t["apiKey"], "agentId": a["id"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--greeting", default=DEFAULT_GREETING)
    ns = ap.parse_args()
    key = os.environ.get("VOX_ADMIN_KEY", "")
    if not key:
        sys.exit("set VOX_ADMIN_KEY")
    print(json.dumps(bootstrap(ns.base_url, key, ns.name, ns.greeting)))
