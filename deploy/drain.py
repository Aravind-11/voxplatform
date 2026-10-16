"""Pod preStop hook: stop taking new calls, then wait for live ones to finish (Kubernetes kills the pod after
terminationGracePeriodSeconds regardless). Env: VOX_ADMIN_KEY. Usage: python drain.py [base_url] [max_seconds]"""
import json
import os
import sys
import time
import urllib.error
import urllib.request


def active_calls(base: str, urlopen=urllib.request.urlopen) -> int:
    try:
        r = urlopen(base + "/healthz", timeout=3)
    except urllib.error.HTTPError as e:            # 503 while draining carries the same JSON body
        r = e
    return int(json.load(r).get("active_calls", 0))


def drain(base: str, key: str, max_s: float = 3500, urlopen=urllib.request.urlopen, sleep=time.sleep, now=time.time) -> int:
    """Returns the number of calls still live when it stopped waiting (0 = clean)."""
    urlopen(urllib.request.Request(base + "/admin/drain", method="POST", headers={"X-Admin-Key": key}), timeout=5)
    deadline = now() + max_s
    while True:
        n = active_calls(base, urlopen)
        if n == 0 or now() >= deadline:
            return n
        sleep(5)


if __name__ == "__main__":
    left = drain(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080", os.environ["VOX_ADMIN_KEY"],
                 float(sys.argv[2]) if len(sys.argv) > 2 else 3500)
    sys.exit(1 if left else 0)
