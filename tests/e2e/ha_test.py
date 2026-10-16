"""High-availability test: the production topology run as real processes, with fake models (no GPU needed).

    Caddy (load balancer, health checks)  ->  2 API nodes  ->  PostgreSQL + Redis
    Prometheus -> Alertmanager -> a webhook receiver that records what it is sent

What it proves (and what it does not):
  * calls spread over both nodes; a killed node (SIGKILL, mid-call) does not stop new calls: Caddy retries elsewhere
  * a phone call whose node dies is reconnected on the surviving node, with its transcript (the /twilio/resume path
    that Twilio follows after <Connect>; Twilio itself is simulated here, so the real Twilio behaviour is unverified)
  * the dead node's slots in the Redis concurrency counter expire by themselves
  * a drained node receives no new calls
  * Prometheus detects the dead node, Alertmanager delivers the alert to the webhook, and it resolves after restart
It does NOT run the Docker images (registries are blocked in this build environment) or any real model.

Usage:
    BIN=/path/with/caddy,prometheus,alertmanager,redis-server  VOX_DB=postgresql://user@host/db  python tests/e2e/ha_test.py
"""
import asyncio
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import httpx
import numpy as np
import redis
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.audio import pcm16_to_mulaw  # noqa: E402

BIN = os.environ.get("BIN", "")
DB = os.environ["VOX_DB"]
LB, A, B = 8180, 8101, 8102
REDIS_PORT, PROM, AM, HOOK = 6390, 9190, 9193, 9199
ADMIN, METRICS_TOKEN = "ha-admin", "ha-metrics"
TTL = 6
W = tempfile.mkdtemp(prefix="ha_")
procs: dict = {}
results: list = []


def exe(name):
    for d in [BIN] + os.environ.get("PATH", "").split(":"):
        p = os.path.join(d, name) if d else ""
        if p and os.path.isfile(p):
            return p
    sys.exit(f"{name} not found (set BIN=dir)")


def start(name, argv, env=None, cwd=None):
    log = open(os.path.join(W, f"{name}.log"), "ab")
    procs[name] = subprocess.Popen(argv, stdout=log, stderr=log, env={**os.environ, **(env or {})}, cwd=cwd)
    return procs[name]


def kill(name, sig=signal.SIGKILL):
    p = procs.get(name)
    if p and p.poll() is None:
        p.send_signal(sig)
        p.wait(10)


def wait_http(url, timeout=30, ok=(200,), headers=None):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = httpx.get(url, timeout=2, headers=headers)
            if r.status_code in ok:
                return r
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise RuntimeError(f"not ready: {url}")


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""), flush=True)


def node_env(port):
    return {"VOX_DB": DB, "VOX_REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}/0", "VOX_REDIS_TTL": str(TTL),
            "VOX_ADMIN_KEY": ADMIN, "VOX_METRICS_TOKEN": METRICS_TOKEN, "VOX_PUBLIC_URL": f"http://127.0.0.1:{LB}",
            "VOX_ALLOW_UNSIGNED_TWILIO": "1", "PYTHONPATH": ROOT}


def start_node(name, port):
    start(name, [sys.executable, os.path.join(ROOT, "tests/e2e/serve_fake.py"), str(port)], env=node_env(port))
    wait_http(f"http://127.0.0.1:{port}/healthz")


def metric(port, name):
    """Sum of a metric's samples from one node's /metrics (None if the node is down)."""
    try:
        t = httpx.get(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"}, timeout=2).text
    except Exception:  # noqa: BLE001
        return None
    return sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))


def active(port):
    try:
        return httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2).json()["active_calls"]
    except Exception:  # noqa: BLE001
        return None


def frames(kind, n):
    return [(np.full(512, 5000 if kind == "speech" else 0, dtype="<i2")).tobytes()] * n


class Browser:
    """A browser-style call through the load balancer."""

    def __init__(self, key, agent=None):
        self.key, self.agent, self.audio, self.closed = key, agent, 0, False

    async def open(self):
        r = httpx.post(f"http://127.0.0.1:{LB}/v1/calls", headers={"X-API-Key": self.key}, json={"agentId": self.agent} if self.agent else {})
        r.raise_for_status()
        self.call = r.json()
        self.ws = await websockets.connect(self.call["joinUrl"], max_size=None)
        self.task = asyncio.create_task(self._rx())

    async def _rx(self):
        try:
            async for m in self.ws:
                if isinstance(m, bytes):
                    self.audio += len(m)
        except Exception:  # noqa: BLE001
            pass
        self.closed = True

    async def talk(self):
        for f in frames("silence", 5) + frames("speech", 15) + frames("silence", 40):
            await self.ws.send(f)
            await asyncio.sleep(0.005)

    async def close(self):
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass


class Phone:
    """A Twilio-style call: webhook -> TwiML -> media stream. Twilio itself is simulated."""

    async def start(self, agent_id=None, twiml_url=None, form=None):
        if twiml_url is None:
            r = httpx.post(f"http://127.0.0.1:{LB}/twilio/voice/{agent_id}", data={"CallSid": "CA1", "From": "+15550001", "To": "+15550002"})
        else:
            r = httpx.post(twiml_url, data=form or {"CallSid": "CA1"})
        self.twiml = r.text
        assert "<Stream" in self.twiml, self.twiml
        self.ws_url = self.twiml.split('url="')[1].split('"')[0]
        self.token = self.twiml.split('name="token" value="')[1].split('"')[0]
        self.resume_url = self.twiml.split('method="POST">')[1].split("</Redirect>")[0]
        self.call_id = self.ws_url.rsplit("/", 1)[1]
        self.ws = await websockets.connect(self.ws_url, max_size=None)
        self.media = 0
        self.task = asyncio.create_task(self._rx())
        await self.ws.send(json.dumps({"event": "connected"}))
        await self.ws.send(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": self.token}}}))

    async def _rx(self):
        try:
            async for m in self.ws:
                if json.loads(m).get("event") == "media":
                    self.media += 1
        except Exception:  # noqa: BLE001
            pass

    async def talk(self):
        loud, quiet = np.full(160, 5000, dtype=np.int16), np.zeros(160, dtype=np.int16)
        for p in [quiet] * 10 + [loud] * 60 + [quiet] * 200:
            await self.ws.send(json.dumps({"event": "media", "streamSid": "MZ1", "media": {
                "track": "inbound", "payload": base64.b64encode(pcm16_to_mulaw(p)).decode()}}))
            await asyncio.sleep(0.002)


async def until(fn, timeout=15, step=0.2):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        await asyncio.sleep(step)
    return None


def prom(q):
    r = httpx.get(f"http://127.0.0.1:{PROM}/api/v1/query", params={"query": q}, timeout=5).json()["data"]["result"]
    return r


def firing(alertname):
    try:
        alerts = httpx.get(f"http://127.0.0.1:{PROM}/api/v1/alerts", timeout=5).json()["data"]["alerts"]
    except Exception:  # noqa: BLE001
        return []
    return [a for a in alerts if a["labels"].get("alertname") == alertname and a["state"] == "firing"]


def hook_events():
    path = os.path.join(W, "hook.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def setup_monitoring():
    mon = os.path.join(ROOT, "deploy/monitoring")
    tg = os.path.join(W, "targets"); os.makedirs(tg)
    open(os.path.join(tg, "voxplatform.yml"), "w").write(f'- targets: ["127.0.0.1:{A}", "127.0.0.1:{B}"]\n')
    open(os.path.join(W, "metrics_token"), "w").write(METRICS_TOKEN)
    open(os.path.join(W, "hook_url"), "w").write(f"http://127.0.0.1:{HOOK}/alerts")
    cfg = open(os.path.join(mon, "prometheus.yml")).read()
    cfg = (cfg.replace("scrape_interval: 10s", "scrape_interval: 2s").replace("evaluation_interval: 10s", "evaluation_interval: 2s")
           .replace("/etc/prometheus/alerts.yml", os.path.join(W, "alerts.yml"))
           .replace("alertmanager:9093", f"127.0.0.1:{AM}")
           .replace("/run/secrets/vox_metrics_token", os.path.join(W, "metrics_token"))
           .replace("/etc/prometheus/targets/voxplatform.yml", os.path.join(tg, "voxplatform.yml"))
           .replace("/etc/prometheus/targets/vllm.yml", os.path.join(tg, "vllm.yml")))
    open(os.path.join(tg, "vllm.yml"), "w").write("[]\n")
    open(os.path.join(W, "prometheus.yml"), "w").write(cfg)
    rules = open(os.path.join(mon, "alerts.yml")).read().replace("for: 30s", "for: 6s")        # same rules, faster for the test
    open(os.path.join(W, "alerts.yml"), "w").write(rules)
    am = open(os.path.join(mon, "alertmanager.yml")).read()
    am = (am.replace("/run/secrets/alert_webhook_url", os.path.join(W, "hook_url"))
          .replace("group_wait: 10s", "group_wait: 1s").replace("group_interval: 2m", "group_interval: 3s"))
    open(os.path.join(W, "alertmanager.yml"), "w").write(am)


HOOK_SRC = r'''
import http.server, sys, json
out = sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        open(out, "ab").write(body.replace(b"\n", b" ") + b"\n")
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


async def main():
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    # --- infrastructure
    start("redis", [exe("redis-server"), "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no"])
    time.sleep(1)
    r = redis.Redis(port=REDIS_PORT, decode_responses=True)
    r.ping()
    start_node("nodeA", A)
    start_node("nodeB", B)
    caddyfile = "{\n\tadmin off\n}\n" + open(os.path.join(ROOT, "deploy/Caddyfile")).read()
    open(os.path.join(W, "Caddyfile"), "w").write(caddyfile)
    start("caddy", [exe("caddy"), "run", "--config", os.path.join(W, "Caddyfile"), "--adapter", "caddyfile"],
          env={"VOX_DOMAIN": f":{LB}", "VOX_UPSTREAMS": f"127.0.0.1:{A} 127.0.0.1:{B}", "HOME": W, "XDG_DATA_HOME": W})
    wait_http(f"http://127.0.0.1:{LB}/healthz")
    setup_monitoring()
    open(os.path.join(W, "hook.py"), "w").write(HOOK_SRC)
    start("hook", [sys.executable, os.path.join(W, "hook.py"), str(HOOK), os.path.join(W, "hook.jsonl")])
    start("alertmanager", [exe("alertmanager"), f"--config.file={W}/alertmanager.yml", f"--web.listen-address=127.0.0.1:{AM}",
                           "--cluster.listen-address=", f"--storage.path={W}/am"])
    start("prometheus", [exe("prometheus"), f"--config.file={W}/prometheus.yml", f"--web.listen-address=127.0.0.1:{PROM}",
                         f"--storage.tsdb.path={W}/prom"])
    wait_http(f"http://127.0.0.1:{PROM}/-/ready", 40)
    wait_http(f"http://127.0.0.1:{AM}/-/ready", 20)

    # --- tenant + agent through the load balancer
    t = httpx.post(f"http://127.0.0.1:{LB}/admin/tenants", headers={"X-Admin-Key": ADMIN}, json={"name": "ha", "concurrencyLimit": 50}).json()
    key, tid = t["apiKey"], t["tenantId"]
    agent = httpx.post(f"http://127.0.0.1:{LB}/v1/agents", headers={"X-API-Key": key},
                       json={"name": "ha", "greeting": "Hello, this is the agent."}).json()["id"]
    check("admin and agent creation work through the load balancer", key and agent)

    # --- 1. calls spread over both nodes and get audio
    calls = [Browser(key) for _ in range(8)]
    for c in calls:
        await c.open()
    for c in calls:
        await c.talk()
    got = await until(lambda: all(c.audio > 0 for c in calls), 20)
    check("8 calls through the proxy all received agent audio", got, f"audio bytes {[c.audio for c in calls]}")
    a_n, b_n = active(A), active(B)
    check("calls are spread over both nodes", a_n and b_n and a_n + b_n == 8, f"node A {a_n}, node B {b_n}")
    check("Redis counts every active call across nodes", r.zcard(f"vox:active:{tid}") == 8, f"zcard {r.zcard(f'vox:active:{tid}')}")

    # --- 2. a phone call on some node, then that node dies
    phone = Phone()
    await phone.start(agent_id=agent)
    await phone.talk()
    ok = await until(lambda: phone.media > 0, 20)
    check("phone call through the proxy gets agent audio", ok, f"{phone.media} media frames")
    serving = A if (active(A) or 0) > (a_n or 0) else B
    survivor = B if serving == A else A
    sname, vname = ("nodeA", "nodeB") if serving == A else ("nodeB", "nodeA")
    survivors_before = sum(1 for c in calls if not c.closed)
    kill(sname)                                                    # SIGKILL: no clean shutdown, like a crashed host
    await until(lambda: phone.task.done(), 10)
    check("the phone call's stream drops when its node is killed", phone.task.done())
    dropped = sum(1 for c in calls if c.closed)
    check("only calls on the dead node dropped; the other node's calls stay up", 0 < dropped < 8 and dropped == (a_n if serving == A else b_n),
          f"dropped {dropped}")

    # --- 3. new calls keep working (the proxy retries / stops using the dead node)
    fresh = [Browser(key) for _ in range(4)]
    for c in fresh:
        await c.open()
        await c.talk()
    ok = await until(lambda: all(c.audio > 0 for c in fresh), 20)
    check("new calls succeed while a node is dead", ok, f"audio {[c.audio for c in fresh]}")

    # --- 4. the phone call is reconnected on the surviving node with its history
    store_msgs = httpx.get(f"http://127.0.0.1:{LB}/v1/calls/{phone.call_id}/messages", headers={"X-API-Key": key}).json()
    resumed = Phone()
    await resumed.start(twiml_url=phone.resume_url)           # what Twilio does when the stream ends
    await resumed.talk()
    ok = await until(lambda: resumed.media > 0, 20)
    check("Twilio's follow-up request reconnects the call and the agent speaks again", ok, f"{resumed.media} media frames")
    info = httpx.get(f"http://127.0.0.1:{LB}/v1/calls/{resumed.call_id}", headers={"X-API-Key": key}).json()
    check("the new call is linked to the lost one", info["metadata"].get("resumedFrom") == phone.call_id, str(info["metadata"]))
    old = httpx.get(f"http://127.0.0.1:{LB}/v1/calls/{phone.call_id}", headers={"X-API-Key": key}).json()
    check("the lost call is closed with reason stream_lost", old["status"] == "ended" and old["end_reason"] == "stream_lost", f"{old['status']} {old['end_reason']}")
    check("a transcript existed to carry over", any(m["role"] == "agent" for m in store_msgs), f"{len(store_msgs)} messages")
    check("resume counter increased", (metric(survivor, "vox_call_resumes_total") or 0) >= 1)

    # --- 5. Redis slots of the dead node expire by themselves
    for c in calls + fresh:
        await c.close()
    await resumed.ws.close()
    await asyncio.sleep(TTL + 3)
    left = r.zcount(f"vox:active:{tid}", time.time(), "+inf")        # what the limiter itself counts (expired members ignored)
    check("the dead node's call slots expired from Redis (no leaked capacity)", left == 0, f"{left} live slots, {r.zcard(f'vox:active:{tid}')} stale members awaiting cleanup")

    # --- 6. alerting
    nd = await until(lambda: firing("NodeDown"), 30)
    check("Prometheus fires NodeDown for the killed node", nd, str([a["labels"].get("instance") for a in (nd or [])]))
    got_hook = await until(lambda: any("NodeDown" in json.dumps(e) and e.get("status") == "firing" for e in hook_events()), 30)
    check("Alertmanager delivered NodeDown to the webhook", got_hook, f"{len(hook_events())} webhook posts")
    start_node(sname, serving)                                      # bring the node back
    resolved = await until(lambda: any("NodeDown" in json.dumps(e) and e.get("status") == "resolved" for e in hook_events()), 40)
    check("the alert resolves after the node returns", resolved)
    check("Prometheus has call metrics from both nodes", len(prom("vox_calls_total")) >= 1 and len(prom("up{job='voxplatform'} == 1")) == 2)

    # --- 7. draining: a drained node gets no new calls
    base_b = metric(B, "vox_calls_total") or 0
    httpx.post(f"http://127.0.0.1:{B}/admin/drain", headers={"X-Admin-Key": ADMIN})
    await asyncio.sleep(7)                                           # > health_interval x 2
    drained = [Browser(key) for _ in range(4)]
    for c in drained:
        await c.open()
        await c.talk()
    await until(lambda: all(c.audio > 0 for c in drained), 20)
    check("calls keep working while a node drains", all(c.audio > 0 for c in drained))
    check("a drained node receives no new calls", (active(B) or 0) == 0, f"node B active {active(B)}")
    for c in drained:
        await c.close()
    httpx.post(f"http://127.0.0.1:{B}/admin/undrain", headers={"X-Admin-Key": ADMIN})

    # --- 8. losing Redis or the database turns readiness red (the proxy then avoids the node)
    kill("redis")
    await asyncio.sleep(1)
    rr = httpx.get(f"http://127.0.0.1:{A}/healthz")
    check("a node reports not-ready when Redis is unreachable", rr.status_code == 503 and rr.json()["checks"].get("redis") is False, rr.text)


try:
    asyncio.run(main())
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("test run completed without an exception", False, repr(e))
finally:
    for n in list(procs):
        kill(n, signal.SIGTERM)
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Logs: {W}")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        for n in ("nodeA", "nodeB", "caddy"):
            p = os.path.join(W, f"{n}.log")
            if os.path.exists(p):
                print(f"--- tail {n}.log"); print(open(p, errors="replace").read()[-1500:])
    sys.exit(1 if failed else 0)
