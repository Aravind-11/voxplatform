"""Drives /dashboard and /demo in headless Chromium with a fake microphone against the fake-model server.
Run: python tests/e2e/browser_test.py     (needs playwright + chromium; no GPU)"""
import os
import subprocess
import sys
import time
import wave

import httpx
import numpy as np
from playwright.sync_api import sync_playwright

PORT = 18080
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("E2E_OUT", "/tmp/e2e")


def make_wav(path):
    sr = 16000
    t = np.arange(int(sr * 1.6)) / sr
    speech = (9000 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.int16)
    x = np.concatenate([np.zeros(sr, np.int16), speech, np.zeros(int(sr * 4.5), np.int16)])
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, "mic.wav")
    make_wav(wav)
    srv = subprocess.Popen([sys.executable, os.path.join(HERE, "serve_fake.py"), str(PORT)])
    try:
        for _ in range(50):
            try:
                httpx.get(BASE + "/healthz")
                break
            except Exception:
                time.sleep(0.2)
        key = httpx.post(BASE + "/admin/tenants", json={"name": "e2e"}, headers={"X-Admin-Key": "adm"}).json()["apiKey"]
        errors = []
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                        f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
            page = b.new_context(viewport={"width": 1100, "height": 800}).new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

            # --- dashboard: create an agent through the UI
            page.goto(BASE + "/dashboard")
            page.fill("#key", key)
            page.click("#login")
            page.click("nav button:has-text('Agents')")
            page.fill("input[placeholder=name]", "ui agent")
            page.fill("textarea[placeholder='system prompt']", "You are brief.")
            page.fill("input[placeholder^='greeting']", "Hello from the dashboard agent.")
            page.check("input[type=checkbox]")
            page.click("button:has-text('Create')")
            page.wait_for_selector("text=ui agent")
            page.screenshot(path=os.path.join(OUT, "dashboard-agents.png"))

            # --- demo: talk to it with the fake microphone
            demo = page.context.new_page()
            demo.on("pageerror", lambda e: errors.append(str(e)))
            demo.goto(BASE + "/demo")
            demo.fill("#key", key)
            demo.dispatch_event("#key", "change")
            demo.wait_for_function("document.querySelectorAll('#agent option').length > 1")
            demo.select_option("#agent", index=1)
            demo.click("#go")
            demo.wait_for_selector(".msg.u", timeout=20000)             # the fake mic speech was heard and transcribed
            demo.wait_for_selector(".msg.a", timeout=20000)             # and the agent answered
            texts = demo.eval_on_selector_all(".msg", "els => els.map(e => e.className + ': ' + e.textContent)")
            demo.screenshot(path=os.path.join(OUT, "demo.png"))
            assert any("Hello from the dashboard agent" in t for t in texts), texts      # greeting reached the page
            assert any(t.startswith("msg u") and "hello agent" in t for t in texts), texts
            assert any(t.startswith("msg a") and "turn 1" in t for t in texts), texts
            demo.click("#go")                                                # End call
            demo.wait_for_function("document.getElementById('go').textContent === 'Start call'")

            # --- dashboard: the call, transcript and recording are visible
            page.click("nav button:has-text('Calls')")
            page.click("table a")
            page.wait_for_selector(".chat b:has-text('agent')")
            page.wait_for_selector("audio", timeout=10000)
            page.screenshot(path=os.path.join(OUT, "dashboard-call.png"))
            for tab in ("Usage", "Keys", "Audit", "Knowledge", "Tools"):
                page.click(f"nav button:has-text('{tab}')")
                page.wait_for_timeout(300)
            page.click("nav button:has-text('Audit')")
            page.wait_for_selector("text=agent.create")
            b.close()
        assert not errors, errors
        print("browser e2e OK; screenshots in", OUT)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
