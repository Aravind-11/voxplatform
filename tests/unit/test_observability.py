import json
import os
import subprocess

import pytest

from core.metrics import METRICS
from tests.unit.test_api import make, tenant

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_healthz_goes_red_when_the_database_is_unreachable(monkeypatch):
    c, store, _ = make()
    assert c.get("/healthz").json()["checks"] == {"db": True}
    monkeypatch.setattr(store, "ping", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    r = c.get("/healthz")
    assert r.status_code == 503 and r.json()["ok"] is False and r.json()["checks"]["db"] is False
    assert 'vox_errors_total{where="healthz_db"}' in METRICS.render()


def test_metrics_expose_local_and_cluster_gauges():
    make()
    text = METRICS.render()
    assert "vox_active_calls " in text and "vox_active_calls_cluster " in text


def test_monitoring_files_are_consistent_with_the_metrics_the_app_exports():
    """Every metric an alert or the dashboard uses must exist in the app (or be vLLM's own)."""
    import re
    import yaml
    from api import app as appmod  # noqa: F401  (imports register the metric descriptions)
    exported = set(re.findall(r"vox_[a-z_]+", open(os.path.join(ROOT, "core", "metrics.py")).read()))
    for mod in ("api/app.py", "api/runtime.py"):
        exported |= set(re.findall(r"vox_[a-z_]+", open(os.path.join(ROOT, mod)).read()))
    used = set()
    for f in ("deploy/monitoring/alerts.yml", "deploy/monitoring/grafana/dashboards/voxplatform.json"):
        used |= set(re.findall(r"vox_[a-z_]+", open(os.path.join(ROOT, f)).read()))
    used = {re.sub(r"_(bucket|sum|count)$", "", m) for m in used}
    assert used <= exported, f"alerts/dashboard use metrics the app never exports: {sorted(used - exported)}"
    rules = yaml.safe_load(open(os.path.join(ROOT, "deploy/monitoring/alerts.yml")))
    names = [r["alert"] for g in rules["groups"] for r in g["rules"]]
    assert len(names) == len(set(names)) >= 8 and all(r["labels"]["severity"] in ("warning", "critical")
                                                       for g in rules["groups"] for r in g["rules"])
    by_name = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}
    near, at = by_name["NodeNearCapacity"], by_name["NodeAtCap"]
    assert near["expr"] == 'vox_active_calls{job="voxplatform"} > 24' and near["for"] == "5m"
    assert near["labels"]["severity"] == "warning"
    assert at["expr"] == 'vox_active_calls{job="voxplatform"} >= 32' and at["for"] == "1m"
    assert at["labels"]["severity"] == "critical"
    from api.app import Settings
    assert Settings().node_max_calls == 32
    main_src = open(os.path.join(ROOT, "api", "main.py")).read()
    assert 'env.get("VOX_NODE_MAX_CALLS", "32")' in main_src
    dash = json.load(open(os.path.join(ROOT, "deploy/monitoring/grafana/dashboards/voxplatform.json")))
    assert len(dash["panels"]) >= 10 and len({p["id"] for p in dash["panels"]}) == len(dash["panels"])


def test_compose_file_is_valid_when_docker_is_available(tmp_path):
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker not installed")
    env = tmp_path / "e.env"
    env.write_text(open(os.path.join(ROOT, "deploy/env.example")).read())
    sec = os.path.join(ROOT, "deploy", "secrets")
    made = not os.path.isdir(sec)
    os.makedirs(sec, exist_ok=True)
    try:
        for f in ("vox_metrics_token", "alert_webhook_url", "grafana_admin_password"):
            open(os.path.join(sec, f), "a").close()
        r = subprocess.run(["docker", "compose", "-f", os.path.join(ROOT, "deploy/docker-compose.yml"), "--env-file", str(env),
                            "config", "-q"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
    finally:
        if made:
            import shutil as sh
            sh.rmtree(sec, ignore_errors=True)
