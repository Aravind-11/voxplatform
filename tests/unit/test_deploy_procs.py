import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from api import security
from scripts.deploy_procs import per_process_cap

_DEPLOY = Path(__file__).resolve().parents[2] / "scripts" / "deploy.sh"


class _FakeRedis:
    """Enough of Redis for RedisLimiter.acquire/release/active without a server."""

    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}

    def register_script(self, _lua):
        def run(keys, args):
            key = keys[0]
            now, exp, member, cap = float(args[0]), float(args[1]), args[2], int(args[3])
            z = self.sets.setdefault(key, {})
            for held, score in list(z.items()):
                if score <= now:
                    del z[held]
            if member in z:
                z[member] = exp
                return 1
            if len(z) >= cap:
                return 0
            z[member] = exp
            return 1
        return run

    def zrem(self, key, member):
        self.sets.get(key, {}).pop(member, None)

    def zcount(self, key, minimum, _maximum):
        now = float(minimum)
        return sum(1 for score in self.sets.get(key, {}).values() if score >= now)

    def scan_iter(self, match="*"):
        prefix = match[:-1] if match.endswith("*") else match
        for key in list(self.sets):
            if key.startswith(prefix):
                yield key


def _both(env, default_concurrency):
    """In-process limiter and a Redis limiter that share nothing but the same env share."""
    local = security.make_limiter(dict(env), default_concurrency)
    remote = security.make_limiter({**env, "VOX_REDIS_URL": "redis://127.0.0.1/9"}, default_concurrency,
                                   redis_client=_FakeRedis())
    return local, remote


def test_per_process_cap_is_an_integer_floor():
    assert per_process_cap(32, 4) == 8
    assert per_process_cap(32, 1) == 32
    assert per_process_cap(100, 6) == 16
    assert 4 * per_process_cap(32, 4) <= 32
    assert 6 * per_process_cap(100, 6) <= 100


def test_per_process_cap_rejects_fewer_than_one_process():
    with pytest.raises(ValueError):
        per_process_cap(32, 0)
    with pytest.raises(ValueError):
        per_process_cap(32, -3)


def test_per_process_cap_sum_does_not_exceed_the_node_cap():
    # Rounding a 0 quotient up to 1 lets procs processes admit more than node_max.
    with pytest.raises(ValueError):
        per_process_cap(3, 4)
    with pytest.raises(ValueError):
        per_process_cap(1, 2)
    # A zero node cap stays 0. Bumping each process to 1 would admit procs calls.
    assert per_process_cap(0, 2) == 0
    assert per_process_cap(0, 1) == 0
    assert per_process_cap(5, 3) == 1
    assert 3 * per_process_cap(5, 3) <= 5
    with pytest.raises(ValueError):
        per_process_cap(-4, 2)


def test_deploy_sh_one_worker_per_port_and_health_on_every_port():
    text = _DEPLOY.read_text()
    guard = re.search(
        r'if ! \[\[ "\$VOX_PROCS" =~ \^\[0-9\]\+\$ \]\] \|\| \[ "\$VOX_PROCS" -lt 1 \]; then\n'
        r'\s*echo "VOX_PROCS must be an integer >= 1.*"; exit 1',
        text,
    )
    assert guard, "VOX_PROCS < 1 must exit 1"
    fn = re.search(r"start_api\(\) \{(?P<body>.*?)\n\}", text, re.S)
    assert fn
    body = fn.group("body")
    assert 'workers="$3"' in body
    uvicorn = [line for line in body.splitlines() if "uvicorn api.main:app" in line]
    assert len(uvicorn) == 1 and "--workers $workers" in uvicorn[0] and "VOX_PROCS" not in uvicorn[0]
    branches = re.search(
        r'if \[ "\$VOX_PROCS" -eq 1 \]; then(?P<one>.*?)else(?P<multi>.*?)fi',
        text,
        re.S,
    )
    assert branches
    one, multi = branches.group("one"), branches.group("multi")
    assert re.search(r'start_api 8080 api (?:"\$VOX_PROCS"|1)(?:\s|$)', one)
    calls = re.findall(r"^[^#\n]*start_api[^\n#]*", multi, re.M)
    assert calls and all(re.search(r"\s1\s*$", call) for call in calls)
    assert "VOX_PROCS" not in "\n".join(calls)
    ready = re.search(r"for _port in \$PORTS; do(?P<body>.*?)done", text, re.S)
    assert ready and "/healthz" in ready.group("body") and "${_port}" in ready.group("body")
    assert "127.0.0.1:8080/healthz" not in text


def test_exported_share_caps_each_process_and_is_not_divided_again():
    # deploy.sh writes per_process_cap into VOX_NODE_MAX_CALLS and leaves VOX_PROCS as the host count.
    share = per_process_cap(32, 4)
    assert share == 8
    env = {"VOX_NODE_MAX_CALLS": str(share), "VOX_PROCS": "4"}
    for lim in _both(env, 64):
        assert lim.process_ceiling == share
        for i in range(share):
            assert lim.acquire("tenant", f"c{i}")
        assert not lim.acquire("tenant", "overflow")
        assert lim.active("tenant") == share and lim.local_active() == share
        assert not lim.acquire("other", "x")
        assert lim.active("other") == 0 and lim.local_active() == share
        assert lim.acquire("tenant", "c0")
        assert lim.active("tenant") == share and lim.local_active() == share
        lim.release("tenant", "c0")
        assert lim.active("tenant") == share - 1 and lim.local_active() == share - 1
        assert lim.acquire("other", "x")
        assert lim.active("other") == 1 and lim.local_active() == share


def test_process_ceiling_zero_admits_nobody_and_unset_follows_the_tenant_cap():
    for lim in _both({"VOX_NODE_MAX_CALLS": "0", "VOX_PROCS": "4"}, 64):
        assert lim.process_ceiling == 0
        assert not lim.acquire("tenant", "c1")
        assert lim.active("tenant") == 0 and lim.local_active() == 0
    for lim in _both({"VOX_PROCS": "4"}, 2):
        assert lim.process_ceiling is None
        assert lim.acquire("t", "1") and lim.acquire("t", "2") and not lim.acquire("t", "3")
        assert lim.active("t") == 2
        assert lim.acquire("o", "1") and lim.acquire("o", "2")
        assert lim.local_active() == 4
    for lim in _both({"VOX_NODE_MAX_CALLS": "8", "VOX_PROCS": "4"}, 1):
        assert lim.acquire("t", "a") and not lim.acquire("t", "b")
        assert lim.active("t") == 1 and lim.local_active() == 1
        assert lim.acquire("u", "c") and lim.local_active() == 2


def test_bad_process_ceiling_fails_startup_instead_of_admitting_without_a_cap():
    assert security.process_call_ceiling({}) is None
    assert security.process_call_ceiling({"VOX_PROCS": "4"}) is None
    assert security.process_call_ceiling({"VOX_NODE_MAX_CALLS": "0"}) == 0
    assert security.process_call_ceiling({"VOX_NODE_MAX_CALLS": "+8", "VOX_PROCS": "4"}) == 8
    for raw in ("-1", "-4", "nope", "1.5", "8.0", "", " ", "True", "4.2", "8a"):
        with pytest.raises(ValueError):
            security.make_limiter({"VOX_NODE_MAX_CALLS": raw, "VOX_REDIS_URL": "redis://127.0.0.1/0"}, 64)


def test_redis_process_ceiling_is_per_process_not_one_shared_key():
    fake = _FakeRedis()
    env = {"VOX_NODE_MAX_CALLS": "2", "VOX_PROCS": "4", "VOX_REDIS_URL": "redis://127.0.0.1/9"}
    a = security.make_limiter(env, 64, redis_client=fake)
    b = security.make_limiter(env, 64, redis_client=fake)
    assert a.acquire("ten", "a1") and a.acquire("ten", "a2") and not a.acquire("ten", "a3")
    assert a.active("ten") == 2 and a.local_active() == 2
    assert b.acquire("ten", "b1") and b.acquire("ten", "b2") and not b.acquire("ten", "b3")
    assert a.local_active() == 2 and b.local_active() == 2
    assert a.active("ten") == 4
    a.release("ten", "a1")
    assert a.local_active() == 1 and a.acquire("ten", "a3") and a.local_active() == 2


def test_concurrent_acquires_do_not_pass_the_process_ceiling():
    for lim in _both({"VOX_NODE_MAX_CALLS": "5", "VOX_PROCS": "4"}, 100):
        with ThreadPoolExecutor(16) as pool:
            got = list(pool.map(lambda i: lim.acquire("t", f"c{i}"), range(40)))
        assert sum(got) == 5
        assert lim.active("t") == 5 and lim.local_active() == 5


def test_build_applies_the_exported_process_ceiling():
    text = Path(__file__).resolve().parents[2].joinpath("api/main.py").read_text()
    assert "security.make_limiter(" in text.split("def build()", 1)[1]


def test_several_processes_refuse_a_sqlite_file_before_they_start():
    """The default deploy DB is one SQLite file. VOX_PROCS>1 on it raises database is locked."""
    from scripts.deploy_procs import require_shared_db

    require_shared_db(1, "/lambda/nfs/arvi/vox.db")
    require_shared_db(1, "")
    require_shared_db(4, "postgresql://vox:pw@localhost/vox")
    require_shared_db(4, "postgres://vox@localhost/vox")
    for db in ("/lambda/nfs/arvi/vox.db", "", "sqlite:///tmp/vox.db", "vox.db"):
        with pytest.raises(ValueError, match="database is locked"):
            require_shared_db(4, db)
    text = _DEPLOY.read_text()
    check = text.find("require_shared_db")
    launch = text.find('if [ "$VOX_PROCS" -eq 1 ]')
    assert check != -1 and launch != -1 and check < launch
    # The check sees the same path the processes will open, including the SQLite default.
    assert '${VOX_DB:-$FS/vox.db}' in text[check - 400:check] or '${VOX_DB:-$FS/vox.db}' in text[check:check + 500]
