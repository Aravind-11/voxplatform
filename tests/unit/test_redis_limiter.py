import shutil
import socket
import subprocess
import time

import pytest

from api.security import RedisLimiter

pytestmark = pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server not installed")


@pytest.fixture(scope="module")
def redis_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    p = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                         stdout=subprocess.DEVNULL)
    import redis
    r = redis.Redis(port=port)
    for _ in range(50):
        try:
            r.ping()
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    yield f"redis://127.0.0.1:{port}/0"
    p.terminate()


def test_cap_is_shared_between_two_nodes_and_per_tenant(redis_url):
    a = RedisLimiter(redis_url, default_limit=2, prefix="t1:")
    b = RedisLimiter(redis_url, default_limit=2, prefix="t1:")            # a second API node
    assert a.acquire("ten", "c1") and b.acquire("ten", "c2")
    assert not a.acquire("ten", "c3") and not b.acquire("ten", "c3")      # cap of 2 holds across nodes
    assert a.acquire("ten", "c1")                                          # re-acquire of a held call is idempotent
    assert a.acquire("other", "x1")                                        # other tenant unaffected
    assert a.active("ten") == 2 and b.total_active() == 3
    b.release("ten", "c2")
    assert a.acquire("ten", "c3") and a.active("ten") == 2
    assert a.acquire("ten", "c9", limit=10)                                # per-tenant override


def test_crashed_node_calls_expire_but_refreshed_ones_survive(redis_url):
    dead = RedisLimiter(redis_url, default_limit=1, ttl_s=0.4, prefix="t2:")
    live = RedisLimiter(redis_url, default_limit=5, ttl_s=0.4, prefix="t2:")
    assert dead.acquire("ten", "c1") and live.acquire("ten", "c2")
    for _ in range(3):
        time.sleep(0.2)
        live.refresh()                                                     # only the live node heartbeats
    assert live.active("ten") == 1                                         # c1 expired, c2 kept
    assert live.acquire("ten", "c3", limit=2)


def test_concurrent_acquires_never_exceed_cap(redis_url):
    from concurrent.futures import ThreadPoolExecutor
    lim = RedisLimiter(redis_url, default_limit=5, prefix="t3:")
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: lim.acquire("ten", f"c{i}"), range(40)))
    assert sum(res) == 5 and lim.active("ten") == 5
