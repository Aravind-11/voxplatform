import json
import queue
import time

from api.callbus import LocalCallBus, RedisCallBus, make_bus


def test_local_bus_delivers_only_to_current_subscribers():
    bus = make_bus("")
    assert isinstance(bus, LocalCallBus)
    got = []
    bus.subscribe("c1", got.append)
    assert bus.publish("c1", {"op": "voicemail", "text": "hi"}) is True
    assert got == [{"op": "voicemail", "text": "hi"}]
    bus.unsubscribe("c1")
    assert bus.publish("c1", {"op": "voicemail", "text": "again"}) is False
    assert got == [{"op": "voicemail", "text": "hi"}]


def test_local_bus_unsubscribe_one_handler_leaves_the_other():
    bus = LocalCallBus()
    a, b = [], []
    ha, hb = a.append, b.append          # bound methods are not interned; keep the same objects
    bus.subscribe("c", ha)
    bus.subscribe("c", hb)
    bus.unsubscribe("c", ha)
    assert bus.publish("c", {"op": "voicemail", "text": "x"}) is True
    assert a == [] and b == [{"op": "voicemail", "text": "x"}]


class _PubSub:
    def __init__(self, redis):
        self.redis = redis
        self.channels: set[str] = set()
        self.queue: queue.Queue = queue.Queue()
        redis.pubsubs.append(self)

    def subscribe(self, *channels):
        self.channels.update(channels)

    def unsubscribe(self, *channels):
        for c in channels:
            self.channels.discard(c)

    def get_message(self, timeout=0.0, **_k):
        try:
            return self.queue.get(timeout=timeout or 0.01)
        except queue.Empty:
            return None


class _Redis:
    """Just enough of redis-py for RedisCallBus: publish fans out to subscribed pubsubs."""

    def __init__(self):
        self.pubsubs: list[_PubSub] = []
        self.published: list[tuple] = []

    def pubsub(self, ignore_subscribe_messages=False):
        return _PubSub(self)

    def publish(self, channel, message):
        self.published.append((channel, message))
        n = 0
        for ps in list(self.pubsubs):
            if channel in ps.channels:
                ps.queue.put({"type": "message", "channel": channel, "data": message})
                n += 1
        return n


def test_redis_bus_reaches_the_other_node_and_does_not_echo_locally():
    r = _Redis()
    a = RedisCallBus(client=r, start_thread=True)
    b = RedisCallBus(client=r, start_thread=True)
    got_a, got_b = [], []
    try:
        a.subscribe("call_1", got_a.append)
        b.subscribe("call_1", got_b.append)
        deadline = time.time() + 2
        while time.time() < deadline and sum("vox:call:call_1" in ps.channels for ps in r.pubsubs) < 2:
            time.sleep(0.02)
        assert sum("vox:call:call_1" in ps.channels for ps in r.pubsubs) >= 2
        assert a.publish("call_1", {"op": "voicemail", "text": "hi"}) is True
        assert got_a == [{"op": "voicemail", "text": "hi"}]          # local handler ran inside publish
        deadline = time.time() + 2
        while time.time() < deadline and not got_b:
            time.sleep(0.02)
        assert got_b == [{"op": "voicemail", "text": "hi"}]
        assert r.published[0][0] == "vox:call:call_1"
        wire = json.loads(r.published[0][1])
        assert wire["origin"] == a.origin and wire["msg"]["text"] == "hi"
        time.sleep(0.4)                                              # own echo must not be delivered twice
        assert got_a == [{"op": "voicemail", "text": "hi"}]
    finally:
        a.close()
        b.close()


def test_redis_publish_reports_a_remote_subscriber_and_ignores_its_own_echo():
    class Counting:
        def __init__(self):
            self.n = 0
            self.published = []

        def publish(self, channel, message):
            self.published.append((channel, message))
            return self.n

    r = Counting()
    bus = RedisCallBus(client=r, start_thread=False)
    assert bus.publish("c2", {"op": "voicemail", "text": "x"}) is False
    r.n = 2
    assert bus.publish("c2", {"op": "voicemail", "text": "x"}) is True
    assert r.published[0][0] == "vox:call:c2"
    seen = []
    bus.subscribe("c2", seen.append)
    bus._on_wire("vox:call:c2", r.published[0][1])                   # our own stamp
    assert seen == []
    bus._on_wire("vox:call:c2", json.dumps({"origin": "other", "msg": {"op": "voicemail", "text": "remote"}}))
    assert seen == [{"op": "voicemail", "text": "remote"}]


def _callbus_errors() -> float:
    from core.metrics import METRICS
    return METRICS.counters.get(("vox_errors_total", (("where", "callbus"),)), 0.0)


def test_create_app_bus_follows_vox_redis_url(monkeypatch):
    """create_app installs a local bus unless VOX_REDIS_URL is set, then a Redis bus."""
    import sys
    import types

    from api import security
    from api.app import Settings, create_app
    from api.runtime import Runtime
    from api.store import Store

    monkeypatch.delenv("VOX_REDIS_URL", raising=False)
    store = Store()
    rt = Runtime(object(), lambda _v: None, lambda: None, store)
    create_app(store, rt, security.ConcurrencyLimiter(1), None, Settings())
    assert isinstance(rt.bus, LocalCallBus)

    client = _Redis()
    seen = {}

    class _FakeRedis:
        @staticmethod
        def from_url(url, decode_responses=False):
            seen["url"] = url
            seen["decode_responses"] = decode_responses
            return client

    mod = types.ModuleType("redis")
    mod.Redis = _FakeRedis
    monkeypatch.setitem(sys.modules, "redis", mod)
    monkeypatch.setenv("VOX_REDIS_URL", "redis://bus.example:6379/0")
    rt2 = Runtime(object(), lambda _v: None, lambda: None, store)
    try:
        create_app(store, rt2, security.ConcurrencyLimiter(1), None, Settings())
        assert isinstance(rt2.bus, RedisCallBus)
        assert seen == {"url": "redis://bus.example:6379/0", "decode_responses": True}
        assert rt2.bus.r is client
    finally:
        if isinstance(getattr(rt2, "bus", None), RedisCallBus):
            rt2.bus.close()


def test_local_bus_publish_does_not_reach_another_call():
    bus = LocalCallBus()
    first, second = [], []
    bus.subscribe("c1", first.append)
    bus.subscribe("c2", second.append)
    assert bus.publish("c1", {"op": "voicemail", "text": "only-c1"}) is True
    assert first == [{"op": "voicemail", "text": "only-c1"}]
    assert second == []


def test_redis_bus_message_does_not_reach_another_call():
    r = _Redis()
    a = RedisCallBus(client=r, start_thread=True)
    b = RedisCallBus(client=r, start_thread=True)
    got_a, got_b_c1, got_b_c2 = [], [], []
    try:
        a.subscribe("c1", got_a.append)
        b.subscribe("c1", got_b_c1.append)
        b.subscribe("c2", got_b_c2.append)
        deadline = time.time() + 2
        while time.time() < deadline and not (
            any("vox:call:c1" in ps.channels for ps in r.pubsubs)
            and any("vox:call:c2" in ps.channels for ps in r.pubsubs)
        ):
            time.sleep(0.02)
        assert any("vox:call:c1" in ps.channels for ps in r.pubsubs)
        assert any("vox:call:c2" in ps.channels for ps in r.pubsubs)
        assert a.publish("c1", {"op": "voicemail", "text": "only-c1"}) is True
        assert got_a == [{"op": "voicemail", "text": "only-c1"}]
        deadline = time.time() + 2
        while time.time() < deadline and not got_b_c1:
            time.sleep(0.02)
        assert got_b_c1 == [{"op": "voicemail", "text": "only-c1"}]
        time.sleep(0.4)   # a second fan-out onto vox:call:c2 would have arrived by now
        assert got_b_c2 == []
        assert a.publish("c2", {"op": "voicemail", "text": "only-c2"}) is True
        deadline = time.time() + 2
        while time.time() < deadline and not got_b_c2:
            time.sleep(0.02)
        assert got_b_c2 == [{"op": "voicemail", "text": "only-c2"}]
        assert got_b_c1 == [{"op": "voicemail", "text": "only-c1"}]
        assert got_a == [{"op": "voicemail", "text": "only-c1"}]
    finally:
        a.close()
        b.close()


def test_local_bus_handler_exception_does_not_stop_the_next():
    bus = LocalCallBus()
    got = []

    def bad(_msg):
        raise RuntimeError("handler down")

    bus.subscribe("c1", bad)
    bus.subscribe("c1", got.append)
    before = _callbus_errors()
    assert bus.publish("c1", {"op": "voicemail", "text": "hi"}) is True
    assert got == [{"op": "voicemail", "text": "hi"}]
    assert _callbus_errors() == before + 1


def test_redis_publish_error_still_delivers_locally():
    class Boom:
        def publish(self, _channel, _message):
            raise RuntimeError("redis down")

    bus = RedisCallBus(client=Boom(), start_thread=False)
    got = []
    bus.subscribe("c1", got.append)
    before = _callbus_errors()
    assert bus.publish("c1", {"op": "voicemail", "text": "hi"}) is True
    assert got == [{"op": "voicemail", "text": "hi"}]
    assert _callbus_errors() == before + 1


def test_redis_listener_delivers_on_the_pubsub_after_get_message_fails():
    class Flaky(_Redis):
        def __init__(self):
            super().__init__()
            self.raised = False
            self.broken = None

        def pubsub(self, ignore_subscribe_messages=False):
            ps = super().pubsub(ignore_subscribe_messages=ignore_subscribe_messages)
            orig = ps.get_message

            def get_message(timeout=0.0, **_k):
                if not self.raised and ps.channels:
                    self.raised = True
                    self.broken = ps
                    raise ConnectionError("redis down")
                return orig(timeout=timeout, **_k)

            ps.get_message = get_message
            return ps

    r = Flaky()
    bus = RedisCallBus(client=r, start_thread=True)
    got = []
    before = _callbus_errors()
    try:
        bus.subscribe("c1", got.append)
        deadline = time.time() + 3
        fresh = []
        while time.time() < deadline:
            fresh = [ps for ps in r.pubsubs if ps is not r.broken and "vox:call:c1" in ps.channels]
            if r.raised and fresh:
                break
            time.sleep(0.02)
        assert r.raised and fresh, "listener did not open a new pubsub after get_message failed"
        r.publish("vox:call:c1", json.dumps({"origin": "other", "msg": {"op": "voicemail", "text": "after"}}))
        deadline = time.time() + 2
        while time.time() < deadline and not got:
            time.sleep(0.02)
        assert got == [{"op": "voicemail", "text": "after"}]
        assert _callbus_errors() == before + 1
    finally:
        bus.close()
