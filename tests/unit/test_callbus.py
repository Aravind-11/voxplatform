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
