"""Transcript writes must leave the event loop, and a dead Postgres connection is retried once."""
import asyncio
import sqlite3
import sys
import threading
import time
import types

import pytest

from api.runtime import Runtime
from api.store import Store
from api.writebehind import MessageLog, to_thread


class _VAD:
    def prob(self, frame):
        return 0.0


class _LLM:
    async def stream(self, history, audio, followups, tools):
        if False:
            yield ""

    async def transcribe(self, audio):
        return "hi"


class _TTS:
    sample_rate = 24000

    async def synthesize(self, text):
        if False:
            yield None


class _Transport:
    out_rate = 8000

    async def end(self):
        return None

    async def send_audio(self, pcm, rate):
        return None

    async def clear(self):
        return None


@pytest.fixture
def pg_mod():
    """Real psycopg when it is installed, otherwise a stub with the two exception classes."""
    try:
        import psycopg
        created = False
    except ImportError:
        mod = types.ModuleType("psycopg")
        mod.OperationalError = type("OperationalError", (Exception,), {})
        mod.ProgrammingError = type("ProgrammingError", (Exception,), {})
        errors = types.ModuleType("psycopg.errors")
        errors.OperationalError = mod.OperationalError
        errors.ProgrammingError = mod.ProgrammingError
        mod.errors = errors
        sys.modules["psycopg"] = mod
        sys.modules["psycopg.errors"] = errors
        psycopg = mod
        created = True
    yield psycopg
    if created:
        sys.modules.pop("psycopg", None)
        sys.modules.pop("psycopg.errors", None)


def test_flush_blocks_until_the_row_is_visible():
    store = Store(":memory:")
    started, release = threading.Event(), threading.Event()
    real = store.add_message

    def slow(call_id, role, text, t):
        started.set()
        assert release.wait(2)
        return real(call_id, role, text, t)

    store.add_message = slow
    log = MessageLog(store)
    try:
        log.add("call_1", "user", "hello", 1.5)
        assert started.wait(1)
        assert store.messages("call_1") == []
        th = threading.Thread(target=log.flush)
        th.start()
        time.sleep(0.05)
        assert th.is_alive()  # join is waiting on the in-flight insert
        release.set()
        th.join(2)
        assert not th.is_alive()
        assert store.messages("call_1") == [{"role": "user", "text": "hello", "t": 1.5}]
    finally:
        release.set()
        log.close()


def test_close_drains_the_queue_and_is_idempotent():
    store = Store(":memory:")
    log = MessageLog(store)
    log.add("call_1", "agent", "bye", 2.0)
    log.close()
    assert not log._thread.is_alive()
    assert store.messages("call_1") == [{"role": "agent", "text": "bye", "t": 2.0}]
    log.close()


async def test_slow_add_message_does_not_stall_the_loop_during_a_logged_turn():
    """A turn's transcript is logged through Runtime.on_event; the insert sleeps off the loop."""
    store = Store(":memory:")
    call = store.create_call("tnt_x", None, "webSocket", {"systemPrompt": "Be brief."}, {})
    started, done = threading.Event(), {}
    real = store.add_message

    def slow(call_id, role, text, t):
        if not started.is_set():
            started.set()
            time.sleep(0.4)
            done["t"] = time.monotonic()
        return real(call_id, role, text, t)

    store.add_message = slow
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    try:
        sess = rt.build_session(call, _Transport())
        sess.on_event({"type": "user_transcript", "text": "hello"})
        for _ in range(50):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await asyncio.sleep(0.05)
        assert "t" not in done  # the 0.4s insert is still in the worker; the loop was free
        await asyncio.to_thread(rt.flush_messages)
        assert "t" in done
        assert [m["text"] for m in store.messages(call["id"])] == ["hello"]
    finally:
        rt.message_log.close()


async def test_to_thread_still_runs_when_the_task_is_already_cancelled():
    ran = {}

    async def body():
        asyncio.current_task().cancel()

        def slow():
            time.sleep(0.05)
            ran["ok"] = True
            return 7

        return await to_thread(slow)

    task = asyncio.create_task(body())
    try:
        got = await task
    except asyncio.CancelledError:
        got = None
    assert ran.get("ok") is True
    assert got in (7, None)


def test_sqlite_programming_mistakes_are_not_retried():
    store = Store(":memory:")
    assert store.dsn == ":memory:" and store.pool is None
    with pytest.raises(sqlite3.OperationalError):
        store._x("SELECT * FROM no_such_table")


def test_postgres_pool_checks_out_a_connection(pg_mod):
    store = Store(":memory:")
    store.pg = True
    store.dsn = "postgresql://example/db"

    class Cur:
        description = ("x",)
        rowcount = 1

        def __init__(self):
            self.sql = self.args = None

        def execute(self, sql, args=()):
            self.sql, self.args = sql, args

        def fetchall(self):
            return [{"x": 1}]

    class Conn:
        closed = False

        def __init__(self):
            self.cur = Cur()
            self.executed = None

        def cursor(self):
            return self.cur

        def execute(self, sql, args=()):
            self.executed = sql

    class Pool:
        def __init__(self):
            self.conn = Conn()
            self.checkouts = 0

        def connection(self):
            pool = self

            class CM:
                def __enter__(self_cm):
                    pool.checkouts += 1
                    return pool.conn

                def __exit__(self_cm, *a):
                    return False

            return CM()

    pool = Pool()
    store.pool = pool
    store.db = pool
    cur = store._run("SELECT ?", (1,))
    assert cur.fetchone() == {"x": 1} and cur.fetchone() is None
    assert pool.conn.cur.sql == "SELECT %s" and pool.conn.cur.args == (1,)
    assert pool.checkouts == 1
    store.ping()
    assert pool.conn.executed == "SELECT 1" and pool.checkouts == 2


def test_operational_error_reconnects_once_with_the_same_dsn(pg_mod, monkeypatch):
    op = pg_mod.OperationalError
    store = Store(":memory:")
    store.pg = True
    store.pool = None
    dsn = "postgresql://user@localhost/vox"
    store.dsn = dsn

    class Cur:
        rowcount = 1
        description = None

        def __init__(self):
            self.sql = self.args = None

        def execute(self, sql, args=()):
            self.sql, self.args = sql, args

    class Conn:
        def __init__(self, fail):
            self.fail = fail
            self.closed = False
            self.cur = Cur()
            self.did_close = False

        def cursor(self):
            if self.fail:
                raise op("server closed the connection")
            return self.cur

        def close(self):
            self.did_close = True

    bad, good = Conn(True), Conn(False)
    store.db = bad
    opens = {"n": 0}

    def open_postgres():
        opens["n"] += 1
        assert store.dsn == dsn
        store.pool = None
        store.db = good

    monkeypatch.setattr(store, "_open_postgres", open_postgres)
    cur = store._run("SELECT ?", (1,))
    assert opens["n"] == 1 and bad.did_close
    assert cur is good.cur and good.cur.sql == "SELECT %s" and good.cur.args == (1,)

    # a connection that is already closed is the same kind of failure
    closed = Conn(False)
    closed.closed = True

    def cursor_should_not_run(self):
        raise AssertionError("cursor on a closed connection")

    closed.cursor = cursor_should_not_run.__get__(closed, Conn)
    store.db = closed
    opens["n"] = 0
    monkeypatch.setattr(store, "_open_postgres", open_postgres)
    # open_postgres installs `good`, whose cursor succeeds
    store._run("SELECT 1")
    assert opens["n"] == 1 and closed.did_close

    # the retry itself fails: reconnect once, then surface the error
    def open_bad():
        opens["n"] += 1
        store.pool = None
        store.db = Conn(True)

    store.db = Conn(True)
    opens["n"] = 0
    monkeypatch.setattr(store, "_open_postgres", open_bad)
    with pytest.raises(op):
        store._run("SELECT 1")
    assert opens["n"] == 1


def test_row_added_while_closing_is_kept_and_flush_returns():
    """A transcript that arrives after close has started is still stored, in order, and flush cannot hang."""
    store = Store(":memory:")
    started, release = threading.Event(), threading.Event()
    real = store.add_message

    def slow(call_id, role, text, t):
        if text == "first":
            started.set()
            assert release.wait(2)
        return real(call_id, role, text, t)

    store.add_message = slow
    log = MessageLog(store)
    try:
        log.add("c", "user", "first", 1.0)
        assert started.wait(1)
        closer = threading.Thread(target=log.close)
        closer.start()
        for _ in range(200):
            if log._closed:
                break
            time.sleep(0.01)
        assert log._closed
        log.add("c", "user", "second", 2.0)
        release.set()
        closer.join(2)
        assert not closer.is_alive()
        done = threading.Event()

        def _flush():
            log.flush()
            done.set()

        ft = threading.Thread(target=_flush, daemon=True)
        ft.start()
        assert done.wait(1)  # a row queued behind the stop sentinel must not wedge join()
        assert store.messages("c") == [
            {"role": "user", "text": "first", "t": 1.0},
            {"role": "user", "text": "second", "t": 2.0},
        ]
    finally:
        release.set()
        if log._thread.is_alive():
            log._closed = True
            log._q.put(log._stop)
            log._thread.join(1)


def test_failed_insert_is_not_a_successful_flush():
    store = Store(":memory:")
    real = store.add_message

    def boom(call_id, role, text, t):
        if text == "bad":
            raise sqlite3.OperationalError("disk full")
        return real(call_id, role, text, t)

    store.add_message = boom
    log = MessageLog(store)
    try:
        log.add("c", "user", "bad", 1.0)
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            log.flush()
        # one bad insert must not kill the writer: the next row still lands
        log.add("c", "user", "good", 2.0)
        log.flush()
        assert store.messages("c") == [{"role": "user", "text": "good", "t": 2.0}]
    finally:
        log.close()


def test_a_failed_insert_stays_with_its_call():
    """Another call's flush must not swallow this call's disk-full error."""
    store = Store(":memory:")
    real = store.add_message

    def boom(call_id, role, text, t):
        if text == "bad":
            raise sqlite3.OperationalError("disk full")
        return real(call_id, role, text, t)

    store.add_message = boom
    log = MessageLog(store)
    try:
        log.add("a", "user", "bad", 1.0)
        log.add("b", "user", "good", 2.0)
        log.flush("b")
        assert store.messages("b") == [{"role": "user", "text": "good", "t": 2.0}]
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            log.flush("a")
    finally:
        try:
            log.close()
        except sqlite3.OperationalError:
            pass


def test_close_reports_a_failed_drain_and_stays_idempotent():
    store = Store(":memory:")

    def boom(call_id, role, text, t):
        raise sqlite3.OperationalError("disk full")

    store.add_message = boom
    log = MessageLog(store)
    log.add("c", "agent", "bye", 1.0)
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        log.close()
    assert not log._thread.is_alive()
    log.close()


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_call_teardown_does_not_touch_the_database_on_the_event_loop():
    """finish() used to read and end the call on the loop, so a slow UPDATE froze every call."""
    from starlette.testclient import TestClient

    from api import security
    from api.app import Settings, create_app
    from tests.unit.test_api import SETTINGS, silence, tenant

    store = Store(":memory:")
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, SETTINGS)
    hits: list[tuple[str, bool]] = []
    real_update, real_get, real_messages = store.update_call, store.get_call_any, store.messages
    watching = {"on": False}

    def update(call_id, **fields):
        if watching["on"] and (fields.get("status") == "ended" or fields.get("end_reason")):
            hits.append(("update", _on_loop()))
        return real_update(call_id, **fields)

    def get(call_id):
        if watching["on"]:
            hits.append(("get", _on_loop()))
        return real_get(call_id)

    def messages(call_id):
        if watching["on"]:
            hits.append(("messages", _on_loop()))
        return real_messages(call_id)

    store.update_call = update
    store.get_call_any = get
    store.messages = messages
    c = TestClient(app)
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers={"X-API-Key": k}).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    watching["on"] = True
    try:
        with c.websocket_connect(path) as ws:
            ws.send_bytes(silence(1)[0])
    finally:
        watching["on"] = False
    assert ("update", False) in hits and ("messages", False) in hits and ("get", False) in hits
    assert all(not on for _, on in hits)
    cur = real_get(call["id"])
    assert cur["status"] == "ended"


def test_failed_transcript_still_ends_the_call_and_is_not_silent(monkeypatch):
    from starlette.testclient import TestClient

    from api import security
    from api.app import create_app
    from tests.unit.test_api import SETTINGS, silence, tenant

    store = Store(":memory:")
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    lim = security.ConcurrencyLimiter(default_limit=2)
    app = create_app(store, rt, lim, None, SETTINGS)

    def boom(call_id, role, text, t):
        raise sqlite3.OperationalError("disk full")

    store.add_message = boom
    c = TestClient(app)
    k, _ = tenant(c)
    call = c.post("/v1/calls", json={"systemPrompt": "p"}, headers={"X-API-Key": k}).json()
    path = call["joinUrl"].replace("wss://vox.example.com", "")
    caught = None
    try:
        with c.websocket_connect(path) as ws:
            for _ in range(50):
                cur = store.get_call_any(call["id"])
                if cur and cur["status"] == "active":
                    break
                time.sleep(0.02)
            rt.message_log.add(call["id"], "user", "hello", 1.0)
            ws.send_bytes(silence(1)[0])
    except BaseException as exc:  # noqa: BLE001 — the disconnect must report the failed write
        caught = exc
    assert store.get_call_any(call["id"])["status"] == "ended"
    assert lim.total_active() == 0
    assert caught is not None and "disk full" in repr(caught)


def test_sip_setup_writes_stay_off_the_event_loop():
    import socket
    import uuid

    from starlette.testclient import TestClient

    from api import security
    from api.app import Settings, create_app
    from tests.unit.test_api import SETTINGS
    from transports import audiosocket as A

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    store = Store(":memory:")
    tid = store.create_tenant("a")
    ag = store.create_agent(tid, "n", {"systemPrompt": "", "voice": "af_heart", "greeting": "Hi", "tools": []})
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    settings = Settings(admin_key="adm", public_base_url=SETTINGS.public_base_url, audiosocket_port=port,
                        audiosocket_agent=ag["id"])
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, settings)
    hits: list[tuple[str, bool]] = []
    real_create, real_update, real_agent = store.create_call, store.update_call, store.get_agent_any

    def create(*a, **k):
        hits.append(("create", _on_loop()))
        return real_create(*a, **k)

    def update(call_id, **fields):
        if "metadata" in fields or fields.get("status") == "ended" or fields.get("end_reason"):
            hits.append(("update", _on_loop()))
        return real_update(call_id, **fields)

    def get_agent(*a, **k):
        hits.append(("agent", _on_loop()))
        return real_agent(*a, **k)

    store.create_call = create
    store.update_call = update
    store.get_agent_any = get_agent
    with TestClient(app):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        try:
            sock.sendall(A.pack(A.KIND_UUID, uuid.uuid4().bytes))
            for _ in range(100):
                if any(name == "update" for name, _ in hits):
                    break
                time.sleep(0.02)
            sock.sendall(A.pack(A.KIND_HANGUP))
        finally:
            sock.close()
        for _ in range(100):
            calls = store.list_calls(tid)
            if calls and calls[0]["status"] == "ended":
                break
            time.sleep(0.05)
    assert ("create", False) in hits and ("agent", False) in hits and ("update", False) in hits
    assert all(not on for _, on in hits)
    assert store.list_calls(tid)[0]["status"] == "ended"


def test_agent_hangup_records_the_reason_off_the_event_loop(monkeypatch):
    """before_end writes end_reason on the loop today, which stalls every other call at hang-up."""
    from starlette.testclient import TestClient

    from api import security
    from api.app import create_app
    from core.session import Session
    from tests.unit.test_api import SETTINGS, sign, tenant, twilio_form
    from transports.twilio import TwilioTransport

    store = Store(":memory:")
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, SETTINGS)
    hits: list[bool] = []
    real_update = store.update_call
    transport: dict = {}

    def update(call_id, **fields):
        if fields.get("end_reason") == "agent_hangup":
            hits.append(_on_loop())
        return real_update(call_id, **fields)

    store.update_call = update
    real_init = TwilioTransport.__init__

    def init(self, ws, factory):
        real_init(self, ws, factory)
        transport["tr"] = self

    monkeypatch.setattr(TwilioTransport, "__init__", init)
    real_start = Session.start

    async def start(self):
        tr = transport["tr"]
        r = tr.before_end() if tr.before_end else None
        if asyncio.iscoroutine(r):
            await r
        await real_start(self)

    monkeypatch.setattr(Session, "start", start)
    import json

    c = TestClient(app)
    k, tid = tenant(c)
    a = c.post("/v1/agents", json={"name": "n"}, headers={"X-API-Key": k}).json()
    form = twilio_form()
    url = f"https://vox.example.com/twilio/voice/{a['id']}"
    c.post(f"/twilio/voice/{a['id']}", data=form, headers=sign(url, form))
    call = store.list_calls(tid)[0]
    with c.websocket_connect(f"/ws/twilio/{call['id']}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": call["join_token"]}}}))
        # receive_text blocks until the server writes a frame; the reason is recorded before that.
        deadline = time.time() + 3
        while time.time() < deadline and not hits:
            time.sleep(0.02)
    assert hits == [False]


async def test_cancelled_shutdown_still_drains_the_message_log():
    """Lifespan shutdown used asyncio.to_thread, which gives up when the task is cancelled."""
    from api import security
    from api.app import Settings, create_app
    from tests.unit.test_api import SETTINGS

    store = Store(":memory:")
    rt = Runtime(_LLM(), lambda voice: _TTS(), _VAD, store)
    app = create_app(store, rt, security.ConcurrencyLimiter(default_limit=2), None, SETTINGS)
    started, release = threading.Event(), threading.Event()
    real = store.add_message

    def slow(call_id, role, text, t):
        started.set()
        assert release.wait(2)
        return real(call_id, role, text, t)

    store.add_message = slow
    rt.message_log.add("c", "user", "last", 1.0)
    assert started.wait(1)
    handler = next(h for h in app.router.on_shutdown if getattr(h, "__name__", "") == "_stop_message_log")
    task = asyncio.create_task(handler())
    try:
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()  # abandoning close here drops the row on process exit
        release.set()
        try:
            await asyncio.wait_for(task, 2)
        except asyncio.CancelledError:
            pass
        assert store.messages("c") == [{"role": "user", "text": "last", "t": 1.0}]
    finally:
        release.set()


def test_programming_error_is_not_retried(pg_mod, monkeypatch):
    err = pg_mod.ProgrammingError
    store = Store(":memory:")
    store.pg = True
    store.pool = None
    store.dsn = "postgresql://example/db"
    store.db = None

    class Cur:
        def execute(self, sql, args=()):
            raise err("syntax error")

    class Conn:
        closed = False

        def cursor(self):
            return Cur()

        def close(self):
            pass

    store.db = Conn()

    def open_postgres():
        raise AssertionError("programming errors must not reconnect")

    monkeypatch.setattr(store, "_open_postgres", open_postgres)
    with pytest.raises(err):
        store._run("NOPE")
