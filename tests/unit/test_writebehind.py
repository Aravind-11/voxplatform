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
