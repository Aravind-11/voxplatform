"""SQL store: SQLite (single node, default) or PostgreSQL (shared by several API nodes).

Store(path)               -> SQLite file (or ':memory:')
Store("postgresql://...") -> PostgreSQL via psycopg 3
The SQL is written once with `?` placeholders and translated for PostgreSQL. Every read that serves a tenant
filters by tenant_id.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants(
  id TEXT PRIMARY KEY, name TEXT, webhook_url TEXT, webhook_secret TEXT,
  concurrency_limit INTEGER, created {REAL});
CREATE TABLE IF NOT EXISTS api_keys(
  key_hash TEXT PRIMARY KEY, tenant_id TEXT, label TEXT, created {REAL});
CREATE TABLE IF NOT EXISTS agents(
  id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, config TEXT, created {REAL});
CREATE TABLE IF NOT EXISTS calls(
  id TEXT PRIMARY KEY, tenant_id TEXT, agent_id TEXT, medium TEXT, status TEXT, config TEXT,
  join_token TEXT, metadata TEXT, created {REAL}, joined {REAL}, ended {REAL}, end_reason TEXT);
CREATE INDEX IF NOT EXISTS calls_tenant ON calls(tenant_id, created);
CREATE INDEX IF NOT EXISTS calls_tenant_created ON calls(tenant_id, created);
CREATE TABLE IF NOT EXISTS messages(
  id {SERIAL}, call_id TEXT, role TEXT, text TEXT, t {REAL});
CREATE INDEX IF NOT EXISTS messages_call ON messages(call_id, id);
CREATE TABLE IF NOT EXISTS audit(
  id {SERIAL}, tenant_id TEXT, t {REAL}, actor TEXT, action TEXT, target TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS audit_tenant ON audit(tenant_id, id);
CREATE TABLE IF NOT EXISTS tools(
  tenant_id TEXT, name TEXT, spec TEXT, created {REAL}, PRIMARY KEY(tenant_id, name));
CREATE TABLE IF NOT EXISTS corpora(
  id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, created {REAL}, UNIQUE(tenant_id, name));
CREATE TABLE IF NOT EXISTS docs(
  id TEXT PRIMARY KEY, corpus_id TEXT, tenant_id TEXT, title TEXT, text TEXT, created {REAL});
CREATE INDEX IF NOT EXISTS docs_corpus ON docs(corpus_id);
CREATE TABLE IF NOT EXISTS ledger(
  id {SERIAL}, tenant_id TEXT, t {REAL}, kind TEXT, micros BIGINT, ref TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ledger_tenant ON ledger(tenant_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS ledger_ref ON ledger(kind, ref);
CREATE TABLE IF NOT EXISTS numbers(
  e164 TEXT PRIMARY KEY, tenant_id TEXT, agent_id TEXT);
CREATE TABLE IF NOT EXISTS sip_setup(
  uuid TEXT PRIMARY KEY, caller TEXT, exten TEXT, callee TEXT, created {REAL});
"""

# columns added after the first release: (table, column, type)
MIGRATIONS = [("api_keys", "id", "TEXT"), ("api_keys", "revoked", "{REAL}"), ("tenants", "rate_per_minute", "{REAL}"),
              ("api_keys", "role", "TEXT"), ("tenants", "retention_days", "INTEGER"),
              ("tenants", "email", "TEXT"), ("tenants", "prepaid", "INTEGER")]
ROLES = ("admin", "developer", "viewer")      # viewer: read only; developer: read + change agents/tools/calls; admin: also keys, audit, webhook, deletes


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def normalize_e164(raw: str) -> str:
    """Drop whitespace and keep a leading '+', so '+1 555 111 0000' matches '+15551110000'."""
    return "".join(str(raw or "").split())


class _BufferedCursor:
    """Rows copied off a pooled connection before it is returned to the pool."""

    def __init__(self, rows: list, rowcount: int):
        self._rows = rows
        self._i = 0
        self.rowcount = rowcount

    def fetchone(self):
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self):
        rows = self._rows[self._i:]
        self._i = len(self._rows)
        return rows


def _buffer_cursor(cur: Any) -> _BufferedCursor:
    if getattr(cur, "description", None):
        rows = [dict(r) for r in cur.fetchall()]
    else:
        rows = []
    return _BufferedCursor(rows, cur.rowcount)


def _connection_closed(conn: Any) -> bool:
    if conn is None:
        return True
    flag = getattr(conn, "closed", False)
    if callable(flag):
        try:
            flag = flag()
        except Exception:  # noqa: BLE001
            return False
    return bool(flag)


class Store:
    def __init__(self, path: str = ":memory:"):
        self.lock = threading.RLock()
        self.dsn = path
        self.pool = None
        self.pg = path.startswith(("postgres://", "postgresql://"))
        if self.pg:
            self._open_postgres()
            real, serial = "DOUBLE PRECISION", "BIGSERIAL PRIMARY KEY"
        else:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            real, serial = "REAL", "INTEGER PRIMARY KEY AUTOINCREMENT"
        self.real = real
        with self.lock:
            for stmt in SCHEMA.format(REAL=real, SERIAL=serial).split(";"):
                if stmt.strip():
                    self._run(stmt)
            for table, col, typ in MIGRATIONS:
                self._add_column(table, col, typ.format(REAL=real))
            self._run("CREATE UNIQUE INDEX IF NOT EXISTS tenants_email ON tenants(email)")    # after the email column exists
            self._run("CREATE INDEX IF NOT EXISTS calls_tenant_created ON calls(tenant_id, created)")  # existing DBs predate the index
            self._x("UPDATE api_keys SET role='admin' WHERE role IS NULL")      # keys from before roles existed keep full rights
            for r in self._all("SELECT key_hash FROM api_keys WHERE id IS NULL"):
                self._x("UPDATE api_keys SET id=? WHERE key_hash=?", (new_id("key"), r["key_hash"]))

    # ---- low level
    def _open_postgres(self) -> None:
        """Connect to `self.dsn`. A pool is used when psycopg_pool imports; otherwise one connection."""
        import psycopg
        from psycopg.rows import dict_row
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            ConnectionPool = None  # noqa: N806  (optional dependency)
        if ConnectionPool is not None:
            try:
                max_size = int(os.environ.get("VOX_DB_POOL", "8"))
            except ValueError:
                max_size = 8
            max_size = max(1, max_size)
            self.pool = ConnectionPool(self.dsn, min_size=1, max_size=max_size, open=True,
                                       kwargs={"autocommit": True, "row_factory": dict_row})
            self.db = self.pool
            return
        self.pool = None
        self.db = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def _reconnect(self) -> None:
        """Replace a dead Postgres connection or pool, same DSN. One retry, not a reconnect loop."""
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception:  # noqa: BLE001
                pass
            self.pool = None
        else:
            db = getattr(self, "db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
        self._open_postgres()

    def _transient_pg(self, exc: BaseException) -> bool:
        """Operational failures and a closed connection are retried once. Programming errors are not."""
        try:
            import psycopg
        except ImportError:
            return False
        if isinstance(exc, psycopg.ProgrammingError):
            return False
        if isinstance(exc, psycopg.OperationalError):
            return True
        if self.pool is None and _connection_closed(getattr(self, "db", None)):
            return True
        return False

    def _ensure_open(self, conn: Any) -> None:
        if not _connection_closed(conn):
            return
        import psycopg
        raise psycopg.OperationalError("connection is closed")

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def _run(self, sql: str, args: tuple = (), *, _retry: bool = True):
        if not self.pg:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur
        try:
            return self._run_pg(sql, args)
        except Exception as exc:  # noqa: BLE001
            if _retry and self._transient_pg(exc):
                self._reconnect()
                return self._run(sql, args, _retry=False)
            raise

    def _run_pg(self, sql: str, args: tuple = ()):
        sql = self._sql(sql)
        if self.pool is not None:
            with self.pool.connection() as conn:  # checks a connection out of the pool
                self._ensure_open(conn)
                cur = conn.cursor()
                cur.execute(sql, args)
                return _buffer_cursor(cur)
        self._ensure_open(self.db)
        cur = self.db.cursor()
        cur.execute(sql, args)
        return cur

    def _add_column(self, table: str, col: str, typ: str) -> None:
        try:
            if self.pg:
                self._run(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
            else:
                self._run(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        except Exception:  # noqa: BLE001  (SQLite: column already exists)
            pass

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            return self._run(sql, args).rowcount

    def _one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        with self.lock:
            r = self._run(sql, args).fetchone()
            return dict(r) if r is not None else None

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self._run(sql, args).fetchall()]

    def ping(self) -> None:
        """Raises if the database cannot answer (readiness check)."""
        if self.pool is not None:
            with self.pool.connection() as conn:  # readiness checks out a real connection
                conn.execute("SELECT 1")
            return
        self._one("SELECT 1 AS x")

    def close(self) -> None:
        with self.lock:
            if self.pool is not None:
                self.pool.close()
            else:
                self.db.close()

    # ---- tenants / keys
    def create_tenant(self, name: str, concurrency_limit: Optional[int] = None, email: Optional[str] = None,
                      prepaid: bool = False) -> str:
        tid = new_id("tnt")
        self._x("INSERT INTO tenants(id, name, webhook_url, webhook_secret, concurrency_limit, created, email, prepaid) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (tid, name, None, None, concurrency_limit, time.time(), email.lower() if email else None, 1 if prepaid else 0))
        return tid

    def tenant_by_email(self, email: str) -> Optional[dict]:
        return self._one("SELECT * FROM tenants WHERE email=?", (email.lower(),))

    def set_prepaid(self, tenant_id: str, prepaid: bool) -> None:
        self._x("UPDATE tenants SET prepaid=? WHERE id=?", (1 if prepaid else 0, tenant_id))

    # ---- prepaid balance. The ledger is the single source of truth: balance = sum of entries, in micro-dollars
    # (1 USD = 1,000,000), so there is no separate counter that could drift. (kind, ref) is unique: a Stripe event
    # or a call can be applied only once, however many times it is delivered.
    def ledger_add(self, tenant_id: str, micros: int, kind: str, ref: str = "", detail: Optional[dict] = None) -> bool:
        """Adds an entry. Returns False (and changes nothing) if an entry with this kind and ref already exists."""
        ref = ref or new_id("ref")                     # entries without a natural reference are never deduplicated
        return self._x("INSERT INTO ledger(tenant_id, t, kind, micros, ref, detail) VALUES(?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                       (tenant_id, time.time(), kind, int(micros), ref, json.dumps(detail or {}, default=str))) > 0

    def balance_micros(self, tenant_id: str) -> int:
        r = self._one("SELECT COALESCE(SUM(micros), 0) AS b FROM ledger WHERE tenant_id=?", (tenant_id,))
        return int(r["b"])

    def ledger_list(self, tenant_id: str, limit: int = 50) -> list[dict]:
        rows = self._all("SELECT t, kind, micros, ref, detail FROM ledger WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
                         (tenant_id, max(1, min(limit, 500))))
        for r in rows:
            r["detail"] = json.loads(r["detail"] or "{}")
            r["micros"] = int(r["micros"])
        return rows

    def add_key(self, tenant_id: str, key_hash: str, label: str = "", role: str = "admin") -> str:
        assert role in ROLES, role
        kid = new_id("key")
        self._x("INSERT INTO api_keys(key_hash, tenant_id, label, created, id, revoked, role) VALUES(?,?,?,?,?,NULL,?)",
                (key_hash, tenant_id, label, time.time(), kid, role))
        return kid

    def tenant_by_key(self, key_hash: str) -> Optional[dict]:
        return self._one("SELECT t.*, k.id AS key_id, k.role AS role FROM tenants t JOIN api_keys k ON k.tenant_id=t.id "
                         "WHERE k.key_hash=? AND k.revoked IS NULL", (key_hash,))

    def key_id_for_hash(self, key_hash: str) -> Optional[str]:
        r = self._one("SELECT id FROM api_keys WHERE key_hash=? AND revoked IS NULL", (key_hash,))
        return r["id"] if r else None

    def list_keys(self, tenant_id: str) -> list[dict]:
        return self._all("SELECT id, label, created, revoked, role FROM api_keys WHERE tenant_id=? ORDER BY created",
                         (tenant_id,))

    def revoke_key(self, tenant_id: str, key_id: str) -> bool:
        return self._x("UPDATE api_keys SET revoked=? WHERE id=? AND tenant_id=? AND revoked IS NULL",
                       (time.time(), key_id, tenant_id)) > 0

    def active_admin_key_count(self, tenant_id: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM api_keys WHERE tenant_id=? AND revoked IS NULL AND role='admin'",
                         (tenant_id,))["n"]

    def list_tenants(self) -> list[dict]:
        return self._all("SELECT * FROM tenants ORDER BY created")

    def expired_calls(self, tenant_id: str, cutoff: float, limit: int = 1000) -> list[str]:
        """Ids of ended calls whose end time is before `cutoff`."""
        return [r["id"] for r in self._all(
            "SELECT id FROM calls WHERE tenant_id=? AND status='ended' AND COALESCE(ended, created) < ? LIMIT ?",
            (tenant_id, cutoff, limit))]

    def update_tenant(self, tenant_id: str, concurrency_limit: Optional[int] = None,
                      rate_per_minute: Optional[float] = None, retention_days: Optional[int] = None) -> Optional[dict]:
        if self.get_tenant(tenant_id) is None:
            return None
        if concurrency_limit is not None:
            self._x("UPDATE tenants SET concurrency_limit=? WHERE id=?", (concurrency_limit, tenant_id))
        if rate_per_minute is not None:
            self._x("UPDATE tenants SET rate_per_minute=? WHERE id=?", (rate_per_minute, tenant_id))
        if retention_days is not None:              # 0 turns retention off for this tenant
            self._x("UPDATE tenants SET retention_days=? WHERE id=?", (retention_days or None, tenant_id))
        return self.get_tenant(tenant_id)

    def delete_call(self, tenant_id: str, call_id: str) -> bool:
        call = self.get_call(tenant_id, call_id)
        if call is None:
            return False
        uid = str((call.get("metadata") or {}).get("callSid") or "")
        self._x("DELETE FROM messages WHERE call_id=?", (call_id,))
        ok = self._x("DELETE FROM calls WHERE id=? AND tenant_id=?", (call_id, tenant_id)) > 0
        if ok and uid:
            self.delete_sip_setup(uid)   # the staging copy of the caller id is not the call record
        return ok

    def get_tenant(self, tenant_id: str) -> Optional[dict]:
        return self._one("SELECT * FROM tenants WHERE id=?", (tenant_id,))

    def set_webhook(self, tenant_id: str, url: str, secret: str) -> None:
        self._x("UPDATE tenants SET webhook_url=?, webhook_secret=? WHERE id=?", (url, secret, tenant_id))

    # ---- audit log
    def audit(self, tenant_id: str, actor: str, action: str, target: str = "", detail: Optional[dict] = None) -> None:
        self._x("INSERT INTO audit(tenant_id, t, actor, action, target, detail) VALUES(?,?,?,?,?,?)",
                (tenant_id, time.time(), actor, action, target, json.dumps(detail or {}, default=str)))

    def list_audit(self, tenant_id: str, limit: int = 100, before_id: Optional[int] = None) -> list[dict]:
        sql, args = "SELECT * FROM audit WHERE tenant_id=?", [tenant_id]
        if before_id is not None:
            sql += " AND id<?"
            args.append(before_id)
        rows = self._all(sql + " ORDER BY id DESC LIMIT ?", (*args, max(1, min(limit, 500))))
        for r in rows:
            r["detail"] = json.loads(r["detail"] or "{}")
        return rows

    # ---- shared tool library
    def put_tool(self, tenant_id: str, name: str, spec: dict) -> None:
        with self.lock:
            if self._x("UPDATE tools SET spec=? WHERE tenant_id=? AND name=?", (json.dumps(spec), tenant_id, name)) == 0:
                self._x("INSERT INTO tools VALUES(?,?,?,?)", (tenant_id, name, json.dumps(spec), time.time()))

    def get_tool(self, tenant_id: str, name: str) -> Optional[dict]:
        r = self._one("SELECT * FROM tools WHERE tenant_id=? AND name=?", (tenant_id, name))
        return {**r, "spec": json.loads(r["spec"])} if r else None

    def list_tools(self, tenant_id: str) -> list[dict]:
        return [{**r, "spec": json.loads(r["spec"])} for r in
                self._all("SELECT * FROM tools WHERE tenant_id=? ORDER BY name", (tenant_id,))]

    def delete_tool(self, tenant_id: str, name: str) -> bool:
        return self._x("DELETE FROM tools WHERE tenant_id=? AND name=?", (tenant_id, name)) > 0

    # ---- knowledge bases
    def create_corpus(self, tenant_id: str, name: str) -> Optional[dict]:
        if self.get_corpus(tenant_id, name):
            return None
        cid = new_id("kb")
        self._x("INSERT INTO corpora VALUES(?,?,?,?)", (cid, tenant_id, name, time.time()))
        return self.get_corpus(tenant_id, name)

    def get_corpus(self, tenant_id: str, name: str) -> Optional[dict]:
        return self._one("SELECT * FROM corpora WHERE tenant_id=? AND name=?", (tenant_id, name))

    def list_corpora(self, tenant_id: str) -> list[dict]:
        return self._all("SELECT c.id, c.name, c.created, (SELECT COUNT(*) FROM docs d WHERE d.corpus_id=c.id) AS docs "
                         "FROM corpora c WHERE c.tenant_id=? ORDER BY c.name", (tenant_id,))

    def delete_corpus(self, tenant_id: str, name: str) -> bool:
        c = self.get_corpus(tenant_id, name)
        if not c:
            return False
        self._x("DELETE FROM docs WHERE corpus_id=?", (c["id"],))
        return self._x("DELETE FROM corpora WHERE id=?", (c["id"],)) > 0

    def add_doc(self, tenant_id: str, corpus_id: str, title: str, text: str) -> dict:
        did = new_id("doc")
        self._x("INSERT INTO docs VALUES(?,?,?,?,?,?)", (did, corpus_id, tenant_id, title, text, time.time()))
        return {"id": did, "title": title, "chars": len(text)}

    def list_docs(self, tenant_id: str, corpus_id: str) -> list[dict]:
        return self._all("SELECT id, title, LENGTH(text) AS chars, created FROM docs WHERE corpus_id=? AND tenant_id=? "
                         "ORDER BY created", (corpus_id, tenant_id))

    def delete_doc(self, tenant_id: str, corpus_id: str, doc_id: str) -> bool:
        return self._x("DELETE FROM docs WHERE id=? AND corpus_id=? AND tenant_id=?", (doc_id, corpus_id, tenant_id)) > 0

    def corpus_docs(self, tenant_id: str, name: str) -> tuple[Optional[str], list[dict]]:
        """(version, [{id,title,text}]) for a knowledge base; version changes whenever its contents do."""
        c = self.get_corpus(tenant_id, name)
        if not c:
            return None, []
        docs = self._all("SELECT id, title, text FROM docs WHERE corpus_id=? ORDER BY created", (c["id"],))
        return f"{len(docs)}:{docs[-1]['id'] if docs else ''}", docs

    # ---- agents
    def create_agent(self, tenant_id: str, name: str, config: dict) -> dict:
        aid = new_id("agt")
        self._x("INSERT INTO agents VALUES(?,?,?,?,?)",
                (aid, tenant_id, name, json.dumps(config), time.time()))
        return self.get_agent(tenant_id, aid)  # type: ignore[return-value]

    @staticmethod
    def _agent(r: dict) -> dict:
        d = dict(r)
        d["config"] = json.loads(d["config"])
        return d

    def get_agent(self, tenant_id: str, agent_id: str) -> Optional[dict]:
        r = self._one("SELECT * FROM agents WHERE id=? AND tenant_id=?", (agent_id, tenant_id))
        return self._agent(r) if r else None

    def get_agent_any(self, agent_id: str) -> Optional[dict]:
        """Unscoped lookup for inbound telephony webhooks (agent id is the capability)."""
        r = self._one("SELECT * FROM agents WHERE id=?", (agent_id,))
        return self._agent(r) if r else None

    def update_agent(self, tenant_id: str, agent_id: str, name: Optional[str], config: dict) -> Optional[dict]:
        cur = self.get_agent(tenant_id, agent_id)
        if cur is None:
            return None
        self._x("UPDATE agents SET name=?, config=? WHERE id=? AND tenant_id=?",
                (name or cur["name"], json.dumps(config), agent_id, tenant_id))
        return self.get_agent(tenant_id, agent_id)

    def delete_agent(self, tenant_id: str, agent_id: str) -> bool:
        return self._x("DELETE FROM agents WHERE id=? AND tenant_id=?", (agent_id, tenant_id)) > 0

    def list_agents(self, tenant_id: str) -> list[dict]:
        return [self._agent(r) for r in
                self._all("SELECT * FROM agents WHERE tenant_id=? ORDER BY created", (tenant_id,))]

    # ---- calls
    @staticmethod
    def _call(r: dict) -> dict:
        d = dict(r)
        d["config"] = json.loads(d["config"])
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return d

    def create_call(self, tenant_id: str, agent_id: Optional[str], medium: str, config: dict,
                    metadata: Optional[dict] = None, *, call_id: Optional[str] = None) -> dict:
        cid, token = call_id or new_id("call"), secrets.token_urlsafe(24)
        self._x("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, tenant_id, agent_id, medium, "created", json.dumps(config), token,
                 json.dumps(metadata or {}), time.time(), None, None, None))
        return self.get_call_any(cid)  # type: ignore[return-value]

    def get_call(self, tenant_id: str, call_id: str) -> Optional[dict]:
        r = self._one("SELECT * FROM calls WHERE id=? AND tenant_id=?", (call_id, tenant_id))
        return self._call(r) if r else None

    def get_call_any(self, call_id: str) -> Optional[dict]:
        r = self._one("SELECT * FROM calls WHERE id=?", (call_id,))
        return self._call(r) if r else None

    def find_open_twilio_call(self, tenant_id: str, call_sid: str) -> Optional[dict]:
        """Oldest non-ended Twilio row for this tenant whose metadata callSid matches.

        Twilio retries the voice webhook when the response is lost. The retry must reuse that row
        instead of reserving a second slot the first status callback can never name.
        """
        if not call_sid:
            return None
        rows = self._all(
            "SELECT * FROM calls WHERE tenant_id=? AND medium=? AND status IN ('created', 'active') ORDER BY created",
            (tenant_id, "twilio"))
        for r in rows:
            call = self._call(r)
            meta = call.get("metadata") or {}
            if str(meta.get("callSid", "") or "") == call_sid:
                return call
        return None

    def find_call_by_sid(self, call_sid: str, medium: Optional[str] = None) -> Optional[dict]:
        """Newest call whose metadata JSON contains this callSid (AudioSocket UUID or Twilio SID).

        json.dumps writes a space after the colon. LIKE wildcards in the id are escaped so a
        sid cannot widen the match.
        """
        if not call_sid:
            return None
        needle = '"callSid": ' + json.dumps(call_sid)
        safe = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._all("SELECT * FROM calls WHERE metadata LIKE ? ESCAPE '\\' ORDER BY created DESC",
                         (f"%{safe}%",))
        for r in rows:
            call = self._call(r)
            if medium is None or call["medium"] == medium:
                return call
        return None

    def list_calls(self, tenant_id: str, limit: int = 50) -> list[dict]:
        return [self._call(r) for r in self._all(
            "SELECT * FROM calls WHERE tenant_id=? ORDER BY created DESC LIMIT ?", (tenant_id, limit))]

    def count_outbound(self, tenant_id: str, since: float) -> int:
        """Outbound calls created at or after `since`. Matches json.dumps' space after the colon."""
        row = self._one("SELECT COUNT(*) AS n FROM calls WHERE tenant_id=? AND created>=? AND metadata LIKE ?",
                        (tenant_id, since, '%"direction": "outbound"%'))
        return int(row["n"]) if row else 0

    def update_call(self, call_id: str, **fields: Any) -> None:
        allowed = {"status", "joined", "ended", "end_reason", "metadata"}
        assert set(fields) <= allowed, f"unexpected fields {set(fields) - allowed}"
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self._x(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))

    def update_call_if_status(self, call_id: str, current: str, **fields: Any) -> bool:
        """Apply `fields` only while the row's status is still `current`. False if that lost the race."""
        allowed = {"status", "joined", "ended", "end_reason", "metadata"}
        assert set(fields) <= allowed, f"unexpected fields {set(fields) - allowed}"
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        if not fields:
            return False
        cols = ", ".join(f"{k}=?" for k in fields)
        return self._x(
            f"UPDATE calls SET {cols} WHERE id=? AND status=?",
            (*fields.values(), call_id, current),
        ) > 0

    def merge_call_metadata(self, call_id: str, patch: dict) -> Optional[dict]:
        """Merge keys into metadata. The compare-and-swap retries so one writer cannot drop the other's keys."""
        if not isinstance(patch, dict) or not patch:
            raise ValueError("metadata patch must be a non-empty dict")
        for _ in range(8):
            with self.lock:
                row = self._run("SELECT metadata FROM calls WHERE id=?", (call_id,)).fetchone()
                if row is None:
                    return None
                raw = dict(row)["metadata"]
                meta = json.loads(raw or "{}")
                meta.update(patch)
                new = json.dumps(meta)
                if raw is not None and new == raw:
                    return meta
                if raw is None:
                    n = self._run("UPDATE calls SET metadata=? WHERE id=? AND metadata IS NULL",
                                  (new, call_id)).rowcount
                else:
                    n = self._run("UPDATE calls SET metadata=? WHERE id=? AND metadata=?",
                                  (new, call_id, raw)).rowcount
                if n:
                    return meta
        raise RuntimeError(f"could not merge metadata for {call_id}")

    def find_sip_call(self, *, call_sid: str = "", audio_uuid: str = "") -> Optional[dict]:
        """Newest not-yet-joined sip call with this ARI channel id or AudioSocket UUID."""
        if call_sid:
            field, value = "callSid", call_sid
        elif audio_uuid:
            field, value = "audioSocketUuid", audio_uuid
        else:
            return None
        if self.pg:
            sql = ("SELECT * FROM calls WHERE medium=? AND status=? AND metadata::json->>? = ? "
                   "ORDER BY created DESC LIMIT 1")
            args = ("sip", "created", field, value)
        else:
            sql = ("SELECT * FROM calls WHERE medium=? AND status=? AND json_extract(metadata, ?) = ? "
                   "ORDER BY created DESC LIMIT 1")
            args = ("sip", "created", f"$.{field}", value)
        r = self._one(sql, args)
        return self._call(r) if r else None

    def find_sip_by_audio_uuid(self, audio_uuid: str) -> Optional[dict]:
        """Newest sip call with this AudioSocket UUID, including one that has already ended.
        The dialplan reads a transfer target after the audio leg drops and the row is ended."""
        if not audio_uuid:
            return None
        if self.pg:
            sql = ("SELECT * FROM calls WHERE medium=? AND metadata::json->>? = ? "
                   "ORDER BY created DESC LIMIT 1")
            args = ("sip", "audioSocketUuid", audio_uuid)
        else:
            sql = ("SELECT * FROM calls WHERE medium=? AND json_extract(metadata, ?) = ? "
                   "ORDER BY created DESC LIMIT 1")
            args = ("sip", "$.audioSocketUuid", audio_uuid)
        r = self._one(sql, args)
        return self._call(r) if r else None

    # ---- usage
    def usage(self, tenant_id: str, days: int = 30) -> dict:
        """Totals plus per-day and per-agent breakdowns (UTC days) over the last `days` days."""
        since = time.time() - days * 86400
        rows = self._all("SELECT agent_id, joined, ended, medium FROM calls WHERE tenant_id=? AND created>=?",
                         (tenant_id, since))
        by_day: dict[str, dict] = {}
        by_agent: dict[str, dict] = {}
        total_s = 0.0
        for r in rows:
            secs = (r["ended"] - r["joined"]) if r["joined"] and r["ended"] else 0.0
            day = datetime.fromtimestamp(r["joined"] or since, timezone.utc).strftime("%Y-%m-%d")
            for bucket, key in ((by_day, day), (by_agent, r["agent_id"] or "(none)")):
                b = bucket.setdefault(key, {"calls": 0, "seconds": 0.0})
                b["calls"] += 1
                b["seconds"] += secs
            total_s += secs
        fmt = lambda d: [{"key": k, "calls": v["calls"], "minutes": round(v["seconds"] / 60, 3)}  # noqa: E731
                         for k, v in sorted(d.items())]
        return {"calls": len(rows), "minutes": round(total_s / 60, 3), "days": days,
                "byDay": fmt(by_day), "byAgent": fmt(by_agent)}

    # ---- messages
    def add_message(self, call_id: str, role: str, text: str, t: float) -> None:
        self._x("INSERT INTO messages(call_id, role, text, t) VALUES(?,?,?,?)", (call_id, role, text, t))

    def messages(self, call_id: str) -> list[dict]:
        return self._all("SELECT role, text, t FROM messages WHERE call_id=? ORDER BY id", (call_id,))

    # ---- dialed numbers (SIP). e164 is globally unique: one number belongs to one tenant.
    def put_number(self, tenant_id: str, e164: str, agent_id: str) -> bool:
        """Point e164 at agent_id for this tenant. A row another tenant already owns is left unchanged.
        Returns whether this tenant owns the number after the write."""
        e164 = normalize_e164(e164)
        # Do not set tenant_id on conflict. Two API nodes can both pass the 409 check before either writes;
        # the WHERE keeps the first owner, and a same-tenant put can still change agent_id.
        self._x("INSERT INTO numbers(e164, tenant_id, agent_id) VALUES(?,?,?) "
                "ON CONFLICT(e164) DO UPDATE SET agent_id=excluded.agent_id "
                "WHERE numbers.tenant_id=excluded.tenant_id",
                (e164, tenant_id, agent_id))
        row = self._one("SELECT tenant_id FROM numbers WHERE e164=?", (e164,))
        return bool(row and row["tenant_id"] == tenant_id)

    def get_number(self, e164: str) -> Optional[dict]:
        key = normalize_e164(e164)
        # _X. posts the dialed digits with no leading '+'. Rows are stored as E.164.
        if key.isdigit():
            key = "+" + key
        r = self._one("SELECT e164, tenant_id, agent_id FROM numbers WHERE e164=?", (key,))
        return dict(r) if r else None

    def delete_number(self, tenant_id: str, e164: str) -> bool:
        return self._x("DELETE FROM numbers WHERE e164=? AND tenant_id=?", (normalize_e164(e164), tenant_id)) > 0

    def list_numbers(self, tenant_id: str) -> list[dict]:
        return self._all("SELECT e164, tenant_id, agent_id FROM numbers WHERE tenant_id=? ORDER BY e164", (tenant_id,))

    # ---- SIP setup rows written by the dialplan just before AudioSocket connects.
    # The third argument is the dialed number. It is stored as both exten and callee.
    # get_sip_setup hides rows older than max_age (default 60s) so a stale post cannot route a new call.
    # Pass max_age=None to read the table itself (retention checks). delete and purge remove the row.
    def put_sip_setup(self, uuid: str, caller: str, exten: str, created: Optional[float] = None) -> None:
        created = time.time() if created is None else created
        dialed = exten or ""
        self._x("INSERT INTO sip_setup(uuid, caller, exten, callee, created) VALUES(?,?,?,?,?) "
                "ON CONFLICT(uuid) DO UPDATE SET caller=excluded.caller, exten=excluded.exten, "
                "callee=excluded.callee, created=excluded.created",
                (uuid, caller, dialed, dialed, created))

    def get_sip_setup(self, uuid: str, max_age: Optional[float] = 60.0) -> Optional[dict]:
        r = self._one("SELECT uuid, caller, exten, callee, created FROM sip_setup WHERE uuid=?", (uuid,))
        if not r:
            return None
        if max_age is not None and time.time() - float(r["created"]) > max_age:
            return None
        row = dict(r)
        dialed = row.get("exten") or row.get("callee") or ""
        row["exten"] = dialed
        row["callee"] = dialed
        return row

    def delete_sip_setup(self, uuid: str) -> None:
        if uuid:
            self._x("DELETE FROM sip_setup WHERE uuid=?", (uuid,))

    def purge_sip_setups(self, cutoff: float) -> int:
        """Drops setup rows posted before `cutoff`, including ones that never became a call."""
        return self._x("DELETE FROM sip_setup WHERE created < ?", (cutoff,))
