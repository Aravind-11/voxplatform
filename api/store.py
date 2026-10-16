"""SQL store: SQLite (single node, default) or PostgreSQL (shared by several API nodes).

Store(path)               -> SQLite file (or ':memory:')
Store("postgresql://...") -> PostgreSQL via psycopg 3
The SQL is written once with `?` placeholders and translated for PostgreSQL. Every read that serves a tenant
filters by tenant_id.
"""
from __future__ import annotations

import json
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
"""

# columns added after the first release: (table, column, type)
MIGRATIONS = [("api_keys", "id", "TEXT"), ("api_keys", "revoked", "{REAL}")]


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Store:
    def __init__(self, path: str = ":memory:"):
        self.lock = threading.RLock()
        self.pg = path.startswith(("postgres://", "postgresql://"))
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row
            self.db = psycopg.connect(path, autocommit=True, row_factory=dict_row)
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
            for r in self._all("SELECT key_hash FROM api_keys WHERE id IS NULL"):
                self._x("UPDATE api_keys SET id=? WHERE key_hash=?", (new_id("key"), r["key_hash"]))

    # ---- low level
    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def _run(self, sql: str, args: tuple = ()):
        if self.pg:
            cur = self.db.cursor()
            cur.execute(self._sql(sql), args)
            return cur
        cur = self.db.execute(sql, args)
        self.db.commit()
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

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # ---- tenants / keys
    def create_tenant(self, name: str, concurrency_limit: Optional[int] = None) -> str:
        tid = new_id("tnt")
        self._x("INSERT INTO tenants VALUES(?,?,?,?,?,?)",
                (tid, name, None, None, concurrency_limit, time.time()))
        return tid

    def add_key(self, tenant_id: str, key_hash: str, label: str = "") -> str:
        kid = new_id("key")
        self._x("INSERT INTO api_keys(key_hash, tenant_id, label, created, id, revoked) VALUES(?,?,?,?,?,NULL)",
                (key_hash, tenant_id, label, time.time(), kid))
        return kid

    def tenant_by_key(self, key_hash: str) -> Optional[dict]:
        return self._one("SELECT t.*, k.id AS key_id FROM tenants t JOIN api_keys k ON k.tenant_id=t.id "
                         "WHERE k.key_hash=? AND k.revoked IS NULL", (key_hash,))

    def key_id_for_hash(self, key_hash: str) -> Optional[str]:
        r = self._one("SELECT id FROM api_keys WHERE key_hash=? AND revoked IS NULL", (key_hash,))
        return r["id"] if r else None

    def list_keys(self, tenant_id: str) -> list[dict]:
        return self._all("SELECT id, label, created, revoked FROM api_keys WHERE tenant_id=? ORDER BY created",
                         (tenant_id,))

    def revoke_key(self, tenant_id: str, key_id: str) -> bool:
        return self._x("UPDATE api_keys SET revoked=? WHERE id=? AND tenant_id=? AND revoked IS NULL",
                       (time.time(), key_id, tenant_id)) > 0

    def active_key_count(self, tenant_id: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM api_keys WHERE tenant_id=? AND revoked IS NULL",
                         (tenant_id,))["n"]

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
                    metadata: Optional[dict] = None) -> dict:
        cid, token = new_id("call"), secrets.token_urlsafe(24)
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

    def list_calls(self, tenant_id: str, limit: int = 50) -> list[dict]:
        return [self._call(r) for r in self._all(
            "SELECT * FROM calls WHERE tenant_id=? ORDER BY created DESC LIMIT ?", (tenant_id, limit))]

    def update_call(self, call_id: str, **fields: Any) -> None:
        allowed = {"status", "joined", "ended", "end_reason", "metadata"}
        assert set(fields) <= allowed, f"unexpected fields {set(fields) - allowed}"
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self._x(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))

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
