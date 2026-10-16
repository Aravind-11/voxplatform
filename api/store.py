"""SQLite-backed store (single node). Every read that serves a tenant filters by tenant_id.

The interface is deliberately small so a Postgres implementation can replace it; only SQLite
is implemented and tested here.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants(
  id TEXT PRIMARY KEY, name TEXT, webhook_url TEXT, webhook_secret TEXT,
  concurrency_limit INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS api_keys(
  key_hash TEXT PRIMARY KEY, tenant_id TEXT, label TEXT, created REAL);
CREATE TABLE IF NOT EXISTS agents(
  id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, config TEXT, created REAL);
CREATE TABLE IF NOT EXISTS calls(
  id TEXT PRIMARY KEY, tenant_id TEXT, agent_id TEXT, medium TEXT, status TEXT, config TEXT,
  join_token TEXT, metadata TEXT, created REAL, joined REAL, ended REAL, end_reason TEXT);
CREATE INDEX IF NOT EXISTS calls_tenant ON calls(tenant_id, created);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT, role TEXT, text TEXT, t REAL);
CREATE INDEX IF NOT EXISTS messages_call ON messages(call_id, id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db.executescript(SCHEMA)

    def _x(self, sql: str, args: tuple = ()) -> None:
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    def _one(self, sql: str, args: tuple = ()) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # ---- tenants / keys
    def create_tenant(self, name: str, concurrency_limit: Optional[int] = None) -> str:
        tid = new_id("tnt")
        self._x("INSERT INTO tenants VALUES(?,?,?,?,?,?)",
                (tid, name, None, None, concurrency_limit, time.time()))
        return tid

    def add_key(self, tenant_id: str, key_hash: str, label: str = "") -> None:
        self._x("INSERT INTO api_keys VALUES(?,?,?,?)", (key_hash, tenant_id, label, time.time()))

    def tenant_by_key(self, key_hash: str) -> Optional[dict]:
        r = self._one("SELECT t.* FROM tenants t JOIN api_keys k ON k.tenant_id=t.id WHERE k.key_hash=?",
                      (key_hash,))
        return dict(r) if r else None

    def get_tenant(self, tenant_id: str) -> Optional[dict]:
        r = self._one("SELECT * FROM tenants WHERE id=?", (tenant_id,))
        return dict(r) if r else None

    def set_webhook(self, tenant_id: str, url: str, secret: str) -> None:
        self._x("UPDATE tenants SET webhook_url=?, webhook_secret=? WHERE id=?", (url, secret, tenant_id))

    # ---- agents
    def create_agent(self, tenant_id: str, name: str, config: dict) -> dict:
        aid = new_id("agt")
        self._x("INSERT INTO agents VALUES(?,?,?,?,?)",
                (aid, tenant_id, name, json.dumps(config), time.time()))
        return self.get_agent(tenant_id, aid)  # type: ignore[return-value]

    @staticmethod
    def _agent(r: sqlite3.Row) -> dict:
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

    def list_agents(self, tenant_id: str) -> list[dict]:
        return [self._agent(r) for r in
                self._all("SELECT * FROM agents WHERE tenant_id=? ORDER BY created", (tenant_id,))]

    # ---- calls
    @staticmethod
    def _call(r: sqlite3.Row) -> dict:
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
        allowed = {"status", "joined", "ended", "end_reason"}
        assert set(fields) <= allowed, f"unexpected fields {set(fields) - allowed}"
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self._x(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))

    # ---- messages
    def add_message(self, call_id: str, role: str, text: str, t: float) -> None:
        self._x("INSERT INTO messages(call_id, role, text, t) VALUES(?,?,?,?)", (call_id, role, text, t))

    def messages(self, call_id: str) -> list[dict]:
        return [dict(r) for r in self._all(
            "SELECT role, text, t FROM messages WHERE call_id=? ORDER BY id", (call_id,))]
