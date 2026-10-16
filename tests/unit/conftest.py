"""Run the whole unit suite against PostgreSQL by setting VOX_TEST_DB=postgresql://user@host/db (tables are emptied
before every test). Without it the tests use in-memory SQLite."""
import os

import pytest

TABLES = ["tenants", "api_keys", "agents", "calls", "messages", "audit", "tools", "corpora", "docs"]


@pytest.fixture(autouse=True)
def _postgres_if_configured(monkeypatch):
    url = os.environ.get("VOX_TEST_DB")
    if not url:
        yield
        return
    import api.store as st
    orig = st.Store.__init__

    def init(self, path=":memory:"):
        orig(self, url if path == ":memory:" else path)
        for t in TABLES:
            self._x(f"TRUNCATE {t} RESTART IDENTITY")
    monkeypatch.setattr(st.Store, "__init__", init)
    yield
