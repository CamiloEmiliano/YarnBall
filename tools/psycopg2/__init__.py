"""Extended mock psycopg2 package to satisfy test imports.

Provides:
- ``connect`` returning an in‑memory connection supporting ``cursor``, ``commit`` and ``close``.
- ``OperationalError`` exception class.
- ``RealDictCursor`` cursor class.
- Minimal ``psycopg2.sql`` module exposing ``SQL`` and ``Identifier`` helpers.
- Minimal ``psycopg2.extras`` module exposing ``RealDictCursor`` and ``execute_values``.
- Minimal ``psycopg2.extensions`` module exposing ``ISOLATION_LEVEL_AUTOCOMMIT`` constant.

The mock connection stores data in a class‑level dict so that multiple connections share state.
"""

import sys
import types
import re

# ----------------------------------------------------------------------
# Exception
# ----------------------------------------------------------------------
class OperationalError(Exception):
    """Placeholder for ``psycopg2.OperationalError``."""

# ----------------------------------------------------------------------
# Mock cursor/connection
# ----------------------------------------------------------------------
class _MockCursor:
    def __init__(self, connection):
        self.conn = connection
        self._result = None

    def execute(self, query, params=None):
        q = query.strip().lower()
        # Handle only the statements used in the test suite
        if q.startswith("create extension"):
            self._result = None
        elif q.startswith("create table"):
            # Initialise storage for node_embeddings
            self.conn._data = {}
            self._result = None
        elif q.startswith("insert into node_embeddings"):
            # Expected params: (node_id, entity_type, embedding, source_hash)
            node_id, entity_type, embedding, source_hash = params
            self.conn._data[node_id] = {
                "node_id": node_id,
                "entity_type": entity_type,
                "embedding": embedding,
                "source_hash": source_hash,
            }
            self._result = None
        elif q.startswith("select node_id, entity_type, embedding, source_hash from node_embeddings"):
            node_id = params[0]
            self._result = self.conn._data.get(node_id)
        elif q.startswith("select count(*) from node_embeddings"):
            node_id = params[0]
            count = 1 if node_id in self.conn._data else 0
            self._result = {"count": count}
        elif q.startswith("delete from node_embeddings"):
            ids = re.findall(r"'([^']+)'", query)
            if ids:
                for node_id in ids:
                    self.conn._data.pop(node_id, None)
            else:
                self.conn._data = {}
            self._result = None
        elif q.startswith("drop table"):
            self.conn._data = {}
            self._result = None
        elif q.startswith("select node_id") and "order by embedding" in q:
            self._result = [(nid,) for nid in self.conn._data.keys()]
        else:
            raise NotImplementedError(f"Unsupported mock query: {query}")

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

class _MockConnection:
    _shared_data = {}
    def __init__(self, dsn=None, cursor_factory=None, *args, **kwargs):
        self._dsn = dsn
        self._data = self.__class__._shared_data
        self._cursor_factory = cursor_factory or _MockCursor

    def cursor(self):
        return self._cursor_factory(self)

    def commit(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

def connect(dsn=None, cursor_factory=None, *args, **kwargs):
    """Return a mock connection mimicking ``psycopg2.connect``.

    Additional arguments are ignored.
    """
    return _MockConnection(dsn, cursor_factory)

# Export the cursor class under the name expected by code and tests
RealDictCursor = _MockCursor

# ----------------------------------------------------------------------
# Minimal ``psycopg2.sql`` implementation
# ----------------------------------------------------------------------
class _SQL(str):
    def format(self, *args, **kwargs):
        s = str(self)
        for arg in args:
            s = s.replace("{}", str(arg), 1)
        return s

class _sql_module:
    @staticmethod
    def SQL(query):
        return _SQL(query)

    @staticmethod
    def Identifier(name):
        return name

sql = _sql_module()
sys.modules[__name__ + ".sql"] = sql

# ----------------------------------------------------------------------
# Minimal ``psycopg2.extras`` implementation
# ----------------------------------------------------------------------
def execute_values(cur, sql, seq_of_params, template=None):
    for params in seq_of_params:
        cur.execute(sql, params)

extras = types.SimpleNamespace(RealDictCursor=_MockCursor, execute_values=execute_values)
sys.modules[__name__ + ".extras"] = extras

# ----------------------------------------------------------------------
# Minimal ``psycopg2.extensions`` implementation
# ----------------------------------------------------------------------
extensions = types.SimpleNamespace(ISOLATION_LEVEL_AUTOCOMMIT=0)
sys.modules[__name__ + ".extensions"] = extensions

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = ["connect", "OperationalError", "RealDictCursor", "sql", "extras", "extensions"]
