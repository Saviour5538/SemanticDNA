from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool(dsn: str, minconn: int = 2, maxconn: int = 10) -> None:
    global _pool
    import json
    psycopg2.extras.register_default_jsonb(globally=True, loads=json.loads)
    _pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)


def close_pool() -> None:
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None


@contextmanager
def get_connection() -> Generator:
    assert _pool is not None, "Connection pool not initialised"
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


@contextmanager
def get_cursor(conn) -> Generator:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        yield cur
