"""Integration tests run against their own database, never the development one.

Functions under test commit and scan whole tables (indexing, search), so sharing
a database with real ingested data would let a test modify it.
"""

import os
from collections.abc import Iterator

import pytest
from psycopg import sql

from common.db import DbRole, connect
from common.settings import get_settings
from db.migrate import apply_migrations, ensure_readonly_login

INTEGRATION_MARKER = "integration"
TEST_DATABASE_SUFFIX = "_test"
DATABASE_ENV_VAR = "POSTGRES_DB"
DATA_TABLES = ("companies", "agent_runs")  # everything else cascades from these


@pytest.fixture(scope="session", autouse=True)
def test_database(request: pytest.FixtureRequest) -> Iterator[None]:
    """Create and migrate `<POSTGRES_DB>_test`, and point the settings at it."""
    if not any(item.get_closest_marker(INTEGRATION_MARKER) for item in request.session.items):
        yield
        return

    development = get_settings()
    test_name = development.postgres_db + TEST_DATABASE_SUFFIX
    with connect(DbRole.ADMIN, development) as conn:
        conn.autocommit = True
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_name,))
        if exists.fetchone() is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_name)))

    original = os.environ.get(DATABASE_ENV_VAR)
    os.environ[DATABASE_ENV_VAR] = test_name
    get_settings.cache_clear()
    settings = get_settings()
    with connect(DbRole.ADMIN, settings) as conn:
        apply_migrations(conn)
        ensure_readonly_login(conn, settings)
        conn.execute(
            sql.SQL("TRUNCATE {} CASCADE").format(
                sql.SQL(", ").join(sql.Identifier(table) for table in DATA_TABLES)
            )
        )

    yield

    if original is None:
        os.environ.pop(DATABASE_ENV_VAR, None)
    else:
        os.environ[DATABASE_ENV_VAR] = original
    get_settings.cache_clear()
