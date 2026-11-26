"""Checks against the real Postgres container: schema, indexes and the read-only role."""

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import errors, sql

from common.db import DbRole, connect
from db.migrate import apply_migrations

pytestmark = pytest.mark.integration

INSERT_COMPANY = "INSERT INTO companies (ticker, cik, name) VALUES ('ZZZZ', 0, 'Should not exist')"


@pytest.fixture
def admin() -> Iterator[psycopg.Connection]:
    with connect(DbRole.ADMIN) as conn:
        yield conn


@pytest.fixture
def readonly() -> Iterator[psycopg.Connection]:
    with connect(DbRole.READONLY) as conn:
        yield conn


def test_migrations_are_idempotent(admin: psycopg.Connection) -> None:
    assert apply_migrations(admin) == []


def test_pgvector_is_installed(admin: psycopg.Connection) -> None:
    row = admin.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None


def test_cosine_distance_operator_works(admin: psycopg.Connection) -> None:
    row = admin.execute("SELECT '[1,0]'::vector <=> '[0,1]'::vector").fetchone()
    assert row is not None
    assert row[0] == pytest.approx(1.0)


def test_chunks_has_hnsw_and_gin_indexes(admin: psycopg.Connection) -> None:
    rows = admin.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'").fetchall()
    index_definitions = " ".join(row[0] for row in rows)

    assert "USING hnsw" in index_definitions
    assert "USING gin" in index_definitions


def test_readonly_can_select_fundamentals(readonly: psycopg.Connection) -> None:
    readonly.execute("SELECT count(*) FROM companies").fetchone()
    readonly.execute("SELECT count(*) FROM financial_facts").fetchone()


def test_readonly_transactions_are_read_only_by_default(readonly: psycopg.Connection) -> None:
    with pytest.raises(errors.ReadOnlySqlTransaction):
        readonly.execute(INSERT_COMPANY)


def test_readonly_cannot_write_even_if_it_disables_read_only_mode(
    readonly: psycopg.Connection,
) -> None:
    # A session can turn the read-only default off; privileges must still block it.
    readonly.autocommit = True
    readonly.execute("SET default_transaction_read_only = off")

    with pytest.raises(errors.InsufficientPrivilege):
        readonly.execute(INSERT_COMPANY)


@pytest.mark.parametrize("table", ["agent_runs", "agent_steps", "filings", "chunks"])
def test_readonly_cannot_read_outside_the_allow_list(
    readonly: psycopg.Connection, table: str
) -> None:
    with pytest.raises(errors.InsufficientPrivilege):
        readonly.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(table)))
