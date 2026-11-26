"""Minimal, idempotent SQL migration runner.

Applies db/migrations/*.sql in filename order, once each, and then makes sure
the read-only login used by the agent exists with the password from .env.

Usage: python -m db.migrate
"""

from pathlib import Path

import psycopg
from psycopg import sql

from common.db import DbRole, connect
from common.settings import Settings, get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
READONLY_GROUP = "agent_readonly"  # created by 005_readonly_role.sql
READONLY_STATEMENT_TIMEOUT = "5s"


def list_migrations(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def pending_migrations(migrations: list[Path], applied: set[str]) -> list[Path]:
    return [path for path in migrations if path.name not in applied]


def _applied_migrations(conn: psycopg.Connection) -> set[str]:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename    TEXT PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def apply_migrations(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations, one transaction each. Returns the applied filenames."""
    pending = pending_migrations(list_migrations(directory), _applied_migrations(conn))
    conn.commit()

    for path in pending:
        with conn.transaction():
            # The file content is trusted, version-controlled SQL (not user input).
            conn.execute(sql.SQL(path.read_text()))
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
    return [path.name for path in pending]


def ensure_readonly_login(conn: psycopg.Connection, settings: Settings) -> None:
    """Create or update the agent's login role and lock it down.

    Defence in depth for the SQL tool: the role can only SELECT the allow-listed
    tables (via the group), its transactions default to read-only, and runaway
    queries are cancelled by a statement timeout.
    """
    user = sql.Identifier(settings.agent_ro_user)
    password = sql.Literal(settings.agent_ro_password.get_secret_value())

    exists = conn.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (settings.agent_ro_user,)
    ).fetchone()
    set_login = sql.SQL(
        "ALTER ROLE {} WITH LOGIN PASSWORD {}" if exists else "CREATE ROLE {} LOGIN PASSWORD {}"
    )

    with conn.transaction():
        conn.execute(set_login.format(user, password))
        conn.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(READONLY_GROUP), user))
        conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(user))
        conn.execute(
            sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                user, sql.Literal(READONLY_STATEMENT_TIMEOUT)
            )
        )


def main() -> None:
    settings = get_settings()
    with connect(DbRole.ADMIN, settings) as conn:
        applied = apply_migrations(conn)
        ensure_readonly_login(conn, settings)

    if applied:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("Schema already up to date.")
    print(f"Read-only login '{settings.agent_ro_user}' is ready.")


if __name__ == "__main__":
    main()
