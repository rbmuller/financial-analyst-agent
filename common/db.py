"""Database connections for the two roles the application uses."""

from enum import StrEnum

import psycopg

from common.settings import Settings, get_settings


class DbRole(StrEnum):
    ADMIN = "admin"  # owns the schema: migrations, ingestion, tracing
    READONLY = "readonly"  # SELECT on fundamentals only: the agent's SQL tool


def connect(role: DbRole = DbRole.ADMIN, settings: Settings | None = None) -> psycopg.Connection:
    """Open a connection as the given role. The caller closes it (use `with`)."""
    settings = settings or get_settings()
    if role is DbRole.READONLY:
        user, password = settings.agent_ro_user, settings.agent_ro_password
    else:
        user, password = settings.postgres_user, settings.postgres_password

    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=user,
        password=password.get_secret_value(),
    )
