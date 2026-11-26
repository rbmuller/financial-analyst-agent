"""Validate SQL written by the LLM before it reaches the database.

This is the middle layer of three. The database role can only SELECT an
allow-list of tables, and its transactions are read-only with a timeout; this
check rejects anything else earlier and with an error message the model can
act on (a database "permission denied" teaches it much less).
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

DIALECT = "postgres"
PUBLIC_SCHEMA = "public"
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Into,
    exp.Lock,
)
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")
FORBIDDEN_FUNCTIONS = frozenset({"set_config", "current_setting", "query_to_xml"})


class UnsafeSqlError(ValueError):
    """The query is not a single read-only SELECT over the allowed tables."""


def _check_tables(statement: exp.Expression, allowed_tables: frozenset[str]) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if table.catalog or (table.db and table.db.lower() != PUBLIC_SCHEMA):
            raise UnsafeSqlError(f"Schema-qualified table '{table.sql()}' is not allowed.")
        if name not in allowed_tables and name not in cte_names:
            raise UnsafeSqlError(
                f"Table '{name}' is not available. Allowed tables: "
                f"{', '.join(sorted(allowed_tables))}."
            )


def _check_functions(statement: exp.Expression) -> None:
    # Functions sqlglot has no dedicated class for (which includes every
    # administrative one) are parsed as Anonymous.
    for function in statement.find_all(exp.Anonymous):
        name = function.name.lower()
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise UnsafeSqlError(f"Function '{name}' is not allowed.")


def validate_select(sql: str, allowed_tables: frozenset[str], max_rows: int) -> str:
    """Return a safe, row-limited version of `sql`, or raise UnsafeSqlError."""
    try:
        parsed = sqlglot.parse(sql, read=DIALECT)
        # A trailing ";" or comment parses as an empty statement, which is harmless.
        statements = [s for s in parsed if s is not None and not isinstance(s, exp.Semicolon)]
    except SqlglotError as error:
        raise UnsafeSqlError(f"Could not parse the SQL: {error}") from error

    if len(statements) != 1:
        raise UnsafeSqlError("Exactly one SQL statement is allowed.")
    statement = statements[0]
    if not isinstance(statement, exp.Select | exp.SetOperation):
        raise UnsafeSqlError("Only SELECT statements are allowed.")
    if any(statement.find(node) for node in FORBIDDEN_NODES):
        raise UnsafeSqlError("Only read-only SELECT statements are allowed.")

    _check_tables(statement, allowed_tables)
    _check_functions(statement)

    # Re-rendering from the parsed tree drops comments and trailing statements;
    # the outer query caps the rows whatever the inner query says.
    return f"SELECT * FROM ({statement.sql(dialect=DIALECT)}) AS guarded LIMIT {int(max_rows)}"
