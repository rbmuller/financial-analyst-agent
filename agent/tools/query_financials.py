"""The SQL tool: read-only queries over the XBRL fundamentals."""

import json
from collections.abc import Callable
from typing import Any

import psycopg

from agent.sources import SourceKind, SourceRegistry
from agent.sql_guard import UnsafeSqlError, validate_select
from agent.tools.base import ToolInputError, ToolOutput

ALLOWED_TABLES = frozenset({"companies", "financial_metrics", "financial_facts"})
PROVENANCE_COLUMNS = ("ticker", "form", "filed_date")
MAX_PROVENANCE_ITEMS = 6
DATA_SOURCE_NAME = "SEC XBRL financial data"

DESCRIPTION_TEMPLATE = """\
Run one read-only PostgreSQL SELECT over financial data reported to the SEC (XBRL). \
Call this for any question that needs a figure: revenue, costs, profit, margins, \
earnings per share, growth, comparisons across periods or companies. It returns rows \
plus a source_id to cite.

Tables:

financial_metrics: one row per company, metric and fiscal period. Use this first.
  ticker, metric, fiscal_year (int), fiscal_period ('Q1','Q2','Q3','Q4','FY'),
  period_start (date), period_end (date), value (numeric), unit ('USD' or 'USD/shares'),
  concept (XBRL tag), accession_number, form ('10-K'/'10-Q'), filed_date (date),
  is_derived (bool: true when Q4 was computed as FY minus Q1..Q3)
  metric is one of: {metrics}
  Quarterly values are for the single quarter, not year to date. Per-share metrics have
  no Q4 row. A company may lack a metric it does not report.

companies: ticker, cik, name, fiscal_year_end ('MMDD').
  Fiscal years differ by company (for example a fiscal year can end in May), so
  fiscal_year 2026 is not the same calendar period for every company. Order by
  period_end to find the most recent periods, and say which period you used.

financial_facts: every raw XBRL fact, for line items not in financial_metrics.
  ticker, taxonomy, concept, unit, period_start, period_end, value, accession_number,
  fiscal_year, fiscal_period, form, filed_date, frame
  Raw facts mix quarterly and year-to-date periods and repeat across filings, so check
  period_start/period_end. Find concepts with: SELECT DISTINCT concept ... ILIKE '%x%'.

Always select ticker, form and filed_date together with the values: they identify the
filing each number came from, which is what the citation shows. At most {max_rows} rows
are returned; aggregate or filter in SQL rather than fetching everything. Margins and
ratios can be computed in SQL or with the calculator tool."""


class QueryFinancialsTool:
    name = "query_financials"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "A single PostgreSQL SELECT statement."},
        },
        "required": ["sql"],
    }

    def __init__(
        self, connect_readonly: Callable[[], psycopg.Connection], metrics: list[str], max_rows: int
    ) -> None:
        self._connect = connect_readonly
        self._max_rows = max_rows
        self.description = DESCRIPTION_TEMPLATE.format(
            metrics=", ".join(sorted(metrics)), max_rows=max_rows
        )

    def run(self, arguments: dict[str, Any], sources: SourceRegistry) -> ToolOutput:
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ToolInputError("'sql' must be a non-empty string.")
        try:
            guarded_sql = validate_select(sql, ALLOWED_TABLES, self._max_rows)
        except UnsafeSqlError as error:
            raise ToolInputError(str(error)) from error

        try:
            with self._connect() as conn, conn.cursor() as cursor:
                cursor.execute(guarded_sql)
                columns = [column.name for column in cursor.description or []]
                rows = cursor.fetchall()
        except psycopg.Error as error:
            # Database errors (unknown column, timeout) are feedback the model can use.
            raise ToolInputError(f"The query failed: {error}".strip()) from error

        records = [dict(zip(columns, row, strict=True)) for row in rows]
        source_id = sources.add(
            SourceKind.DATA, sql, label=_provenance_label(records), detail=" ".join(sql.split())
        )
        return ToolOutput(
            json.dumps(
                {"source_id": source_id, "row_count": len(records), "rows": records}, default=str
            )
        )


def _provenance_label(records: list[dict[str, Any]]) -> str:
    """Name the filings behind a result when the query selected the provenance columns."""
    if not records or not all(column in records[0] for column in PROVENANCE_COLUMNS):
        return DATA_SOURCE_NAME
    filings = sorted(
        {(str(r["ticker"]), str(r["form"]), str(r["filed_date"])) for r in records},
        key=lambda filing: (filing[0], filing[2]),
        reverse=True,
    )
    shown = [f"{ticker} {form} filed {filed}" for ticker, form, filed in filings]
    extra = len(shown) - MAX_PROVENANCE_ITEMS
    suffix = f" and {extra} more" if extra > 0 else ""
    return f"{DATA_SOURCE_NAME}: {'; '.join(shown[:MAX_PROVENANCE_ITEMS])}{suffix}"
