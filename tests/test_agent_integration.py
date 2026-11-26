"""The SQL tool and the tracer against the test database."""

import json
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from functools import partial

import psycopg
import pytest

from agent.pricing import TokenUsage
from agent.sources import SourceRegistry
from agent.tools.base import ToolInputError
from agent.tools.query_financials import QueryFinancialsTool
from agent.tracing import PostgresTracer, RunOrigin, RunStatus, Step, StepType
from common.db import DbRole, connect
from ingestion.companies import Company, upsert_company
from ingestion.xbrl.metrics import MetricValue, replace_metrics

pytestmark = pytest.mark.integration

TICKER = "ZZAGNT"
COMPANY = Company(ticker=TICKER, cik=999999996, name="Agent Test Inc.", fiscal_year_end="0531")


@pytest.fixture
def seeded() -> Iterator[None]:
    with connect(DbRole.ADMIN) as conn:
        upsert_company(conn, COMPANY)
        metric = MetricValue(
            TICKER, "revenue", 2026, "Q1", date(2025, 6, 1), date(2025, 8, 31), Decimal(1000),
            "USD", "Revenues", "9999999996-25-000001", "10-Q", date(2025, 10, 1),
        )  # fmt: skip
        replace_metrics(conn, TICKER, [metric])
        conn.commit()
        yield
        conn.execute("DELETE FROM companies WHERE ticker = %s", (TICKER,))
        conn.commit()


@pytest.fixture
def tool() -> QueryFinancialsTool:
    return QueryFinancialsTool(partial(connect, DbRole.READONLY), ["revenue"], max_rows=5)


def test_query_returns_rows_and_a_citable_source(seeded: None, tool: QueryFinancialsTool) -> None:
    sources = SourceRegistry()

    output = tool.run(
        {
            "sql": "SELECT ticker, form, filed_date, value FROM financial_metrics "
            f"WHERE ticker = '{TICKER}'"
        },
        sources,
    )

    payload = json.loads(output.content)
    assert payload["rows"] == [
        {"ticker": TICKER, "form": "10-Q", "filed_date": "2025-10-01", "value": "1000"}
    ]
    source = sources.get(payload["source_id"])
    assert source is not None
    assert source.label.endswith(f"{TICKER} 10-Q filed 2025-10-01")
    assert source.detail is not None and source.detail.startswith("SELECT ticker")


def test_row_limit_is_enforced(seeded: None, tool: QueryFinancialsTool) -> None:
    output = tool.run({"sql": "SELECT generate_series(1, 100) AS n"}, SourceRegistry())

    assert json.loads(output.content)["row_count"] == 5


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT * FROM agent_runs", "not available"),
        ("DELETE FROM companies", "Only SELECT"),
        ("SELECT no_such_column FROM companies", "query failed"),
    ],
)
def test_bad_queries_come_back_as_errors_the_model_can_fix(
    tool: QueryFinancialsTool, sql: str, message: str
) -> None:
    with pytest.raises(ToolInputError, match=message):
        tool.run({"sql": sql}, SourceRegistry())


def test_tracer_persists_run_and_steps() -> None:
    tracer = PostgresTracer(connect(DbRole.ADMIN))
    usage = TokenUsage(input_tokens=100, output_tokens=20, cache_read_tokens=50)

    run_id = tracer.start_run("a question", "test-model", RunOrigin("test", slack_channel="C1"))
    tracer.record_step(run_id, 0, Step(StepType.LLM, 12, usage=usage, cost_usd=Decimal("0.01")))
    tracer.record_step(
        run_id,
        1,
        Step(StepType.TOOL, 5, tool_name="calculator", tool_input={"values": [1, 2]}, output="x"),
    )
    tracer.finish_run(run_id, RunStatus.OK, "an answer", None, usage, Decimal("0.01"), 17)

    with connect(DbRole.ADMIN) as conn:
        run = conn.execute(
            "SELECT status, answer, input_tokens, output_tokens, cost_usd, slack_channel, "
            "finished_at IS NOT NULL FROM agent_runs WHERE id = %s",
            (run_id,),
        ).fetchone()
        steps = conn.execute(
            "SELECT step_type, tool_name, tool_input, tool_output->>'preview', input_tokens "
            "FROM agent_steps WHERE run_id = %s ORDER BY step_index",
            (run_id,),
        ).fetchall()
        assert isinstance(conn, psycopg.Connection)

    assert run == ("ok", "an answer", 150, 20, Decimal("0.010000"), "C1", True)
    assert steps == [
        ("llm", None, None, "", 150),
        ("tool", "calculator", {"values": [1, 2]}, "x", None),
    ]
