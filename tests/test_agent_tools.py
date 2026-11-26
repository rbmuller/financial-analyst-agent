import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent.pricing import ModelPrice, Pricing, TokenUsage
from agent.sources import SourceKind, SourceRegistry, cited_ids, resolve_citations
from agent.sql_guard import UnsafeSqlError, validate_select
from agent.tools.base import ToolInputError
from agent.tools.calculator import CalculatorTool
from agent.tools.query_financials import DATA_SOURCE_NAME, _provenance_label

ALLOWED = frozenset({"companies", "financial_metrics"})
MAX_ROWS = 50


def guard(sql: str) -> str:
    return validate_select(sql, ALLOWED, MAX_ROWS)


def calculate(**arguments: object) -> float:
    output = CalculatorTool().run(dict(arguments), SourceRegistry())
    return float(json.loads(output.content)["result"])


# --- SQL guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ticker, value FROM financial_metrics WHERE metric = 'revenue'",
        "select m.value from public.financial_metrics m join companies c using (ticker)",
        "WITH recent AS (SELECT * FROM financial_metrics ORDER BY period_end DESC LIMIT 4) "
        "SELECT avg(value) FROM recent",
        "SELECT ticker FROM companies UNION SELECT ticker FROM financial_metrics",
        "SELECT ticker FROM companies; -- trailing comment",
    ],
)
def test_guard_accepts_read_only_selects_and_caps_rows(sql: str) -> None:
    guarded = guard(sql)

    assert guarded.startswith("SELECT * FROM (")
    assert guarded.endswith(f"LIMIT {MAX_ROWS}")
    assert "--" not in guarded


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM companies", "Only SELECT"),
        ("UPDATE companies SET name = 'x'", "Only SELECT"),
        ("DROP TABLE companies", "Only SELECT"),
        ("SELECT 1; DROP TABLE companies", "Exactly one"),
        ("SELECT * INTO backup FROM companies", "read-only"),
        ("WITH gone AS (DELETE FROM companies RETURNING *) SELECT * FROM gone", "read-only"),
        ("SELECT * FROM agent_runs", "not available"),
        ("SELECT * FROM pg_catalog.pg_roles", "not allowed"),
        ("SELECT * FROM other_schema.companies", "not allowed"),
        ("SELECT pg_sleep(60)", "pg_sleep"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT * FROM companies WHERE", "parse"),
        ("", "Exactly one"),
    ],
)
def test_guard_rejects_everything_else(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSqlError, match=reason):
        guard(sql)


def test_guard_error_lists_the_allowed_tables() -> None:
    with pytest.raises(UnsafeSqlError, match="companies, financial_metrics"):
        guard("SELECT * FROM filings")


# --- calculator --------------------------------------------------------------


def test_calculator_operations() -> None:
    assert calculate(operation="percent_change", values=[200, 150]) == -25.0
    assert calculate(operation="percent_change", values=[-100, -50]) == 50.0
    assert calculate(operation="cagr", values=[100, 121], periods=2) == pytest.approx(10.0)
    assert calculate(operation="average", values=[1, 2, 6]) == 3.0
    assert calculate(operation="ratio_percent", values=[4530, 11279]) == pytest.approx(40.1631)
    assert calculate(operation="difference", values=[42.2, 40.6]) == pytest.approx(-1.6)


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "sqrt", "values": [4]},
        {"operation": "percent_change", "values": [0, 10]},
        {"operation": "percent_change", "values": [1, 2, 3]},
        {"operation": "cagr", "values": [100, 121]},
        {"operation": "cagr", "values": [-100, 121], "periods": 2},
        {"operation": "ratio_percent", "values": [1, 0]},
        {"operation": "average", "values": []},
        {"operation": "average", "values": ["1", 2]},
        {"operation": "average", "values": "1,2"},
    ],
)
def test_calculator_rejects_bad_input_with_a_fixable_error(arguments: dict[str, object]) -> None:
    with pytest.raises(ToolInputError):
        CalculatorTool().run(arguments, SourceRegistry())


# --- sources and citations ---------------------------------------------------


def test_registry_assigns_stable_ids_per_kind() -> None:
    registry = SourceRegistry()

    first = registry.add(SourceKind.FILING, "chunk-10", "Nike 10-K", "https://example.test/a")
    second = registry.add(SourceKind.FILING, "chunk-11", "Nike 10-Q")
    data = registry.add(SourceKind.DATA, "SELECT 1", "data", detail="SELECT 1")
    again = registry.add(SourceKind.FILING, "chunk-10", "Nike 10-K")

    assert (first, second, data, again) == ("S1", "S2", "D1", "S1")
    source = registry.get("S1")
    assert source is not None and source.url == "https://example.test/a"


def test_cited_ids_in_order_without_duplicates() -> None:
    answer = "Margin fell [D1]. Management blamed tariffs [S2, S1] and discounts [S2][D1; S3]."

    assert cited_ids(answer) == ["D1", "S2", "S1", "S3"]
    assert cited_ids("No citations, and [1] or [Source 2] do not count.") == []


def test_resolve_citations_flags_ids_no_tool_returned() -> None:
    registry = SourceRegistry()
    registry.add(SourceKind.FILING, "chunk-1", "Nike 10-K")

    sources, unknown = resolve_citations("Real [S1], invented [S7] and [D2].", registry)

    assert [source.id for source in sources] == ["S1"]
    assert unknown == ["S7", "D2"]


def test_data_source_label_names_the_filings_when_provenance_is_selected() -> None:
    rows = [
        {"ticker": "NKE", "form": "10-Q", "filed_date": "2026-04-01", "value": 1},
        {"ticker": "NKE", "form": "10-K", "filed_date": "2026-07-15", "value": 2},
        {"ticker": "NKE", "form": "10-K", "filed_date": "2026-07-15", "value": 3},
    ]

    assert _provenance_label(rows) == (
        f"{DATA_SOURCE_NAME}: NKE 10-K filed 2026-07-15; NKE 10-Q filed 2026-04-01"
    )
    assert _provenance_label([{"ticker": "NKE", "value": 1}]) == DATA_SOURCE_NAME
    assert _provenance_label([]) == DATA_SOURCE_NAME


# --- pricing -----------------------------------------------------------------


def test_cost_counts_cached_tokens_at_their_own_rates() -> None:
    price = ModelPrice(input=Decimal(4), output=Decimal(20), cache_read=Decimal("0.2"))
    pricing = Pricing({"model-a": price}, cache_write_multiplier=Decimal("1.25"))
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )

    assert pricing.cost_usd("model-a", usage) == Decimal("11.2")  # 4 + 2 + 0.2 + 5
    assert pricing.cost_usd("unknown-model", usage) is None
    assert (usage + usage).total_input_tokens == 6_000_000


def test_pricing_file_loads() -> None:
    pricing = Pricing.load(Path("config"))

    assert pricing.cost_usd("claude-opus-5-5", TokenUsage(output_tokens=1_000_000)) == Decimal(20)
