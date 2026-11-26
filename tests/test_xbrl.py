from datetime import date
from decimal import Decimal
from typing import Any

from ingestion.xbrl.facts import FinancialFact, flatten_company_facts
from ingestion.xbrl.metrics import MetricDefinition, MetricValue, build_metrics

TICKER = "TEST"
REVENUE = MetricDefinition("revenue", "USD", additive=True, concepts=("NewRevenueTag", "Revenues"))
EPS = MetricDefinition("eps", "USD/shares", additive=False, concepts=("EarningsPerShareDiluted",))


def fact(
    start: str,
    end: str,
    value: int | str,
    *,
    fy: int,
    fp: str,
    filed: str,
    concept: str = "Revenues",
    unit: str = "USD",
) -> FinancialFact:
    return FinancialFact(
        ticker=TICKER,
        taxonomy="us-gaap",
        concept=concept,
        unit=unit,
        period_start=date.fromisoformat(start),
        period_end=date.fromisoformat(end),
        value=Decimal(value),
        accession_number=f"acc-{filed}",
        fiscal_year=fy,
        fiscal_period=fp,
        form="10-K" if fp == "FY" else "10-Q",
        filed_date=date.fromisoformat(filed),
        frame=None,
    )


def by_label(metrics: list[MetricValue]) -> dict[tuple[int, str], MetricValue]:
    return {(m.fiscal_year, m.fiscal_period): m for m in metrics}


# A fiscal year that ends in May, like Nike's: FY2024 = Jun 2023 .. May 2024.
FY2024_FACTS = [
    fact("2023-06-01", "2023-08-31", 100, fy=2024, fp="Q1", filed="2023-10-01"),
    fact("2023-09-01", "2023-11-30", 110, fy=2024, fp="Q2", filed="2024-01-01"),
    fact("2023-06-01", "2023-11-30", 210, fy=2024, fp="Q2", filed="2024-01-01"),  # year to date
    fact("2023-12-01", "2024-02-29", 120, fy=2024, fp="Q3", filed="2024-04-01"),
    fact("2023-06-01", "2024-02-29", 330, fy=2024, fp="Q3", filed="2024-04-01"),  # year to date
    fact("2023-06-01", "2024-05-31", 470, fy=2024, fp="FY", filed="2024-07-20"),
]


def test_flatten_company_facts_reads_nested_payload_and_dedupes() -> None:
    row: dict[str, Any] = {
        "start": "2023-06-01",
        "end": "2023-08-31",
        "val": 12939000000,
        "accn": "0000000000-23-000001",
        "fy": 2024,
        "fp": "Q1",
        "form": "10-Q",
        "filed": "2023-10-05",
        "frame": "CY2023Q3",
    }
    instant = {k: v for k, v in row.items() if k not in ("start", "frame")}
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [row, dict(row)]}},
                "Assets": {"units": {"USD": [instant]}},
            }
        }
    }

    facts = {f.concept: f for f in flatten_company_facts(TICKER, payload)}

    assert len(facts) == 2
    assert facts["Revenues"].value == Decimal("12939000000")
    assert facts["Revenues"].period_start == date(2023, 6, 1)
    assert facts["Revenues"].frame == "CY2023Q3"
    assert facts["Assets"].period_start is None


def test_quarters_ignore_year_to_date_and_fourth_quarter_is_derived() -> None:
    metrics = by_label(build_metrics(TICKER, FY2024_FACTS, [REVENUE]))

    assert {label: m.value for label, m in metrics.items()} == {
        (2024, "Q1"): 100,
        (2024, "Q2"): 110,
        (2024, "Q3"): 120,
        (2024, "Q4"): 140,
        (2024, "FY"): 470,
    }
    fourth = metrics[(2024, "Q4")]
    assert fourth.is_derived
    assert (fourth.period_start, fourth.period_end) == (date(2024, 3, 1), date(2024, 5, 31))
    assert not metrics[(2024, "Q3")].is_derived


def test_fourth_quarter_is_not_derived_when_a_quarter_is_missing() -> None:
    facts = [f for f in FY2024_FACTS if f.fiscal_period != "Q2"]

    assert (2024, "Q4") not in by_label(build_metrics(TICKER, facts, [REVENUE]))


def test_comparative_keeps_original_label_and_restated_value_wins() -> None:
    # The FY2025 Q1 filing repeats FY2024 Q1 as a comparative, restated to 105.
    facts = [
        *FY2024_FACTS,
        fact("2024-06-01", "2024-08-31", 130, fy=2025, fp="Q1", filed="2024-10-01"),
        fact("2023-06-01", "2023-08-31", 105, fy=2025, fp="Q1", filed="2024-10-01"),
    ]

    metrics = by_label(build_metrics(TICKER, facts, [REVENUE]))

    assert metrics[(2024, "Q1")].value == 105
    assert metrics[(2024, "Q1")].filed_date == date(2024, 10, 1)
    assert metrics[(2025, "Q1")].value == 130


def test_label_collision_keeps_the_latest_period() -> None:
    # Oldest filing in the history: its comparative has no original filing.
    facts = [
        fact("2023-06-01", "2023-08-31", 100, fy=2024, fp="Q1", filed="2023-10-01"),
        fact("2022-06-01", "2022-08-31", 90, fy=2024, fp="Q1", filed="2023-10-01"),
    ]

    metrics = by_label(build_metrics(TICKER, facts, [REVENUE]))

    assert metrics[(2024, "Q1")].value == 100


def test_first_concept_wins_and_later_concepts_fill_gaps() -> None:
    facts = [
        fact("2023-06-01", "2023-08-31", 100, fy=2024, fp="Q1", filed="2023-10-01"),
        fact("2023-09-01", "2023-11-30", 110, fy=2024, fp="Q2", filed="2024-01-01"),
        fact(
            "2023-09-01", "2023-11-30", 111, fy=2024, fp="Q2", filed="2024-01-01",
            concept="NewRevenueTag",
        ),
    ]  # fmt: skip

    metrics = by_label(build_metrics(TICKER, facts, [REVENUE]))

    assert {label: (m.value, m.concept) for label, m in metrics.items()} == {
        (2024, "Q1"): (Decimal(100), "Revenues"),
        (2024, "Q2"): (Decimal(111), "NewRevenueTag"),
    }


def test_non_additive_metric_never_derives_fourth_quarter() -> None:
    facts = [
        fact(f.period_start.isoformat(), f.period_end.isoformat(), "1.5", fy=2024,  # type: ignore[union-attr]
             fp=f.fiscal_period or "", filed=f.filed_date.isoformat(),
             concept="EarningsPerShareDiluted", unit="USD/shares")
        for f in FY2024_FACTS
    ]  # fmt: skip

    labels = by_label(build_metrics(TICKER, facts, [EPS]))

    assert (2024, "FY") in labels
    assert (2024, "Q4") not in labels


def test_other_units_and_taxonomies_are_ignored() -> None:
    facts = [
        fact("2023-06-01", "2023-08-31", 100, fy=2024, fp="Q1", filed="2023-10-01", unit="EUR")
    ]

    assert build_metrics(TICKER, facts, [REVENUE]) == []


def test_non_periodic_forms_are_not_sources() -> None:
    proxy = fact("2023-06-01", "2023-08-31", 999, fy=2025, fp="Q1", filed="2024-11-01")
    facts = [*FY2024_FACTS, FinancialFact(**{**proxy.__dict__, "form": "DEF 14A"})]

    metrics = by_label(build_metrics(TICKER, facts, [REVENUE]))

    assert (metrics[(2024, "Q1")].value, metrics[(2024, "Q1")].form) == (Decimal(100), "10-Q")
