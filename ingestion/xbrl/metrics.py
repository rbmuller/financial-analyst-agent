"""Curated metrics: one value per company, metric and fiscal period.

Raw XBRL facts are not directly usable for "the last 3 quarters" questions:

* a 10-Q reports the quarter and the year-to-date amount under the same tag;
* every filing repeats prior periods as comparatives, sometimes restated;
* the `fy`/`fp` fields describe the *filing*, not the period of each fact;
* the 10-K reports the full year, so the fourth quarter is usually missing;
* companies use different tags for the same line item.

`build_metrics` resolves all of that with explicit, testable rules.
"""

from collections import defaultdict
from dataclasses import astuple, dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg
import yaml

from ingestion.xbrl.facts import FinancialFact

METRICS_FILE = "metrics.yaml"
US_GAAP = "us-gaap"

FULL_YEAR = "FY"
REPORTED_QUARTERS = ("Q1", "Q2", "Q3")
FOURTH_QUARTER = "Q4"

# Only periodic reports are sources for metrics. Other forms repeat some figures
# (a proxy statement restates net income) and, being filed later, would otherwise
# take over as the cited source.
PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})

# Fiscal calendars are not exact (52/53-week years, 13-week quarters).
QUARTER_LENGTH_DAYS = range(80, 101)
YEAR_LENGTH_DAYS = range(350, 381)

type PeriodLabel = tuple[int, str]  # (fiscal_year, fiscal_period)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    additive: bool
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class MetricValue:
    ticker: str
    metric: str
    fiscal_year: int
    fiscal_period: str
    period_start: date
    period_end: date
    value: Decimal
    unit: str
    concept: str
    accession_number: str
    form: str
    filed_date: date
    is_derived: bool = False


def load_metric_definitions(config_dir: Path) -> list[MetricDefinition]:
    payload = yaml.safe_load((config_dir / METRICS_FILE).read_text())
    return [
        MetricDefinition(
            name=item["name"],
            unit=item["unit"],
            additive=bool(item["additive"]),
            concepts=tuple(item["concepts"]),
        )
        for item in payload["metrics"]
    ]


def _period_label(observations: list[FinancialFact]) -> PeriodLabel | None:
    """Fiscal label of a period, taken from the first filing that reported it.

    In that filing the period is the current one, so the filing's fy/fp describe
    it. Later filings repeat the period as a comparative under their own fy/fp.
    """
    first = min(observations, key=lambda fact: fact.filed_date)
    if first.fiscal_year is None or first.fiscal_period is None or first.period_start is None:
        return None

    length_days = (first.period_end - first.period_start).days + 1
    if length_days in YEAR_LENGTH_DAYS and first.fiscal_period == FULL_YEAR:
        return first.fiscal_year, FULL_YEAR
    if length_days in QUARTER_LENGTH_DAYS:
        if first.fiscal_period in REPORTED_QUARTERS:
            return first.fiscal_year, first.fiscal_period
        if first.fiscal_period == FULL_YEAR:
            # A quarter first reported by a 10-K is the fourth quarter.
            return first.fiscal_year, FOURTH_QUARTER
    return None  # year-to-date and other odd-length periods


def _values_for_concept(
    ticker: str, definition: MetricDefinition, facts: list[FinancialFact]
) -> dict[PeriodLabel, MetricValue]:
    by_period: dict[tuple[date, date], list[FinancialFact]] = defaultdict(list)
    for fact in facts:
        if (
            fact.unit == definition.unit
            and fact.period_start is not None
            and fact.form in PERIODIC_FORMS
        ):
            by_period[(fact.period_start, fact.period_end)].append(fact)

    values: dict[PeriodLabel, MetricValue] = {}
    for (period_start, period_end), observations in by_period.items():
        label = _period_label(observations)
        if label is None:
            continue
        # The most recently filed number wins, so restatements are picked up.
        latest = max(observations, key=lambda fact: fact.filed_date)
        candidate = MetricValue(
            ticker=ticker,
            metric=definition.name,
            fiscal_year=label[0],
            fiscal_period=label[1],
            period_start=period_start,
            period_end=period_end,
            value=latest.value,
            unit=latest.unit,
            concept=latest.concept,
            accession_number=latest.accession_number,
            form=latest.form,
            filed_date=latest.filed_date,
        )
        # Comparatives of the oldest filings have no original filing in XBRL and
        # inherit that filing's label; the real owner of a label ends last.
        current = values.get(label)
        if current is None or candidate.period_end > current.period_end:
            values[label] = candidate
    return values


def _derive_fourth_quarters(values: dict[PeriodLabel, MetricValue]) -> list[MetricValue]:
    derived: list[MetricValue] = []
    for fiscal_year in {year for year, _ in values}:
        full_year = values.get((fiscal_year, FULL_YEAR))
        quarters = [values.get((fiscal_year, quarter)) for quarter in REPORTED_QUARTERS]
        if full_year is None or (fiscal_year, FOURTH_QUARTER) in values:
            continue
        reported = [quarter for quarter in quarters if quarter is not None]
        if len(reported) != len(REPORTED_QUARTERS):
            continue
        derived.append(
            MetricValue(
                ticker=full_year.ticker,
                metric=full_year.metric,
                fiscal_year=fiscal_year,
                fiscal_period=FOURTH_QUARTER,
                period_start=reported[-1].period_end + timedelta(days=1),
                period_end=full_year.period_end,
                value=full_year.value - sum((quarter.value for quarter in reported), Decimal(0)),
                unit=full_year.unit,
                concept=full_year.concept,
                accession_number=full_year.accession_number,
                form=full_year.form,
                filed_date=full_year.filed_date,
                is_derived=True,
            )
        )
    return derived


def build_metrics(
    ticker: str, facts: list[FinancialFact], definitions: list[MetricDefinition]
) -> list[MetricValue]:
    facts_by_concept: dict[str, list[FinancialFact]] = defaultdict(list)
    for fact in facts:
        if fact.taxonomy == US_GAAP:
            facts_by_concept[fact.concept].append(fact)

    metrics: list[MetricValue] = []
    for definition in definitions:
        merged: dict[PeriodLabel, MetricValue] = {}
        for concept in definition.concepts:  # priority order: first concept wins
            concept_values = _values_for_concept(ticker, definition, facts_by_concept[concept])
            for label, value in concept_values.items():
                merged.setdefault(label, value)
        if definition.additive:
            for value in _derive_fourth_quarters(merged):
                merged[(value.fiscal_year, value.fiscal_period)] = value
        metrics.extend(merged.values())

    return sorted(metrics, key=lambda m: (m.metric, m.fiscal_year, m.fiscal_period))


INSERT_METRIC = """
    INSERT INTO financial_metrics (
        ticker, metric, fiscal_year, fiscal_period, period_start, period_end, value,
        unit, concept, accession_number, form, filed_date, is_derived
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def replace_metrics(conn: psycopg.Connection, ticker: str, metrics: list[MetricValue]) -> int:
    """Rebuild a company's metrics atomically (derived data: delete + insert)."""
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute("DELETE FROM financial_metrics WHERE ticker = %s", (ticker,))
        cursor.executemany(INSERT_METRIC, [astuple(metric) for metric in metrics])
    return len(metrics)
