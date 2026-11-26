"""Raw XBRL facts: extract from the Company Facts API, flatten, upsert.

Payload shape (one JSON per company):
    facts -> taxonomy -> concept -> units -> unit -> [observation, ...]
where an observation is
    {"start"?, "end", "val", "accn", "fy", "fp", "form", "filed", "frame"?}
"""

from dataclasses import astuple, dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import psycopg

from ingestion.sec_client import SecClient, company_facts_url


@dataclass(frozen=True)
class FinancialFact:
    ticker: str
    taxonomy: str
    concept: str
    unit: str
    period_start: date | None  # None for point-in-time facts
    period_end: date
    value: Decimal
    accession_number: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed_date: date
    frame: str | None

    @property
    def natural_key(self) -> tuple[object, ...]:
        return (
            self.ticker,
            self.taxonomy,
            self.concept,
            self.unit,
            self.period_start,
            self.period_end,
            self.accession_number,
        )


def extract_company_facts(client: SecClient, cik: int) -> dict[str, Any]:
    payload: dict[str, Any] = client.get_json(company_facts_url(cik))
    return payload


def _to_fact(
    ticker: str, taxonomy: str, concept: str, unit: str, row: dict[str, Any]
) -> FinancialFact:
    start = row.get("start")
    return FinancialFact(
        ticker=ticker,
        taxonomy=taxonomy,
        concept=concept,
        unit=unit,
        period_start=date.fromisoformat(start) if start else None,
        period_end=date.fromisoformat(row["end"]),
        value=Decimal(str(row["val"])),
        accession_number=row["accn"],
        fiscal_year=row.get("fy"),
        fiscal_period=row.get("fp"),
        form=row["form"],
        filed_date=date.fromisoformat(row["filed"]),
        frame=row.get("frame"),
    )


def flatten_company_facts(ticker: str, payload: dict[str, Any]) -> list[FinancialFact]:
    """Turn the nested payload into one row per fact, unique by natural key."""
    facts: dict[tuple[object, ...], FinancialFact] = {}
    for taxonomy, concepts in payload.get("facts", {}).items():
        for concept, body in concepts.items():
            for unit, rows in body.get("units", {}).items():
                for row in rows:
                    fact = _to_fact(ticker, taxonomy, concept, unit, row)
                    facts[fact.natural_key] = fact
    return list(facts.values())


UPSERT_FACT = """
    INSERT INTO financial_facts (
        ticker, taxonomy, concept, unit, period_start, period_end, value,
        accession_number, fiscal_year, fiscal_period, form, filed_date, frame
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT ON CONSTRAINT financial_facts_natural_key DO UPDATE
    SET value = EXCLUDED.value,
        fiscal_year = EXCLUDED.fiscal_year,
        fiscal_period = EXCLUDED.fiscal_period,
        form = EXCLUDED.form,
        filed_date = EXCLUDED.filed_date,
        frame = EXCLUDED.frame
"""


def load_facts(conn: psycopg.Connection, facts: list[FinancialFact]) -> int:
    """Idempotent upsert. Returns the number of facts sent."""
    with conn.cursor() as cursor:
        cursor.executemany(UPSERT_FACT, [astuple(fact) for fact in facts])
    return len(facts)
