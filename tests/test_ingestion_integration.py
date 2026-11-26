"""Load functions against the real Postgres. Everything is rolled back at the end."""

from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import psycopg
import pytest

from common.db import DbRole, connect
from ingestion.companies import Company, upsert_company
from ingestion.filings.listing import FilingRef
from ingestion.filings.parsing import Section
from ingestion.filings.store import filing_exists, save_filing
from ingestion.xbrl.facts import FinancialFact, load_facts
from ingestion.xbrl.metrics import MetricValue, replace_metrics

pytestmark = pytest.mark.integration

COMPANY = Company(
    ticker="ZZTEST", cik=999999999, name="Integration Test Inc.", fiscal_year_end="0531"
)
ACCESSION = "9999999999-24-000001"
SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/999999999/doc.htm"
FILING = FilingRef(ACCESSION, "10-Q", date(2024, 10, 1), date(2024, 8, 31), "doc.htm")


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    with connect(DbRole.ADMIN) as connection:
        upsert_company(connection, COMPANY)
        yield connection
        connection.rollback()


def make_fact(value: int, period_start: date | None = date(2024, 6, 1)) -> FinancialFact:
    return FinancialFact(
        ticker=COMPANY.ticker,
        taxonomy="us-gaap",
        concept="Revenues" if period_start else "Assets",
        unit="USD",
        period_start=period_start,
        period_end=date(2024, 8, 31),
        value=Decimal(value),
        accession_number=ACCESSION,
        fiscal_year=2025,
        fiscal_period="Q1",
        form="10-Q",
        filed_date=date(2024, 10, 1),
        frame=None,
    )


def count(conn: psycopg.Connection, table: str) -> int:
    row = conn.execute(
        psycopg.sql.SQL("SELECT count(*) FROM {} WHERE {} = %s").format(
            psycopg.sql.Identifier(table),
            psycopg.sql.Identifier("ticker" if table != "filing_sections" else "accession_number"),
        ),
        (COMPANY.ticker if table != "filing_sections" else ACCESSION,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_load_facts_is_idempotent_and_updates_values(conn: psycopg.Connection) -> None:
    # The instant fact has a NULL period_start: it must still upsert, not duplicate.
    load_facts(conn, [make_fact(100), make_fact(500, period_start=None)])
    load_facts(conn, [make_fact(101), make_fact(500, period_start=None)])

    assert count(conn, "financial_facts") == 2
    row = conn.execute(
        "SELECT value FROM financial_facts WHERE ticker = %s AND concept = 'Revenues'",
        (COMPANY.ticker,),
    ).fetchone()
    assert row is not None and row[0] == 101


def test_replace_metrics_rebuilds_the_company(conn: psycopg.Connection) -> None:
    def metric(period: str, value: int) -> MetricValue:
        return MetricValue(
            COMPANY.ticker, "revenue", 2025, period, date(2024, 6, 1), date(2024, 8, 31),
            Decimal(value), "USD", "Revenues", ACCESSION, "10-Q", date(2024, 10, 1),
        )  # fmt: skip

    replace_metrics(conn, COMPANY.ticker, [metric("Q1", 100), metric("Q2", 110)])
    replace_metrics(conn, COMPANY.ticker, [metric("Q1", 105)])

    assert count(conn, "financial_metrics") == 1


def test_save_filing_takes_fiscal_label_from_facts_and_replaces_sections(
    conn: psycopg.Connection,
) -> None:
    load_facts(conn, [make_fact(100)])
    assert not filing_exists(conn, ACCESSION)

    two_sections = [Section(0, "Part I, Item 1", "Financial Statements", "numbers"),
                    Section(1, "Part I, Item 2", "MD&A", "discussion")]  # fmt: skip
    save_filing(conn, COMPANY, FILING, SOURCE_URL, two_sections)
    save_filing(conn, COMPANY, FILING, SOURCE_URL, two_sections[:1])

    assert filing_exists(conn, ACCESSION)
    assert count(conn, "filing_sections") == 1
    row = conn.execute(
        "SELECT fiscal_year, fiscal_period, source_url FROM filings WHERE accession_number = %s",
        (ACCESSION,),
    ).fetchone()
    assert row == (2025, "Q1", SOURCE_URL)
