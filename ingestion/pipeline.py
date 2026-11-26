"""Ingestion steps, one function per unit of work.

Each function is idempotent and takes everything it needs as arguments, so the
same functions can be wrapped as Airflow tasks later (one DAG run per ticker:
company -> xbrl -> filings).
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import psycopg

from common.settings import Settings
from ingestion.companies import Company, build_company, upsert_company
from ingestion.filings.listing import list_filings
from ingestion.filings.parsing import html_to_text, missing_key_items, split_sections
from ingestion.filings.store import fetch_filing_html, filing_exists, save_filing
from ingestion.sec_client import SecClient, submissions_url
from ingestion.xbrl.facts import extract_company_facts, flatten_company_facts, load_facts
from ingestion.xbrl.metrics import MetricDefinition, build_metrics, replace_metrics

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365


class Step(StrEnum):
    XBRL = "xbrl"
    FILINGS = "filings"


@dataclass
class TickerReport:
    ticker: str
    facts: int = 0
    metrics: int = 0
    filings_new: int = 0
    filings_skipped: int = 0
    sections: int = 0
    warnings: list[str] = field(default_factory=list)


def ingest_company(
    conn: psycopg.Connection, client: SecClient, ticker: str, cik: int
) -> tuple[Company, dict[str, Any]]:
    submissions: dict[str, Any] = client.get_json(submissions_url(cik))
    company = build_company(ticker, cik, submissions)
    upsert_company(conn, company)
    conn.commit()
    return company, submissions


def ingest_xbrl(
    conn: psycopg.Connection,
    client: SecClient,
    company: Company,
    definitions: list[MetricDefinition],
    report: TickerReport,
) -> None:
    facts = flatten_company_facts(company.ticker, extract_company_facts(client, company.cik))
    report.facts = load_facts(conn, facts)
    metrics = build_metrics(company.ticker, facts, definitions)
    report.metrics = replace_metrics(conn, company.ticker, metrics)
    conn.commit()

    covered = {metric.metric for metric in metrics}
    for definition in definitions:
        if definition.name not in covered:
            report.warnings.append(f"metric '{definition.name}': no XBRL concept matched")


def ingest_filings(
    conn: psycopg.Connection,
    client: SecClient,
    company: Company,
    submissions: dict[str, Any],
    settings: Settings,
    report: TickerReport,
    *,
    force: bool = False,
    today: date | None = None,
) -> None:
    since = (today or date.today()) - timedelta(
        days=DAYS_PER_YEAR * settings.filings_lookback_years
    )

    for ref in list_filings(client, submissions, since):
        if not force and filing_exists(conn, ref.accession_number):
            report.filings_skipped += 1
            continue

        html, source_url = fetch_filing_html(client, company, ref, settings.data_dir)
        sections = split_sections(html_to_text(html), ref.form)
        save_filing(conn, company, ref, source_url, sections)
        conn.commit()

        report.filings_new += 1
        report.sections += len(sections)
        missing = missing_key_items(sections, ref.form)
        if missing:
            report.warnings.append(
                f"{ref.form} {ref.filing_date} ({ref.accession_number}): "
                f"section(s) not found: {', '.join(missing)}"
            )
        logger.info(
            "%s %s %s: %d sections", company.ticker, ref.form, ref.filing_date, len(sections)
        )
