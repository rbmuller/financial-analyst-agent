"""Download (with a local raw cache) and persist filings and their sections."""

from pathlib import Path

import psycopg

from ingestion.companies import Company
from ingestion.filings.listing import FilingRef
from ingestion.filings.parsing import Section
from ingestion.sec_client import SecClient, filing_document_url

RAW_FILINGS_DIR = "raw/filings"


def fetch_filing_html(
    client: SecClient, company: Company, ref: FilingRef, data_dir: Path
) -> tuple[str, str]:
    """Return (html, source_url). Filings are immutable, so the cache never expires."""
    url = filing_document_url(company.cik, ref.accession_number, ref.primary_document)
    cache_path = data_dir / RAW_FILINGS_DIR / company.ticker / f"{ref.accession_number}.html"
    if cache_path.exists():
        return cache_path.read_text(), url

    html = client.get_text(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html)
    return html, url


def filing_exists(conn: psycopg.Connection, accession_number: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM filings WHERE accession_number = %s", (accession_number,)
    ).fetchone()
    return row is not None


def fiscal_label(conn: psycopg.Connection, accession_number: str) -> tuple[int | None, str | None]:
    """Fiscal year/period of a filing, as stated by its own XBRL facts."""
    row = conn.execute(
        """
        SELECT fiscal_year, fiscal_period
        FROM financial_facts
        WHERE accession_number = %s AND fiscal_year IS NOT NULL AND fiscal_period IS NOT NULL
        GROUP BY fiscal_year, fiscal_period
        ORDER BY count(*) DESC
        LIMIT 1
        """,
        (accession_number,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def save_filing(
    conn: psycopg.Connection,
    company: Company,
    ref: FilingRef,
    source_url: str,
    sections: list[Section],
) -> None:
    """Replace a filing and its sections atomically.

    Deleting cascades to sections and chunks, so a re-ingested filing never
    keeps chunks that point at stale text.
    """
    fiscal_year, fiscal_period = fiscal_label(conn, ref.accession_number)
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute("DELETE FROM filings WHERE accession_number = %s", (ref.accession_number,))
        cursor.execute(
            """
            INSERT INTO filings (
                accession_number, ticker, form, filing_date, period_of_report,
                fiscal_year, fiscal_period, source_url
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ref.accession_number,
                company.ticker,
                ref.form,
                ref.filing_date,
                ref.period_of_report,
                fiscal_year,
                fiscal_period,
                source_url,
            ),
        )
        cursor.executemany(
            """
            INSERT INTO filing_sections (accession_number, position, item, title, content)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    ref.accession_number,
                    section.position,
                    section.item,
                    section.title,
                    section.content,
                )
                for section in sections
            ],
        )
