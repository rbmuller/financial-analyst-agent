"""Which filings to ingest, read from the SEC submissions API.

The API returns filings in columnar form (one list per field). The main JSON
holds the most recent ones under filings.recent; older ones are split into
extra pages listed under filings.files.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

from ingestion.sec_client import SecClient, submissions_page_url

TEXT_FORMS = frozenset({"10-K", "10-Q"})


@dataclass(frozen=True)
class FilingRef:
    accession_number: str
    form: str
    filing_date: date
    period_of_report: date
    primary_document: str


def iter_filing_refs(columns: dict[str, list[Any]], forms: frozenset[str]) -> Iterator[FilingRef]:
    rows = zip(
        columns["accessionNumber"],
        columns["form"],
        columns["filingDate"],
        columns["reportDate"],
        columns["primaryDocument"],
        strict=True,
    )
    for accession_number, form, filing_date, report_date, primary_document in rows:
        if form in forms:
            yield FilingRef(
                accession_number=accession_number,
                form=form,
                filing_date=date.fromisoformat(filing_date),
                period_of_report=date.fromisoformat(report_date),
                primary_document=primary_document,
            )


def list_filings(
    client: SecClient,
    submissions: dict[str, Any],
    since: date,
    forms: frozenset[str] = TEXT_FORMS,
) -> list[FilingRef]:
    """Filings of the given forms filed on or after `since`, newest first."""
    filings = submissions["filings"]
    refs = list(iter_filing_refs(filings["recent"], forms))
    for page in filings.get("files", []):
        if date.fromisoformat(page["filingTo"]) >= since:
            refs.extend(
                iter_filing_refs(client.get_json(submissions_page_url(page["name"])), forms)
            )

    selected = [ref for ref in refs if ref.filing_date >= since]
    return sorted(selected, key=lambda ref: ref.filing_date, reverse=True)
