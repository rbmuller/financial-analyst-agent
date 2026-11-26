"""Company reference data: tickers from config, CIK and names from the SEC."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

COMPANIES_FILE = "companies.yaml"


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    fiscal_year_end: str | None  # MMDD


class UnknownTickerError(ValueError):
    pass


def load_tickers(config_dir: Path) -> list[str]:
    payload = yaml.safe_load((config_dir / COMPANIES_FILE).read_text())
    return [str(ticker).upper() for ticker in payload["tickers"]]


def resolve_ciks(ticker_map: dict[str, Any], tickers: list[str]) -> dict[str, int]:
    """Map tickers to CIKs using the SEC's company_tickers.json payload."""
    cik_by_ticker = {str(row["ticker"]).upper(): int(row["cik_str"]) for row in ticker_map.values()}
    missing = [ticker for ticker in tickers if ticker not in cik_by_ticker]
    if missing:
        raise UnknownTickerError(f"Tickers not found in the SEC ticker map: {', '.join(missing)}")
    return {ticker: cik_by_ticker[ticker] for ticker in tickers}


def build_company(ticker: str, cik: int, submissions: dict[str, Any]) -> Company:
    return Company(
        ticker=ticker,
        cik=cik,
        name=str(submissions["name"]),
        fiscal_year_end=submissions.get("fiscalYearEnd") or None,
    )


def upsert_company(conn: psycopg.Connection, company: Company) -> None:
    conn.execute(
        """
        INSERT INTO companies (ticker, cik, name, fiscal_year_end)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE
        SET cik = EXCLUDED.cik,
            name = EXCLUDED.name,
            fiscal_year_end = EXCLUDED.fiscal_year_end,
            updated_at = now()
        """,
        (company.ticker, company.cik, company.name, company.fiscal_year_end),
    )
