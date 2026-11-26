"""Ingest SEC data for the configured companies.

Usage:
    python -m ingestion.cli                       # everything, all tickers
    python -m ingestion.cli --tickers NKE AAPL    # a subset
    python -m ingestion.cli --only xbrl           # numbers only
    python -m ingestion.cli --force               # re-parse filings already stored
"""

import argparse
import logging
import sys

from common.db import DbRole, connect
from common.logs import configure_logging
from common.settings import get_settings
from ingestion.companies import load_tickers, resolve_ciks
from ingestion.pipeline import Step, TickerReport, ingest_company, ingest_filings, ingest_xbrl
from ingestion.sec_client import TICKER_MAP_URL, SecClient, SecConfigError
from ingestion.xbrl.metrics import load_metric_definitions

logger = logging.getLogger("ingestion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tickers", nargs="+", help="default: config/companies.yaml")
    parser.add_argument("--only", choices=[step.value for step in Step], help="run a single step")
    parser.add_argument("--force", action="store_true", help="re-ingest filings already stored")
    return parser.parse_args(argv)


def print_summary(reports: list[TickerReport]) -> None:
    header = (
        f"{'ticker':<8}{'facts':>9}{'metrics':>9}{'new filings':>13}{'skipped':>9}{'sections':>10}"
    )
    print(f"\n{header}")
    for report in reports:
        print(
            f"{report.ticker:<8}{report.facts:>9}{report.metrics:>9}"
            f"{report.filings_new:>13}{report.filings_skipped:>9}{report.sections:>10}"
        )
    warnings = [(report.ticker, warning) for report in reports for warning in report.warnings]
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for ticker, warning in warnings:
            print(f"  [{ticker}] {warning}")


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    settings = get_settings()
    steps = {Step(args.only)} if args.only else set(Step)

    try:
        client = SecClient(settings.sec_user_agent, settings.sec_requests_per_second)
    except SecConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    tickers = (
        [t.upper() for t in args.tickers] if args.tickers else load_tickers(settings.config_dir)
    )
    definitions = load_metric_definitions(settings.config_dir)
    reports: list[TickerReport] = []

    with client, connect(DbRole.ADMIN, settings) as conn:
        ciks = resolve_ciks(client.get_json(TICKER_MAP_URL), tickers)
        for ticker in tickers:
            logger.info("Ingesting %s (CIK %d)", ticker, ciks[ticker])
            report = TickerReport(ticker)
            company, submissions = ingest_company(conn, client, ticker, ciks[ticker])
            if Step.XBRL in steps:
                ingest_xbrl(conn, client, company, definitions, report)
            if Step.FILINGS in steps:
                ingest_filings(
                    conn, client, company, submissions, settings, report, force=args.force
                )
            reports.append(report)

    print_summary(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
