"""HTTP client for SEC EDGAR: identifies itself, throttles and retries.

The SEC asks automated clients to send a User-Agent with contact details and
to stay under 10 requests per second; clients that do not are blocked (403).
"""

import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import httpx

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"
TICKER_MAP_URL = f"{SEC_WWW}/files/company_tickers.json"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
TIMEOUT_SECONDS = 30.0


class SecConfigError(RuntimeError):
    """The SEC client cannot be built from the current configuration."""


def submissions_url(cik: int) -> str:
    return f"{SEC_DATA}/submissions/CIK{cik:010d}.json"


def submissions_page_url(page_name: str) -> str:
    return f"{SEC_DATA}/submissions/{page_name}"


def company_facts_url(cik: int) -> str:
    return f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json"


def filing_document_url(cik: int, accession_number: str, document: str) -> str:
    folder = accession_number.replace("-", "")
    return f"{SEC_WWW}/Archives/edgar/data/{cik}/{folder}/{document}"


class SecClient:
    def __init__(
        self,
        user_agent: str | None,
        requests_per_second: float = 5.0,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise SecConfigError(
                "SEC_USER_AGENT is not set. The SEC requires a User-Agent with contact "
                'details, e.g. SEC_USER_AGENT="Your Company you@example.com" in .env.'
            )
        self._http = httpx.Client(
            headers={"User-Agent": user_agent.strip()},
            timeout=TIMEOUT_SECONDS,
            transport=transport,
            follow_redirects=True,
        )
        self._min_interval = 1.0 / requests_per_second
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._http.close()

    def get_json(self, url: str) -> Any:
        return self._get(url).json()

    def get_text(self, url: str) -> str:
        return self._get(url).text

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            wait = self._min_interval - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _get(self, url: str) -> httpx.Response:
        attempt = 1
        while True:
            self._throttle()
            response = self._http.get(url)
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                self._sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                attempt += 1
                continue
            response.raise_for_status()
            return response
