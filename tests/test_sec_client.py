import httpx
import pytest

from ingestion.companies import UnknownTickerError, build_company, resolve_ciks
from ingestion.sec_client import (
    MAX_ATTEMPTS,
    SecClient,
    SecConfigError,
    company_facts_url,
    filing_document_url,
)

USER_AGENT = "Test Company test@example.com"
URL = "https://data.sec.gov/anything.json"


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(handler: httpx.MockTransport, fake_time: FakeTime) -> SecClient:
    return SecClient(
        USER_AGENT,
        requests_per_second=5,
        transport=handler,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )


@pytest.mark.parametrize("user_agent", [None, "", "   "])
def test_client_requires_a_user_agent(user_agent: str | None) -> None:
    with pytest.raises(SecConfigError, match="SEC_USER_AGENT"):
        SecClient(user_agent)


def test_client_sends_user_agent_and_throttles_requests() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, json={"ok": True})

    fake_time = FakeTime()
    client = make_client(httpx.MockTransport(handler), fake_time)

    assert client.get_json(URL) == {"ok": True}
    client.get_json(URL)

    assert seen == [USER_AGENT, USER_AGENT]
    assert fake_time.sleeps == [pytest.approx(0.2)]  # 5 requests per second


def test_client_retries_transient_errors_with_backoff() -> None:
    statuses = iter([429, 503, 200])
    fake_time = FakeTime()
    client = make_client(
        httpx.MockTransport(lambda request: httpx.Response(next(statuses), json={})), fake_time
    )

    client.get_json(URL)

    backoff_sleeps = [seconds for seconds in fake_time.sleeps if seconds >= 1]
    assert backoff_sleeps == [2.0, 4.0]


def test_client_gives_up_after_max_attempts_and_does_not_retry_403() -> None:
    calls = {"count": 0}

    def always(status: int) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(status)

        return httpx.MockTransport(handler)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(always(503), FakeTime()).get_text(URL)
    assert calls["count"] == MAX_ATTEMPTS

    calls["count"] = 0
    with pytest.raises(httpx.HTTPStatusError):
        make_client(always(403), FakeTime()).get_text(URL)
    assert calls["count"] == 1


def test_urls_pad_cik_and_strip_accession_dashes() -> None:
    assert company_facts_url(1234) == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000001234.json"
    )
    assert filing_document_url(1234, "0000001234-24-000007", "doc.htm") == (
        "https://www.sec.gov/Archives/edgar/data/1234/000000123424000007/doc.htm"
    )


def test_resolve_ciks_and_build_company() -> None:
    ticker_map = {
        "0": {"cik_str": 1234, "ticker": "TEST", "title": "Test Inc."},
        "1": {"cik_str": 5678, "ticker": "OTHR", "title": "Other Corp"},
    }

    assert resolve_ciks(ticker_map, ["TEST"]) == {"TEST": 1234}
    with pytest.raises(UnknownTickerError, match="NOPE"):
        resolve_ciks(ticker_map, ["TEST", "NOPE"])

    company = build_company("TEST", 1234, {"name": "Test Inc.", "fiscalYearEnd": "0531"})
    assert (company.name, company.fiscal_year_end) == ("Test Inc.", "0531")
