from datetime import date
from typing import Any

import httpx

from ingestion.filings.listing import TEXT_FORMS, iter_filing_refs, list_filings
from ingestion.filings.parsing import (
    FORM_10K,
    FORM_10Q,
    FULL_DOCUMENT,
    html_to_text,
    missing_key_items,
    split_sections,
)
from ingestion.sec_client import SecClient

LONG_TEXT = "Demand softened and promotions increased during the period. " * 10


def columns(*rows: tuple[str, str, str, str]) -> dict[str, list[Any]]:
    return {
        "accessionNumber": [row[0] for row in rows],
        "form": [row[1] for row in rows],
        "filingDate": [row[2] for row in rows],
        "reportDate": [row[3] for row in rows],
        "primaryDocument": [f"{row[0]}.htm" for row in rows],
    }


def test_iter_filing_refs_keeps_only_requested_forms() -> None:
    recent = columns(
        ("a-1", "10-Q", "2024-10-01", "2024-08-31"),
        ("a-2", "4", "2024-09-15", ""),
        ("a-3", "10-K", "2024-07-20", "2024-05-31"),
    )

    refs = list(iter_filing_refs(recent, TEXT_FORMS))

    assert [ref.accession_number for ref in refs] == ["a-1", "a-3"]
    assert refs[0].period_of_report == date(2024, 8, 31)
    assert refs[0].primary_document == "a-1.htm"


def test_list_filings_applies_cutoff_and_reads_only_needed_older_pages() -> None:
    requested: list[str] = []
    old_page = columns(("b-1", "10-K", "2022-07-20", "2022-05-31"))

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, json=old_page)

    submissions = {
        "filings": {
            "recent": columns(
                ("a-1", "10-Q", "2024-10-01", "2024-08-31"),
                ("a-2", "10-K", "2023-07-20", "2023-05-31"),
            ),
            "files": [
                {"name": "page-1.json", "filingFrom": "2021-01-01", "filingTo": "2022-12-31"},
                {"name": "page-2.json", "filingFrom": "2015-01-01", "filingTo": "2020-12-31"},
            ],
        }
    }
    client = SecClient("Test test@example.com", transport=httpx.MockTransport(handler))

    refs = list_filings(client, submissions, since=date(2022, 1, 1))

    assert [ref.accession_number for ref in refs] == ["a-1", "a-2", "b-1"]
    assert requested == ["/submissions/page-1.json"]


def test_html_to_text_drops_hidden_content_and_keeps_block_structure() -> None:
    html = """
    <html><head><title>ignored</title></head><body>
      <ix:header><span>hidden xbrl metadata</span></ix:header>
      <div style="display:none">invisible</div>
      <div><span>Item 1A.</span><span>&nbsp;Risk Factors</span></div>
      <p>Our <b>gross margin</b> may decline.</p>
      <table><tr><td>Revenue</td><td>100</td></tr></table>
    </body></html>
    """

    assert html_to_text(html).split("\n") == [
        "Item 1A. Risk Factors",
        "Our gross margin may decline.",
        "Revenue 100",
    ]


def test_html_to_text_drops_page_numbers_and_table_of_contents_links() -> None:
    html = """
    <body>
      <p>Revenues grew 5% in fiscal 2026.</p>
      <div>41</div><div><a href="#toc">Table of Contents</a></div><p>Page 42</p>
      <p>We operate 41 stores.</p>
      <table><tr><td>2026</td><td>41</td></tr></table>
    </body>
    """

    assert html_to_text(html).split("\n") == [
        "Revenues grew 5% in fiscal 2026.",
        "We operate 41 stores.",
        "2026 41",
    ]


def test_split_sections_10k_prefers_real_sections_over_table_of_contents() -> None:
    text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "Item 1A. Risk Factors 12",
            "Item 7. Management's Discussion and Analysis 30",
            "PART I",
            "ITEM 1A. RISK FACTORS",
            f"Risks: {LONG_TEXT}",
            "Item 4. Mine Safety Disclosures",
            "Not applicable.",
            "PART II",
            "Item 7 — Management's Discussion and Analysis",
            f"Margins: {LONG_TEXT}",
        ]
    )

    sections = {section.item: section for section in split_sections(text, FORM_10K)}

    assert list(sections) == ["Item 1A", "Item 7", "Item 4"]
    assert sections["Item 1A"].title == "Risk Factors"
    assert sections["Item 1A"].content.startswith("ITEM 1A. RISK FACTORS\nRisks:")
    assert " 12" not in sections["Item 1A"].content
    assert sections["Item 7"].content.startswith("Item 7 — Management")
    assert sections["Item 4"].content.endswith("Not applicable.")
    assert missing_key_items(list(sections.values()), FORM_10K) == []


def test_split_sections_10q_separates_items_by_part() -> None:
    text = "\n".join(
        [
            "PART I - FINANCIAL INFORMATION",
            "Item 2. Management's Discussion and Analysis",
            f"MD&A: {LONG_TEXT}",
            "PART II - OTHER INFORMATION",
            "Item 2. Unregistered Sales of Equity Securities",
            f"Buybacks: {LONG_TEXT}",
        ]
    )

    sections = split_sections(text, FORM_10Q)

    assert [(s.item, s.title) for s in sections] == [
        ("Part I, Item 2", "Management's Discussion and Analysis"),
        ("Part II, Item 2", "Unregistered Sales of Equity Securities and Use of Proceeds"),
    ]
    assert "Buybacks" not in sections[0].content


def test_split_sections_ignores_unknown_items_and_long_lines() -> None:
    text = "\n".join(
        [
            "Item 7. Management's Discussion and Analysis",
            f"Margins: {LONG_TEXT}",
            "Item 601 exhibits are listed elsewhere.",
            f"Item 8 is discussed in this very long sentence. {LONG_TEXT}",
        ]
    )

    sections = split_sections(text, FORM_10K)

    assert [s.item for s in sections] == ["Item 7"]
    assert "Item 601" in sections[0].content


def test_split_sections_falls_back_to_full_document() -> None:
    sections = split_sections("No headings here.", FORM_10K)

    assert [(s.item, s.content) for s in sections] == [(FULL_DOCUMENT, "No headings here.")]
    assert missing_key_items(sections, FORM_10K) == ["Item 1A", "Item 7"]
