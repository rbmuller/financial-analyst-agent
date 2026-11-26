from datetime import date

import pytest

from rag.chunking import chunk_text
from rag.embeddings.base import to_pgvector
from rag.indexing import embedding_input
from rag.search import SearchFilters, SearchResult, reciprocal_rank_fusion


def count_words(text: str) -> int:
    return len(text.split())


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text(
        "One paragraph.\nAnother one.", count_words, max_tokens=50, overlap_tokens=5
    )

    assert [(c.index, c.content, c.token_count) for c in chunks] == [
        (0, "One paragraph.\nAnother one.", 4)
    ]


def test_paragraphs_are_packed_and_never_exceed_the_limit() -> None:
    text = "\n".join(words(f"p{paragraph}w", 30) for paragraph in range(10))

    chunks = chunk_text(text, count_words, max_tokens=100, overlap_tokens=0)

    assert [c.token_count for c in chunks] == [90, 90, 90, 30]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert "\n".join(c.content for c in chunks) == text  # nothing lost, nothing repeated


def test_consecutive_chunks_share_the_overlap() -> None:
    text = "\n".join(words(f"p{paragraph}w", 30) for paragraph in range(6))

    chunks = chunk_text(text, count_words, max_tokens=100, overlap_tokens=30)

    first_lines = [c.content.split("\n") for c in chunks]
    assert first_lines[1][0] == first_lines[0][-1]
    assert first_lines[2][0] == first_lines[1][-1]
    assert all(c.token_count <= 100 for c in chunks)


def test_long_paragraph_is_split_by_sentence_then_by_words() -> None:
    sentences = " ".join(f"{words(f's{n}w', 9).capitalize()} end." for n in range(5))
    endless = words("x", 45)  # no sentence boundary at all

    chunks = chunk_text(f"{sentences}\n{endless}", count_words, max_tokens=20, overlap_tokens=0)

    assert all(c.token_count <= 20 for c in chunks)
    assert chunks[0].content.endswith("end.")
    assert " ".join(c.content.replace("\n", " ") for c in chunks).split() == (
        f"{sentences} {endless}".split()
    )


def test_blank_text_produces_no_chunks() -> None:
    assert chunk_text(" \n\n ", count_words, max_tokens=10, overlap_tokens=2) == []


def test_rrf_rewards_agreement_between_retrievers() -> None:
    vector_ranking = [1, 2, 3]
    keyword_ranking = [4, 2, 5]

    fused = reciprocal_rank_fusion([vector_ranking, keyword_ranking], k=60)

    assert fused[0][0] == 2  # second in both beats first in only one
    assert fused[0][1] == pytest.approx(2 / 62)
    assert {chunk_id for chunk_id, _ in fused} == {1, 2, 3, 4, 5}
    assert fused[1][1] == fused[2][1] == pytest.approx(1 / 61)


def test_rrf_with_a_single_ranking_keeps_its_order() -> None:
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([[7, 8, 9]])] == [7, 8, 9]


def test_filters_build_parameterized_conditions() -> None:
    assert SearchFilters().to_sql() == ("", {})

    sql, params = SearchFilters(tickers=["nke"], fiscal_years=[2025, 2026], forms=["10-k"]).to_sql()

    assert sql == (
        "AND c.ticker = ANY(%(tickers)s) "
        "AND c.fiscal_year = ANY(%(fiscal_years)s) "
        "AND c.form = ANY(%(forms)s)"
    )
    assert params == {"tickers": ["NKE"], "fiscal_years": [2025, 2026], "forms": ["10-K"]}


def test_to_pgvector_format() -> None:
    assert to_pgvector([0.5, -1.0, 0.123456789]) == "[0.5,-1,0.1234568]"


def test_embedding_input_adds_context_header() -> None:
    text = embedding_input(
        "Gross margin decreased.", "NIKE, Inc.", "NKE", "10-Q", 2026, "Q2", "Part I, Item 2", "MD&A"
    )

    assert text == "NIKE, Inc. (NKE) 10-Q FY2026 Q2 | Part I, Item 2: MD&A\nGross margin decreased."


def test_citation_has_company_form_period_date_and_section() -> None:
    result = SearchResult(
        chunk_id=1, score=0.5, vector_rank=1, keyword_rank=None, ticker="NKE",
        company="NIKE, Inc.", form="10-K", fiscal_year=2026, fiscal_period="FY",
        filing_date=date(2026, 7, 15), item="Item 7",
        title="Management's Discussion and Analysis", source_url="https://example.test/doc",
        accession_number="0000000000-26-000001", content="text",
    )  # fmt: skip

    assert result.citation == (
        "NIKE, Inc. (NKE), 10-K FY2026 FY, filed 2026-07-15, "
        "Item 7: Management's Discussion and Analysis"
    )
