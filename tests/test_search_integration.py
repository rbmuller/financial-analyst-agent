"""Indexing and hybrid search against the real Postgres, with a fake embedder.

The fake maps a few known words to fixed directions, so "semantic" neighbours
are predictable without loading a model. Runs in the test database (see conftest).
"""

from collections.abc import Iterator
from datetime import date

import psycopg
import pytest

from common.db import DbRole, connect
from ingestion.companies import Company, upsert_company
from ingestion.filings.listing import FilingRef
from ingestion.filings.parsing import Section
from ingestion.filings.store import save_filing
from rag.embeddings.base import to_pgvector
from rag.indexing import (
    DimensionMismatchError,
    check_dimension,
    chunk_pending_sections,
    embed_pending_chunks,
)
from rag.search import SearchFilters, SearchMode, search

pytestmark = pytest.mark.integration

DIMENSION = 384
TICKER = "ZZSRCH"
COMPANY = Company(ticker=TICKER, cik=999999998, name="Search Test Inc.", fiscal_year_end="1231")
TOPICS = {"margin": 0, "profitability": 0, "lawsuit": 1, "litigation": 1, "weather": 2}

MARGIN_TEXT = "Gross margin declined because of higher promotions."
PROFIT_TEXT = "Profitability was pressured by discounting during the quarter."
LAWSUIT_TEXT = "A lawsuit was filed against the company in Delaware."


class FakeEmbedder:
    model_name = "fake-topic-model"
    dimension = DIMENSION

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSION
        for word, axis in TOPICS.items():
            if word in text.lower():
                vector[axis] = 1.0
        if not any(vector):
            vector[DIMENSION - 1] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def add_filing(conn: psycopg.Connection, accession: str, form: str, texts: list[str]) -> None:
    ref = FilingRef(accession, form, date(2026, 2, 1), date(2025, 12, 31), "doc.htm")
    sections = [
        Section(position, f"Item {position + 1}", "Test section", text)
        for position, text in enumerate(texts)
    ]
    save_filing(conn, COMPANY, ref, f"https://example.test/{accession}", sections)


NOISE_TICKER = "ZZNOIS"
NOISE_COMPANY = Company(
    ticker=NOISE_TICKER, cik=999999997, name="Noise Corp", fiscal_year_end="1231"
)
NOISE_CHUNKS = 300  # more than hnsw.ef_search, all as close to a "margin" query as possible


def add_noise_chunks(conn: psycopg.Connection) -> None:
    """Another company's chunks that are the global nearest neighbours of "margin"."""
    upsert_company(conn, NOISE_COMPANY)
    ref = FilingRef("9999999997-26-000001", "10-K", date(2026, 2, 1), date(2025, 12, 31), "d.htm")
    save_filing(conn, NOISE_COMPANY, ref, "https://example.test/noise", [])
    section = conn.execute(
        """
        INSERT INTO filing_sections (accession_number, position, item, title, content)
        VALUES (%s, 0, 'Item 1', 'Noise', 'noise') RETURNING id
        """,
        (ref.accession_number,),
    ).fetchone()
    assert section is not None
    conn.execute(
        """
        INSERT INTO chunks (section_id, chunk_index, ticker, form, fiscal_year, content,
                            token_count, embedding, embedding_model)
        SELECT %s, n, %s, '10-K', 2025, 'unrelated filler ' || n, 3, %s::vector, %s
        FROM generate_series(1, %s) AS n
        """,
        (
            section[0],
            NOISE_TICKER,
            to_pgvector(FakeEmbedder().embed_query("margin")),
            FakeEmbedder.model_name,
            NOISE_CHUNKS,
        ),
    )


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    with connect(DbRole.ADMIN) as connection:
        try:
            upsert_company(connection, COMPANY)
            add_filing(connection, "9999999998-26-000001", "10-K", [MARGIN_TEXT, LAWSUIT_TEXT])
            add_filing(connection, "9999999998-26-000002", "10-Q", [PROFIT_TEXT])
            add_noise_chunks(connection)
            connection.commit()  # indexing commits per section, so the setup must exist
            yield connection
        finally:
            connection.rollback()
            connection.execute(
                "DELETE FROM companies WHERE ticker = ANY(%s)", ([TICKER, NOISE_TICKER],)
            )
            connection.commit()


@pytest.fixture
def indexed(conn: psycopg.Connection) -> FakeEmbedder:
    embedder = FakeEmbedder()
    chunk_pending_sections(conn, embedder, max_tokens=50, overlap_tokens=0)
    embed_pending_chunks(conn, embedder, batch_size=2)
    return embedder


def contents(
    conn: psycopg.Connection, embedder: FakeEmbedder, query: str, **kwargs: object
) -> list[str]:
    filters = kwargs.pop("filters", SearchFilters(tickers=[TICKER]))
    results = search(conn, embedder, query, filters, **kwargs)  # type: ignore[arg-type]
    return [result.content for result in results]


def test_indexing_is_idempotent(conn: psycopg.Connection, indexed: FakeEmbedder) -> None:
    assert chunk_pending_sections(conn, indexed, max_tokens=50, overlap_tokens=0) == (0, 0)
    assert embed_pending_chunks(conn, indexed, batch_size=2) == 0

    row = conn.execute(
        "SELECT count(*), count(embedding) FROM chunks WHERE ticker = %s", (TICKER,)
    ).fetchone()
    assert row == (3, 3)


def test_vector_search_matches_meaning_without_shared_words(
    conn: psycopg.Connection, indexed: FakeEmbedder
) -> None:
    found = contents(conn, indexed, "profitability", mode=SearchMode.VECTOR, top_k=2)

    assert set(found) == {MARGIN_TEXT, PROFIT_TEXT}


def test_keyword_search_matches_exact_terms_only(
    conn: psycopg.Connection, indexed: FakeEmbedder
) -> None:
    assert contents(conn, indexed, "Delaware weather", mode=SearchMode.KEYWORD) == [LAWSUIT_TEXT]
    assert contents(conn, indexed, "the of and", mode=SearchMode.KEYWORD) == []


def test_hybrid_ranks_first_what_both_retrievers_agree_on(
    conn: psycopg.Connection, indexed: FakeEmbedder
) -> None:
    results = search(conn, indexed, "gross margin promotions", SearchFilters(tickers=[TICKER]))

    assert results[0].content == MARGIN_TEXT
    assert results[0].vector_rank is not None and results[0].keyword_rank == 1
    assert results[0].company == COMPANY.name
    assert results[0].source_url.endswith("000001")


def test_filters_are_applied_inside_the_vector_scan(
    conn: psycopg.Connection, indexed: FakeEmbedder
) -> None:
    # Another company owns the 300 nearest neighbours of this query. Without
    # iterative scan the index returns only those and the filter drops them all.
    only_10q = SearchFilters(tickers=[TICKER], forms=["10-Q"])

    assert contents(conn, indexed, "margin", mode=SearchMode.VECTOR, filters=only_10q) == [
        PROFIT_TEXT
    ]


def test_reranker_reorders_candidates(conn: psycopg.Connection, indexed: FakeEmbedder) -> None:
    class PreferLawsuit:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [1.0 if "lawsuit" in passage else 0.0 for passage in passages]

    found = contents(conn, indexed, "margin lawsuit", reranker=PreferLawsuit(), top_k=1)

    assert found == [LAWSUIT_TEXT]


def test_dimension_check_rejects_a_model_that_does_not_fit(conn: psycopg.Connection) -> None:
    class WrongSize(FakeEmbedder):
        dimension = 768

    check_dimension(conn, FakeEmbedder())
    with pytest.raises(DimensionMismatchError, match="768"):
        check_dimension(conn, WrongSize())
