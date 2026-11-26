"""Hybrid retrieval over filing chunks.

Two retrievers with complementary strengths run over the same filtered set:

* vector search finds passages that *mean* the same as the question, even with
  different words ("profitability fell" ~ "margin contraction");
* keyword search finds passages with the *exact* terms, which embeddings blur
  (product names, "ASC 606", a specific figure).

Their ranked lists are merged with Reciprocal Rank Fusion, and an optional
cross-encoder re-ranks the merged candidates.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import psycopg

from rag.embeddings.base import EmbeddingProvider, to_pgvector
from rag.rerank import Reranker

# Standard RRF constant: dampens the gap between rank 1 and rank 2 so that one
# retriever's top hit does not automatically beat a passage both retrievers like.
RRF_K = 60

# With a WHERE filter, a plain HNSW scan returns its nearest neighbours first and
# filters afterwards, which can leave fewer rows than requested. Iterative scan
# (pgvector >= 0.8) keeps walking the graph until enough rows pass the filter.
HNSW_SESSION_SETTINGS = (
    "SET LOCAL hnsw.iterative_scan = relaxed_order",
    "SET LOCAL hnsw.ef_search = 100",
)

# ts_rank_cd normalization flag: divide the rank by itself + 1 (scales to 0..1).
TS_RANK_NORMALIZATION = 32


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


@dataclass(frozen=True)
class SearchFilters:
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """SQL conditions (each prefixed with AND) and their parameters."""
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if self.tickers:
            conditions.append("AND c.ticker = ANY(%(tickers)s)")
            params["tickers"] = [ticker.upper() for ticker in self.tickers]
        if self.fiscal_years:
            conditions.append("AND c.fiscal_year = ANY(%(fiscal_years)s)")
            params["fiscal_years"] = self.fiscal_years
        if self.forms:
            conditions.append("AND c.form = ANY(%(forms)s)")
            params["forms"] = [form.upper() for form in self.forms]
        return " ".join(conditions), params


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    ticker: str
    company: str
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: date
    item: str
    title: str
    source_url: str
    accession_number: str
    content: str

    @property
    def citation(self) -> str:
        period = f"FY{self.fiscal_year} {self.fiscal_period}" if self.fiscal_year else "n/a"
        return (
            f"{self.company} ({self.ticker}), {self.form} {period}, "
            f"filed {self.filing_date:%Y-%m-%d}, {self.item}: {self.title}"
        )


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked id lists: score(id) = sum over lists of 1 / (k + rank).

    Only ranks are used, never the raw scores, which is why a cosine distance
    and a full-text rank (different scales) can be combined at all.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def vector_search(
    conn: psycopg.Connection, query_vector: list[float], filters: SearchFilters, limit: int
) -> list[int]:
    conditions, params = filters.to_sql()
    with conn.transaction():
        for setting in HNSW_SESSION_SETTINGS:
            conn.execute(setting)
        rows = conn.execute(
            f"""
            SELECT c.id
            FROM chunks c
            WHERE c.embedding IS NOT NULL {conditions}
            ORDER BY c.embedding <=> %(query_vector)s::vector
            LIMIT %(limit)s
            """,
            {**params, "query_vector": to_pgvector(query_vector), "limit": limit},
        ).fetchall()
    return [row[0] for row in rows]


def keyword_search(
    conn: psycopg.Connection, query: str, filters: SearchFilters, limit: int
) -> list[int]:
    """Full-text search. Terms are OR-ed: a question rarely appears word for word,
    and ts_rank_cd already rewards passages that cover more of its terms."""
    conditions, params = filters.to_sql()
    rows = conn.execute(
        f"""
        WITH parsed AS (
            SELECT NULLIF(
                replace(plainto_tsquery('english', %(query)s)::text, '&', '|'), ''
            )::tsquery AS tsquery
        )
        SELECT c.id
        FROM chunks c, parsed
        WHERE c.content_tsv @@ parsed.tsquery {conditions}
        ORDER BY ts_rank_cd(c.content_tsv, parsed.tsquery, {TS_RANK_NORMALIZATION}) DESC, c.id
        LIMIT %(limit)s
        """,
        {**params, "query": query, "limit": limit},
    ).fetchall()
    return [row[0] for row in rows]


def _fetch_results(
    conn: psycopg.Connection,
    scored: list[tuple[int, float]],
    vector_ids: list[int],
    keyword_ids: list[int],
) -> list[SearchResult]:
    rows = conn.execute(
        """
        SELECT c.id, c.ticker, co.name, f.form, f.fiscal_year, f.fiscal_period, f.filing_date,
               s.item, s.title, f.source_url, f.accession_number, c.content
        FROM chunks c
        JOIN filing_sections s ON s.id = c.section_id
        JOIN filings f ON f.accession_number = s.accession_number
        JOIN companies co ON co.ticker = f.ticker
        WHERE c.id = ANY(%s)
        """,
        ([chunk_id for chunk_id, _ in scored],),
    ).fetchall()
    by_id = {row[0]: row for row in rows}
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
    keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}

    return [
        SearchResult(
            chunk_id,
            score,
            vector_rank.get(chunk_id),
            keyword_rank.get(chunk_id),
            *by_id[chunk_id][1:],
        )
        for chunk_id, score in scored
        if chunk_id in by_id
    ]


def _rerank(
    reranker: Reranker, query: str, results: list[SearchResult], top_k: int
) -> list[SearchResult]:
    scores = reranker.score(query, [result.content for result in results])
    rescored = [
        SearchResult(**{**result.__dict__, "score": score})
        for result, score in zip(results, scores, strict=True)
    ]
    return sorted(rescored, key=lambda result: result.score, reverse=True)[:top_k]


def search(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    query: str,
    filters: SearchFilters | None = None,
    *,
    top_k: int = 8,
    candidates: int = 40,
    mode: SearchMode = SearchMode.HYBRID,
    reranker: Reranker | None = None,
) -> list[SearchResult]:
    filters = filters or SearchFilters()
    vector_ids: list[int] = []
    keyword_ids: list[int] = []
    if mode is not SearchMode.KEYWORD:
        vector_ids = vector_search(conn, embedder.embed_query(query), filters, candidates)
    if mode is not SearchMode.VECTOR:
        keyword_ids = keyword_search(conn, query, filters, candidates)

    fused = reciprocal_rank_fusion([ids for ids in (vector_ids, keyword_ids) if ids])
    # Without a re-ranker the fused order is final; with one, it judges a wider pool.
    pool = fused[:candidates] if reranker else fused[:top_k]
    results = _fetch_results(conn, pool, vector_ids, keyword_ids)
    return _rerank(reranker, query, results, top_k) if reranker else results
