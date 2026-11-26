"""Retrieval eval: does search put the right passage near the top?

This isolates the "R" of RAG. It needs no LLM call, so it is free and fast, and
it compares the search configurations on the same questions.
"""

from dataclasses import dataclass

import psycopg

from evals.cases import EvalCase
from rag.embeddings.base import EmbeddingProvider
from rag.rerank import Reranker
from rag.search import SearchMode, search

RANK_CUTOFF = 10
HIT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class SearchSetup:
    name: str
    mode: SearchMode
    rerank: bool = False


SETUPS = (
    SearchSetup("vector", SearchMode.VECTOR),
    SearchSetup("keyword", SearchMode.KEYWORD),
    SearchSetup("hybrid", SearchMode.HYBRID),
    SearchSetup("hybrid + rerank", SearchMode.HYBRID, rerank=True),
)


@dataclass(frozen=True)
class RetrievalMetrics:
    setup: str
    cases: int
    hit_rate: dict[int, float]  # share of cases with a relevant passage in the top k
    mrr: float  # mean of 1/rank of the first relevant passage (0 when not found)


def first_relevant_rank(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    case: EvalCase,
    setup: SearchSetup,
    reranker: Reranker | None,
    candidates: int,
) -> int | None:
    """1-based rank of the first relevant passage in the top RANK_CUTOFF, or None."""
    assert case.retrieval is not None
    results = search(
        conn,
        embedder,
        case.retrieval.query,
        case.retrieval.filters,
        top_k=RANK_CUTOFF,
        candidates=candidates,
        mode=setup.mode,
        reranker=reranker if setup.rerank else None,
    )
    for rank, result in enumerate(results, start=1):
        if case.retrieval.is_relevant(result):
            return rank
    return None


def summarize(setup: str, ranks: list[int | None]) -> RetrievalMetrics:
    total = len(ranks)
    return RetrievalMetrics(
        setup=setup,
        cases=total,
        hit_rate={
            k: sum(1 for rank in ranks if rank is not None and rank <= k) / total
            for k in HIT_CUTOFFS
        },
        mrr=sum(1 / rank for rank in ranks if rank is not None) / total,
    )


def run_retrieval_eval(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    reranker: Reranker,
    cases: list[EvalCase],
    candidates: int,
) -> tuple[list[RetrievalMetrics], dict[str, dict[str, int | None]]]:
    """Returns (metrics per setup, rank per case id per setup)."""
    with_targets = [case for case in cases if case.retrieval is not None]
    ranks_by_case: dict[str, dict[str, int | None]] = {case.id: {} for case in with_targets}
    metrics: list[RetrievalMetrics] = []
    for setup in SETUPS:
        ranks = [
            first_relevant_rank(conn, embedder, case, setup, reranker, candidates)
            for case in with_targets
        ]
        for case, rank in zip(with_targets, ranks, strict=True):
            ranks_by_case[case.id][setup.name] = rank
        metrics.append(summarize(setup.name, ranks))
    return metrics, ranks_by_case
