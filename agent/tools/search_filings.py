"""The RAG tool: retrieve filing passages and register them as citable sources."""

import json
from collections.abc import Callable
from typing import Any

import psycopg

from agent.sources import SourceKind, SourceRegistry
from agent.tools.base import ToolInputError, ToolOutput
from rag.embeddings.base import EmbeddingProvider
from rag.rerank import Reranker
from rag.search import SearchFilters, search

DEFAULT_TOP_K = 6
MAX_TOP_K = 10


class SearchFilingsTool:
    name = "search_filings"
    description = (
        "Search the text of SEC 10-K and 10-Q filings (MD&A, risk factors, notes to the "
        "financial statements) for the covered companies. Call this whenever the question "
        "is about what a company or its management said, explained, expects or disclosed: "
        "reasons for a change, strategy, risks, outlook. It returns passages, each with a "
        "source_id to cite. The filings are in English, so write the query in English even "
        "if the user asked in another language. Prefer several focused searches (one per "
        "company or per period) over one broad search, and use the filters whenever the "
        "question names a company, a year or a report type. Do not use it for exact "
        "financial figures: query_financials is the reliable source for numbers."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, in English, e.g. 'reasons for gross margin "
                "decline'.",
            },
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict to these tickers, e.g. ['NKE'].",
            },
            "fiscal_years": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Restrict to filings of these fiscal years (the company's own "
                "fiscal calendar), e.g. [2025, 2026].",
            },
            "forms": {
                "type": "array",
                "items": {"type": "string", "enum": ["10-K", "10-Q"]},
                "description": "10-K = annual report, 10-Q = quarterly report.",
            },
            "top_k": {
                "type": "integer",
                "description": f"Passages to return (default {DEFAULT_TOP_K}, max {MAX_TOP_K}).",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        connect: Callable[[], psycopg.Connection],
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
        candidates: int = 40,
    ) -> None:
        self._connect = connect
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates

    def run(self, arguments: dict[str, Any], sources: SourceRegistry) -> ToolOutput:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("'query' must be a non-empty string.")
        top_k = min(int(arguments.get("top_k") or DEFAULT_TOP_K), MAX_TOP_K)
        filters = SearchFilters(
            tickers=list(arguments.get("tickers") or []),
            fiscal_years=list(arguments.get("fiscal_years") or []),
            forms=list(arguments.get("forms") or []),
        )

        with self._connect() as conn:
            results = search(
                conn,
                self._embedder,
                query,
                filters,
                top_k=top_k,
                candidates=self._candidates,
                reranker=self._reranker,
            )

        passages = [
            {
                "source_id": sources.add(
                    SourceKind.FILING, str(result.chunk_id), result.citation, result.source_url
                ),
                "citation": result.citation,
                "text": result.content,
            }
            for result in results
        ]
        if not passages:
            return ToolOutput(
                json.dumps(
                    {
                        "passages": [],
                        "note": "No passages matched. Try other words or fewer filters; if "
                        "nothing turns up, tell the user the filings do not cover it.",
                    }
                )
            )
        return ToolOutput(json.dumps({"passages": passages}))
