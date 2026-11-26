"""Build and query the search index.

Usage:
    python -m rag.cli index [--rebuild]
    python -m rag.cli search "why did gross margin decline" --ticker NKE --year 2026
    python -m rag.cli search "tariffs" --mode keyword --form 10-K --rerank
"""

import argparse
import sys
import textwrap
import time

from common.db import DbRole, connect
from common.logs import configure_logging
from common.settings import Settings, get_settings
from rag.embeddings.local import SentenceTransformerEmbedder
from rag.indexing import check_dimension, chunk_pending_sections, embed_pending_chunks
from rag.rerank import CrossEncoderReranker
from rag.search import SearchFilters, SearchMode, SearchResult, search

PREVIEW_CHARS = 500
PREVIEW_WIDTH = 100
COMMAND_INDEX = "index"
COMMAND_SEARCH = "search"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser(COMMAND_INDEX, help="chunk and embed ingested filings")
    index.add_argument("--rebuild", action="store_true", help="delete all chunks and start over")

    query = commands.add_parser(COMMAND_SEARCH, help="search the indexed filings")
    query.add_argument("query")
    query.add_argument("--ticker", nargs="+", default=[])
    query.add_argument("--year", nargs="+", type=int, default=[], help="fiscal year(s)")
    query.add_argument("--form", nargs="+", default=[], help="10-K and/or 10-Q")
    query.add_argument("--top-k", type=int)
    query.add_argument(
        "--mode", choices=[mode.value for mode in SearchMode], default=SearchMode.HYBRID.value
    )
    query.add_argument("--rerank", action="store_true", help="re-rank with the cross-encoder")
    return parser.parse_args(argv)


def run_index(settings: Settings, rebuild: bool) -> None:
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    with connect(DbRole.ADMIN, settings) as conn:
        check_dimension(conn, embedder)
        if rebuild:
            conn.execute("TRUNCATE chunks")
            conn.commit()

        started = time.monotonic()
        sections, chunks = chunk_pending_sections(
            conn, embedder, settings.chunk_max_tokens, settings.chunk_overlap_tokens
        )
        print(f"Chunked {sections} section(s) into {chunks} chunk(s).")
        embedded = embed_pending_chunks(conn, embedder, settings.embedding_batch_size)
        print(f"Embedded {embedded} chunk(s) in {time.monotonic() - started:.0f}s.")


def print_results(results: list[SearchResult], elapsed_seconds: float) -> None:
    if not results:
        print("No results.")
        return
    for position, result in enumerate(results, start=1):
        ranks = f"vector #{result.vector_rank or '-'}, keyword #{result.keyword_rank or '-'}"
        print(f"\n[{position}] score {result.score:.4f} ({ranks})")
        print(f"    {result.citation}")
        print(f"    {result.source_url}")
        preview = " ".join(result.content[:PREVIEW_CHARS].split())
        print(textwrap.indent(textwrap.fill(preview, PREVIEW_WIDTH), "    | "))
    print(f"\n{len(results)} result(s) in {elapsed_seconds:.2f}s (model loading not included).")


def run_search(settings: Settings, args: argparse.Namespace) -> None:
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model) if args.rerank else None
    filters = SearchFilters(tickers=args.ticker, fiscal_years=args.year, forms=args.form)

    with connect(DbRole.ADMIN, settings) as conn:
        started = time.monotonic()
        results = search(
            conn,
            embedder,
            args.query,
            filters,
            top_k=args.top_k or settings.search_top_k,
            candidates=settings.search_candidates,
            mode=SearchMode(args.mode),
            reranker=reranker,
        )
        print_results(results, time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    settings = get_settings()
    if args.command == COMMAND_INDEX:
        run_index(settings, args.rebuild)
    else:
        run_search(settings, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
