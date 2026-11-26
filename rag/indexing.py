"""Build the search index: sections -> chunks -> embeddings.

Two idempotent, resumable steps (each maps to a future Airflow task):

1. chunk every section that has no chunks yet;
2. embed every chunk that has no embedding from the current model.

Embedding is the slow step and commits per batch, so an interrupted run
continues where it stopped.
"""

import logging

import psycopg

from rag.chunking import SKIPPED_ITEMS, chunk_text
from rag.embeddings.base import EmbeddingProvider, to_pgvector

logger = logging.getLogger(__name__)

PROGRESS_EVERY_BATCHES = 10
# A chunk needs (re-)embedding when it has no vector or one from another model.
NEEDS_EMBEDDING = "c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %s"


class DimensionMismatchError(RuntimeError):
    pass


def check_dimension(conn: psycopg.Connection, embedder: EmbeddingProvider) -> None:
    """Fail early if the model does not fit the `chunks.embedding` column."""
    row = conn.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """
    ).fetchone()
    column_dimension = row[0] if row else None
    if column_dimension != embedder.dimension:
        raise DimensionMismatchError(
            f"Model '{embedder.model_name}' produces {embedder.dimension}-dimensional vectors, "
            f"but chunks.embedding is vector({column_dimension}). "
            "Changing models needs a migration that alters the column."
        )


def embedding_input(
    content: str,
    company: str,
    ticker: str,
    form: str,
    fiscal_year: int | None,
    fiscal_period: str | None,
    item: str,
    title: str,
) -> str:
    """Text that is actually embedded: the chunk plus a one-line context header.

    A chunk like "Gross margin decreased 120 basis points" does not say whose
    margin or when. The header puts that context into the vector; the stored
    `content` stays clean for display and citations.
    """
    period = f"FY{fiscal_year} {fiscal_period}" if fiscal_year else ""
    header = " ".join(f"{company} ({ticker}) {form} {period}".split())
    return f"{header} | {item}: {title}\n{content}"


def chunk_pending_sections(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[int, int]:
    """Returns (sections chunked, chunks created)."""
    sections = conn.execute(
        """
        SELECT s.id, s.content, f.ticker, f.form, f.fiscal_year
        FROM filing_sections s
        JOIN filings f ON f.accession_number = s.accession_number
        WHERE s.item <> ALL(%s)
          AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.section_id = s.id)
        ORDER BY s.id
        """,
        (sorted(SKIPPED_ITEMS),),
    ).fetchall()

    created = 0
    for section_id, content, ticker, form, fiscal_year in sections:
        chunks = chunk_text(content, embedder.count_tokens, max_tokens, overlap_tokens)
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (
                    section_id, chunk_index, ticker, form, fiscal_year, content, token_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (section_id, c.index, ticker, form, fiscal_year, c.content, c.token_count)
                    for c in chunks
                ],
            )
        conn.commit()
        created += len(chunks)
    return len(sections), created


def embed_pending_chunks(
    conn: psycopg.Connection, embedder: EmbeddingProvider, batch_size: int
) -> int:
    """Embed chunks with no vector, or with a vector from another model."""
    pending = conn.execute(
        f"SELECT count(*) FROM chunks c WHERE {NEEDS_EMBEDDING}",
        (embedder.model_name,),
    ).fetchone()
    total = pending[0] if pending else 0

    embedded = 0
    batches = 0
    while True:
        rows = conn.execute(
            f"""
            SELECT c.id, c.content, co.name, c.ticker, f.form, f.fiscal_year, f.fiscal_period,
                   s.item, s.title
            FROM chunks c
            JOIN filing_sections s ON s.id = c.section_id
            JOIN filings f ON f.accession_number = s.accession_number
            JOIN companies co ON co.ticker = f.ticker
            WHERE {NEEDS_EMBEDDING}
            ORDER BY c.id
            LIMIT %s
            """,
            (embedder.model_name, batch_size),
        ).fetchall()
        if not rows:
            return embedded

        vectors = embedder.embed_documents([embedding_input(*row[1:]) for row in rows])
        with conn.cursor() as cursor:
            cursor.executemany(
                "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
                [
                    (to_pgvector(vector), embedder.model_name, row[0])
                    for row, vector in zip(rows, vectors, strict=True)
                ],
            )
        conn.commit()

        embedded += len(rows)
        batches += 1
        if batches % PROGRESS_EVERY_BATCHES == 0:
            logger.info("Embedded %d / %d chunks", embedded, total)
