-- Unstructured side: filings -> sections -> chunks (the unit that is embedded).

CREATE TABLE filings (
    accession_number  TEXT PRIMARY KEY,
    ticker            TEXT NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    form              TEXT NOT NULL,          -- 10-K, 10-Q
    filing_date       DATE NOT NULL,
    period_of_report  DATE NOT NULL,
    fiscal_year       INTEGER,
    fiscal_period     TEXT,
    source_url        TEXT NOT NULL,          -- used verbatim in citations
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX filings_ticker_idx ON filings (ticker, form, filing_date);

CREATE TABLE filing_sections (
    id                BIGSERIAL PRIMARY KEY,
    accession_number  TEXT NOT NULL REFERENCES filings (accession_number) ON DELETE CASCADE,
    position          INTEGER NOT NULL,       -- order inside the filing
    item              TEXT NOT NULL,          -- e.g. "Item 7"
    title             TEXT NOT NULL,          -- e.g. "Management's Discussion and Analysis"
    content           TEXT NOT NULL,
    UNIQUE (accession_number, position)
);

-- ticker / form / fiscal_year are denormalized from `filings` on purpose:
-- search filters on them, and keeping them next to the vector avoids joins
-- in the hot path.
CREATE TABLE chunks (
    id                BIGSERIAL PRIMARY KEY,
    section_id        BIGINT NOT NULL REFERENCES filing_sections (id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,
    ticker            TEXT NOT NULL,
    form              TEXT NOT NULL,
    fiscal_year       INTEGER,
    content           TEXT NOT NULL,
    token_count       INTEGER NOT NULL,
    -- Dimension must match the embedding model (384 = BAAI/bge-small-en-v1.5).
    -- Switching to a model with another dimension needs a migration + re-embedding.
    embedding         vector(384),
    embedding_model   TEXT,
    -- Full-text side of hybrid search, maintained by Postgres itself.
    content_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (section_id, chunk_index)
);

-- Approximate nearest-neighbour index for semantic search (cosine distance).
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted index for keyword search.
CREATE INDEX chunks_content_tsv_idx ON chunks USING gin (content_tsv);

CREATE INDEX chunks_filter_idx ON chunks (ticker, fiscal_year, form);
