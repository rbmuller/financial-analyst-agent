-- Structured side: companies and the numbers they report in XBRL.

CREATE TABLE companies (
    ticker           TEXT PRIMARY KEY,
    cik              BIGINT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    -- Fiscal year end as MMDD (e.g. Nike's is not December).
    fiscal_year_end  CHAR(4),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per fact as published by the SEC Company Facts API, kept raw.
-- The same (concept, period) can be reported by several filings (restatements,
-- comparatives), so the accession number is part of the identity.
CREATE TABLE financial_facts (
    id                BIGSERIAL PRIMARY KEY,
    ticker            TEXT NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    taxonomy          TEXT NOT NULL,          -- us-gaap, dei, ...
    concept           TEXT NOT NULL,          -- XBRL tag, e.g. GrossProfit
    unit              TEXT NOT NULL,          -- USD, shares, USD/shares, ...
    period_start      DATE,                   -- NULL for point-in-time facts (balance sheet)
    period_end        DATE NOT NULL,
    value             NUMERIC NOT NULL,
    accession_number  TEXT NOT NULL,
    fiscal_year       INTEGER,
    fiscal_period     TEXT,                   -- FY, Q1, Q2, Q3
    form              TEXT NOT NULL,          -- 10-K, 10-Q, ...
    filed_date        DATE NOT NULL,
    frame             TEXT,                   -- SEC calendar frame, e.g. CY2024Q1
    -- Natural key that makes ingestion idempotent (upsert target).
    CONSTRAINT financial_facts_natural_key UNIQUE NULLS NOT DISTINCT
        (ticker, taxonomy, concept, unit, period_start, period_end, accession_number)
);

CREATE INDEX financial_facts_lookup_idx
    ON financial_facts (ticker, concept, period_end);
