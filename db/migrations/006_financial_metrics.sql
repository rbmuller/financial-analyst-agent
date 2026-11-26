-- Curated layer on top of the raw facts: one value per company, metric and
-- fiscal period, already resolved across XBRL tag variants and restatements.
-- Rebuilt from financial_facts on every ingestion (see ingestion/xbrl/metrics.py).
CREATE TABLE financial_metrics (
    ticker            TEXT NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    metric            TEXT NOT NULL,          -- revenue, gross_profit, ...
    fiscal_year       INTEGER NOT NULL,
    fiscal_period     TEXT NOT NULL CHECK (fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')),
    period_start      DATE NOT NULL,
    period_end        DATE NOT NULL,
    value             NUMERIC NOT NULL,
    unit              TEXT NOT NULL,
    -- Provenance, so every number can be cited.
    concept           TEXT NOT NULL,          -- XBRL tag the value came from
    accession_number  TEXT NOT NULL,
    form              TEXT NOT NULL,
    filed_date        DATE NOT NULL,
    -- true when computed here instead of reported (Q4 = FY - Q1 - Q2 - Q3).
    is_derived        BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (ticker, metric, fiscal_year, fiscal_period)
);

GRANT SELECT ON financial_metrics TO agent_readonly;
