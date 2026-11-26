-- Observability: one row per question (run) and one per step inside it.

CREATE TABLE agent_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    source           TEXT NOT NULL,           -- cli, slack, eval
    slack_channel    TEXT,
    slack_thread_ts  TEXT,
    slack_user       TEXT,
    question         TEXT NOT NULL,
    answer           TEXT,
    status           TEXT NOT NULL DEFAULT 'running',  -- running, ok, error
    error            TEXT,
    model            TEXT,
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_usd         NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms       INTEGER
);

CREATE TABLE agent_steps (
    id             BIGSERIAL PRIMARY KEY,
    run_id         UUID NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    step_index     INTEGER NOT NULL,
    step_type      TEXT NOT NULL,             -- llm, tool
    tool_name      TEXT,
    tool_input     JSONB,
    tool_output    JSONB,
    is_error       BOOLEAN NOT NULL DEFAULT false,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cost_usd       NUMERIC(12, 6),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_index)
);
