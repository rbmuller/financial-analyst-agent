-- Privileges for the agent's SQL tool. This is a group role without login:
-- the login user (name and password come from .env) is created by
-- db/migrate.py and made a member of it, so no secret lives in SQL files.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_readonly') THEN
        CREATE ROLE agent_readonly NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO agent_readonly;

-- Allow-list: only the fundamentals. Nothing else (filings, agent_runs, ...)
-- is readable through the SQL tool.
GRANT SELECT ON companies, financial_facts TO agent_readonly;
