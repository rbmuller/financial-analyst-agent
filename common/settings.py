"""Application settings, loaded from environment variables (and .env locally)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: SecretStr

    agent_ro_user: str
    agent_ro_password: SecretStr

    # SEC EDGAR: a User-Agent with contact details is mandatory. Optional here
    # so that commands that never call the SEC (migrations, tests) still run.
    sec_user_agent: str | None = None
    sec_requests_per_second: float = 5.0  # SEC's published ceiling is 10
    filings_lookback_years: int = 3

    # Retrieval. The embedding dimension must match the `chunks.embedding` column.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    chunk_max_tokens: int = 400  # the model reads 512; the rest is room for the context header
    chunk_overlap_tokens: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    search_top_k: int = 8
    search_candidates: int = 40  # per retriever, before fusion

    # Agent. ANTHROPIC_API_KEY is read from the environment by the SDK itself.
    agent_model: str = "claude-opus-5-5"
    agent_effort: str = "medium"  # low | medium | high | xhigh | max
    agent_max_tokens: int = 16000
    agent_max_steps: int = 12  # LLM calls per question, a guard against loops
    agent_sql_max_rows: int = 200

    # Evals. The judge is a different model from the agent it grades.
    eval_judge_model: str = "claude-sonnet-5-5"

    # Slack (Socket Mode). Optional so that every other command runs without them.
    slack_bot_token: SecretStr | None = None  # xoxb-...
    slack_app_token: SecretStr | None = None  # xapp-...
    slack_rerank: bool = False

    config_dir: Path = Path("config")
    data_dir: Path = Path("data")  # raw download cache, gitignored


@lru_cache
def get_settings() -> Settings:
    return Settings()
