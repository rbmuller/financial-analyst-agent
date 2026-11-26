"""Wire the agent's dependencies together (used by the CLI, Slack and evals)."""

from functools import partial

import psycopg

from agent.llm import ClaudeLLM
from agent.loop import Agent
from agent.pricing import Pricing
from agent.prompts import build_system_prompt
from agent.tools.base import Tool
from agent.tools.calculator import CalculatorTool
from agent.tools.query_financials import QueryFinancialsTool
from agent.tools.search_filings import SearchFilingsTool
from agent.tracing import PostgresTracer
from common.db import DbRole, connect
from common.settings import Settings
from ingestion.xbrl.metrics import load_metric_definitions
from rag.embeddings.local import SentenceTransformerEmbedder
from rag.rerank import CrossEncoderReranker


def covered_companies(conn: psycopg.Connection) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT ticker, name FROM companies ORDER BY ticker").fetchall()
    return [(row[0], row[1]) for row in rows]


def build_agent(settings: Settings, *, rerank: bool = False) -> Agent:
    """Load the models once and return an agent that can answer many questions."""
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model) if rerank else None
    metrics = [definition.name for definition in load_metric_definitions(settings.config_dir)]

    tools: list[Tool] = [
        SearchFilingsTool(
            partial(connect, DbRole.ADMIN, settings),
            embedder,
            reranker,
            candidates=settings.search_candidates,
        ),
        QueryFinancialsTool(
            partial(connect, DbRole.READONLY, settings), metrics, settings.agent_sql_max_rows
        ),
        CalculatorTool(),
    ]
    with connect(DbRole.ADMIN, settings) as conn:
        system_prompt = build_system_prompt(covered_companies(conn))

    return Agent(
        llm=ClaudeLLM(settings.agent_model, settings.agent_effort, settings.agent_max_tokens),
        tools=tools,
        system_prompt=system_prompt,
        tracer=PostgresTracer(connect(DbRole.ADMIN, settings)),
        pricing=Pricing.load(settings.config_dir),
        max_steps=settings.agent_max_steps,
    )
