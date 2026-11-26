"""Record every agent run and each of its steps in Postgres.

An agent decides its own path, so "what did it do and why" cannot be read from
the code. The trace answers it after the fact: which tools were called with
which arguments, how long each step took, and what it cost.
"""

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from agent.pricing import TokenUsage

OUTPUT_PREVIEW_CHARS = 2000


class RunStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    REFUSED = "refused"
    STEP_LIMIT = "step_limit"
    TRUNCATED = "truncated"


class StepType(StrEnum):
    LLM = "llm"
    TOOL = "tool"


@dataclass(frozen=True)
class RunOrigin:
    source: str  # cli, slack, eval
    slack_channel: str | None = None
    slack_thread_ts: str | None = None
    slack_user: str | None = None


@dataclass(frozen=True)
class Step:
    type: StepType
    latency_ms: int
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    output: str = ""
    is_error: bool = False
    usage: TokenUsage | None = None
    cost_usd: Decimal | None = None


class Tracer(Protocol):
    def start_run(self, question: str, model: str, origin: RunOrigin) -> str: ...

    def record_step(self, run_id: str, index: int, step: Step) -> None: ...

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        answer: str | None,
        error: str | None,
        usage: TokenUsage,
        cost_usd: Decimal | None,
        latency_ms: int,
    ) -> None: ...


def preview(text: str) -> dict[str, Any]:
    """Tool outputs can be large; keep the start and the full size."""
    return {"chars": len(text), "preview": text[:OUTPUT_PREVIEW_CHARS]}


class PostgresTracer:
    """Writes on its own autocommit connection, so a crashed run still leaves its steps."""

    def __init__(self, conn: psycopg.Connection) -> None:
        conn.autocommit = True
        self._conn = conn

    def start_run(self, question: str, model: str, origin: RunOrigin) -> str:
        row = self._conn.execute(
            """
            INSERT INTO agent_runs (source, slack_channel, slack_thread_ts, slack_user,
                                    question, model)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                origin.source,
                origin.slack_channel,
                origin.slack_thread_ts,
                origin.slack_user,
                question,
                model,
            ),
        ).fetchone()
        assert row is not None
        run_id: UUID = row[0]
        return str(run_id)

    def record_step(self, run_id: str, index: int, step: Step) -> None:
        usage = step.usage
        self._conn.execute(
            """
            INSERT INTO agent_steps (run_id, step_index, step_type, tool_name, tool_input,
                                     tool_output, is_error, latency_ms, input_tokens,
                                     output_tokens, cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                index,
                step.type.value,
                step.tool_name,
                Jsonb(step.tool_input, dumps=_dumps) if step.tool_input is not None else None,
                Jsonb(preview(step.output)),
                step.is_error,
                step.latency_ms,
                usage.total_input_tokens if usage else None,
                usage.output_tokens if usage else None,
                step.cost_usd,
            ),
        )

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        answer: str | None,
        error: str | None,
        usage: TokenUsage,
        cost_usd: Decimal | None,
        latency_ms: int,
    ) -> None:
        self._conn.execute(
            """
            UPDATE agent_runs
            SET finished_at = now(), status = %s, answer = %s, error = %s,
                input_tokens = %s, output_tokens = %s, cost_usd = COALESCE(%s, 0),
                latency_ms = %s
            WHERE id = %s
            """,
            (
                status.value,
                answer,
                error,
                usage.total_input_tokens,
                usage.output_tokens,
                cost_usd,
                latency_ms,
                run_id,
            ),
        )


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)
