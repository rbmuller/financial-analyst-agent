"""Earlier questions and answers of a Slack thread, read back from agent_runs.

The API is stateless, so a follow-up like "and for Apple?" only makes sense if
the previous exchange is sent again. The trace table already has it.
"""

from collections.abc import Callable

import psycopg

from agent.tracing import RunStatus

MAX_EXCHANGES = 3

type Exchange = tuple[str, str]


def thread_history_loader(
    connect: Callable[[], psycopg.Connection], limit: int = MAX_EXCHANGES
) -> Callable[[str, str], list[Exchange]]:
    def load(channel: str, thread_ts: str) -> list[Exchange]:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer
                FROM agent_runs
                WHERE slack_channel = %s AND slack_thread_ts = %s
                  AND status = %s AND answer IS NOT NULL
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (channel, thread_ts, RunStatus.OK.value, limit),
            ).fetchall()
        return [(row[0], row[1]) for row in reversed(rows)]

    return load
