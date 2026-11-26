from collections.abc import Callable
from decimal import Decimal
from typing import Any

from agent.loop import AgentResult
from agent.pricing import TokenUsage
from agent.sources import Source, SourceKind, strip_citations
from agent.tracing import RunOrigin, RunStatus, Step, StepType
from slack_app.formatting import (
    PROGRESS_BY_TOOL,
    PROGRESS_WRITING,
    TEXT_ERROR,
    TEXT_THINKING,
    result_blocks,
    split_text,
    to_mrkdwn,
)
from slack_app.handler import TEXT_EMPTY_QUESTION, QuestionHandler, extract_question
from slack_app.main import is_direct_question

CHANNEL = "C123"
MENTION_EVENT = {
    "channel": CHANNEL,
    "ts": "100.1",
    "user": "U1",
    "text": "<@UBOT> como vai a Nike?",
}


def make_result(**overrides: Any) -> AgentResult:
    values: dict[str, Any] = {
        "run_id": "12345678-aaaa-bbbb-cccc-000000000000",
        "status": RunStatus.OK,
        "answer": "Margem caiu **1,3 p.p.** [S1].",
        "sources": [Source("S1", SourceKind.FILING, "NIKE 10-Q <Q3>", "https://sec.test/doc")],
        "usage": TokenUsage(input_tokens=10, output_tokens=5),
        "cost_usd": Decimal("0.1706"),
        "latency_ms": 30600,
        "steps": 20,
    }
    return AgentResult(**{**values, **overrides})


class FakeSlack:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:  # noqa: N802
        self.calls.append(("post", kwargs))
        return {"ts": "200.2"}

    def chat_update(self, **kwargs: Any) -> None:
        self.calls.append(("update", kwargs))


class FakeAgent:
    def __init__(self, result: AgentResult | None = None, steps: list[Step] | None = None) -> None:
        self._result = result
        self._steps = steps or []
        self.received: dict[str, Any] = {}

    def run(
        self,
        question: str,
        origin: RunOrigin,
        on_step: Callable[[Step], None] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> AgentResult:
        self.received = {"question": question, "origin": origin, "history": history}
        for step in self._steps:
            assert on_step is not None
            on_step(step)
        if self._result is None:
            raise RuntimeError("boom")
        return self._result


def no_history(channel: str, thread_ts: str) -> list[tuple[str, str]]:
    return []


# --- formatting --------------------------------------------------------------


def test_markdown_becomes_slack_mrkdwn() -> None:
    markdown = "\n".join(
        [
            "## Resumo",
            "A margem **caiu** <2 p.p.> & [o 10-Q](https://sec.test/a) explica [S1].",
            "- primeiro",
            "  * segundo",
        ]
    )

    assert to_mrkdwn(markdown).split("\n") == [
        "*Resumo*",
        "A margem *caiu* &lt;2 p.p.&gt; &amp; <https://sec.test/a|o 10-Q> explica [S1].",
        "• primeiro",
        "  • segundo",
    ]


def test_tables_become_a_monospace_block() -> None:
    markdown = "Antes\n| Tri | **Margem** |\n|---|---|\n| Q3 | 40,2% |\nDepois"

    assert to_mrkdwn(markdown).split("\n") == [
        "Antes",
        "```",
        "| Tri | Margem |",
        "| Q3 | 40,2% |",
        "```",
        "Depois",
    ]


def test_split_text_respects_the_block_limit_without_losing_text() -> None:
    text = "\n".join(["a" * 40, "b" * 40, "c" * 120, "d" * 10])

    pieces = split_text(text, limit=50)

    assert all(len(piece) <= 50 for piece in pieces)
    assert "".join(pieces).replace("\n", "") == text.replace("\n", "")
    assert split_text("short", limit=50) == ["short"]


def test_result_blocks_have_answer_sources_and_footer() -> None:
    blocks = result_blocks(make_result())

    assert [block["type"] for block in blocks] == ["section", "divider", "section", "context"]
    assert blocks[0]["text"]["text"] == "Margem caiu *1,3 p.p.* [S1]."
    assert blocks[2]["text"]["text"] == (
        "*Fontes*\n• `S1` <https://sec.test/doc|NIKE 10-Q &lt;Q3&gt;>"
    )
    assert blocks[3]["elements"][0]["text"] == "31s · 20 passos · US$ 0.17 · run `12345678`"


def test_result_blocks_warn_about_unknown_citations_and_bad_status() -> None:
    result = make_result(
        sources=[], unknown_citations=["S9"], status=RunStatus.STEP_LIMIT, cost_usd=None
    )

    blocks = result_blocks(result)

    assert [block["type"] for block in blocks] == ["section", "context", "context"]
    assert "S9" in blocks[1]["elements"][0]["text"]
    assert "limite de passos" in blocks[2]["elements"][0]["text"]
    assert "US$" not in blocks[2]["elements"][0]["text"]


def test_long_answers_are_split_into_several_blocks() -> None:
    blocks = result_blocks(make_result(answer="\n".join(["linha " + "x" * 90] * 80), sources=[]))

    sections = [block for block in blocks if block["type"] == "section"]
    assert len(sections) == 3
    assert all(len(block["text"]["text"]) <= 3000 for block in sections)


# --- handler -----------------------------------------------------------------


def test_extract_question_removes_mentions() -> None:
    assert extract_question("<@U0BOT>  qual a receita\n da <@U2> Apple?") == (
        "qual a receita da Apple?"
    )


def test_answers_in_a_thread_by_replacing_the_thinking_placeholder() -> None:
    slack, agent = FakeSlack(), FakeAgent(make_result())

    QuestionHandler(agent, no_history).handle(MENTION_EVENT, slack)

    (post_kind, post), (update_kind, update) = slack.calls
    assert (post_kind, post) == (
        "post",
        {"channel": CHANNEL, "thread_ts": "100.1", "text": TEXT_THINKING},
    )
    assert update_kind == "update"
    assert (update["channel"], update["ts"]) == (CHANNEL, "200.2")
    assert update["text"] == "Margem caiu **1,3 p.p.** [S1]."
    assert update["blocks"][0]["type"] == "section"
    assert agent.received["question"] == "como vai a Nike?"
    assert agent.received["origin"] == RunOrigin("slack", CHANNEL, "100.1", "U1")


def test_follow_up_stays_in_its_thread_and_gets_the_thread_history() -> None:
    slack, agent = FakeSlack(), FakeAgent(make_result())
    requested: list[tuple[str, str]] = []

    def history(channel: str, thread_ts: str) -> list[tuple[str, str]]:
        requested.append((channel, thread_ts))
        return [("pergunta anterior", "resposta anterior")]

    event = {**MENTION_EVENT, "ts": "150.5", "thread_ts": "100.1"}
    QuestionHandler(agent, history).handle(event, slack)

    assert slack.calls[0][1]["thread_ts"] == "100.1"
    assert requested == [(CHANNEL, "100.1")]
    assert agent.received["history"] == [("pergunta anterior", "resposta anterior")]


def test_empty_mention_gets_a_hint_and_does_not_run_the_agent() -> None:
    slack, agent = FakeSlack(), FakeAgent(make_result())

    QuestionHandler(agent, no_history).handle({**MENTION_EVENT, "text": "<@UBOT> "}, slack)

    assert [(kind, call["text"]) for kind, call in slack.calls] == [("post", TEXT_EMPTY_QUESTION)]
    assert agent.received == {}


def test_agent_failure_replaces_the_placeholder_with_an_error() -> None:
    slack = FakeSlack()

    QuestionHandler(FakeAgent(result=None), no_history).handle(MENTION_EVENT, slack)

    assert slack.calls[-1] == ("update", {"channel": CHANNEL, "ts": "200.2", "text": TEXT_ERROR})


def test_progress_updates_are_throttled_and_deduplicated() -> None:
    search = Step(StepType.TOOL, 10, tool_name="search_filings")
    sql = Step(StepType.TOOL, 10, tool_name="query_financials")
    model = Step(StepType.LLM, 10)
    times = iter([0.0, 2.0, 2.1, 5.0, 8.0, 11.0])
    slack = FakeSlack()
    agent = FakeAgent(make_result(), steps=[search, sql, sql, sql, model])

    QuestionHandler(agent, no_history, clock=lambda: next(times)).handle(MENTION_EVENT, slack)

    progress = [call["text"] for kind, call in slack.calls[1:-1] if kind == "update"]
    assert progress == [
        PROGRESS_BY_TOOL["search_filings"],  # t=2.0
        PROGRESS_BY_TOOL["query_financials"],  # t=5.0 (t=2.1 was too soon)
        PROGRESS_WRITING,  # t=11.0 (t=8.0 repeated the same text)
    ]


def test_only_plain_direct_messages_from_people_count_as_questions() -> None:
    assert is_direct_question({"channel_type": "im", "text": "oi"})
    assert not is_direct_question({"channel_type": "channel", "text": "oi"})
    assert not is_direct_question({"channel_type": "im", "subtype": "message_changed"})
    assert not is_direct_question({"channel_type": "im", "bot_id": "B1"})


# --- thread history in the agent ---------------------------------------------


def test_strip_citations_removes_markers_from_replayed_answers() -> None:
    assert (
        strip_citations("Caiu 2% [D1]. Tarifas [S1, S2] pesaram.") == "Caiu 2% . Tarifas pesaram."
    )
