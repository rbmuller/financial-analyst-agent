"""What happens when someone asks the bot a question (independent of Bolt).

Flow: post a "thinking" placeholder in the thread, run the agent while updating
the placeholder with what it is doing, then replace it with the final answer.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from agent.loop import AgentResult
from agent.tracing import RunOrigin, Step
from slack_app.formatting import (
    TEXT_ERROR,
    TEXT_THINKING,
    fallback_text,
    progress_text,
    result_blocks,
)

logger = logging.getLogger(__name__)

ORIGIN_SLACK = "slack"
MENTION = re.compile(r"<@[A-Z0-9]+>")
# chat.update is rate limited per channel; progress is cosmetic, so it is throttled.
PROGRESS_MIN_INTERVAL_SECONDS = 1.5
TEXT_EMPTY_QUESTION = (
    "Mande uma pergunta junto com a menção, por exemplo: "
    "_Qual foi a receita da Apple no último ano fiscal?_"
)

type Exchange = tuple[str, str]  # (question, answer)


class SlackClient(Protocol):
    def chat_postMessage(self, **kwargs: Any) -> Any: ...  # noqa: N802 (Slack API name)

    def chat_update(self, **kwargs: Any) -> Any: ...


class AnsweringAgent(Protocol):
    def run(
        self,
        question: str,
        origin: RunOrigin,
        on_step: Callable[[Step], None] | None = None,
        history: list[Exchange] | None = None,
    ) -> AgentResult: ...


def extract_question(text: str) -> str:
    return " ".join(MENTION.sub(" ", text).split())


class QuestionHandler:
    def __init__(
        self,
        agent: AnsweringAgent,
        load_history: Callable[[str, str], list[Exchange]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent = agent
        self._load_history = load_history
        self._clock = clock

    def handle(self, event: dict[str, Any], client: SlackClient) -> None:
        channel = event["channel"]
        # Reply in the thread the question came from, or start one under it.
        thread_ts = event.get("thread_ts") or event["ts"]
        question = extract_question(event.get("text", ""))
        if not question:
            client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=TEXT_EMPTY_QUESTION)
            return

        placeholder = client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=TEXT_THINKING
        )
        placeholder_ts = placeholder["ts"]
        origin = RunOrigin(ORIGIN_SLACK, channel, thread_ts, event.get("user"))

        try:
            result = self._agent.run(
                question,
                origin,
                on_step=self._progress_updater(client, channel, placeholder_ts),
                history=self._load_history(channel, thread_ts),
            )
        except Exception:
            # The agent already recorded the failure in agent_runs; the user must
            # not be left looking at "thinking" forever.
            logger.exception("Agent run failed for a Slack question")
            client.chat_update(channel=channel, ts=placeholder_ts, text=TEXT_ERROR)
            return

        client.chat_update(
            channel=channel,
            ts=placeholder_ts,
            text=fallback_text(result),
            blocks=result_blocks(result),
        )

    def _progress_updater(
        self, client: SlackClient, channel: str, placeholder_ts: str
    ) -> Callable[[Step], None]:
        last_update = self._clock()
        last_text = TEXT_THINKING

        def on_step(step: Step) -> None:
            nonlocal last_update, last_text
            text = progress_text(step)
            now = self._clock()
            if text == last_text or now - last_update < PROGRESS_MIN_INTERVAL_SECONDS:
                return
            try:
                client.chat_update(channel=channel, ts=placeholder_ts, text=text)
            except Exception:  # progress is best effort; never fail the answer over it
                logger.warning("Could not update the progress message", exc_info=True)
            last_update, last_text = now, text

        return on_step
