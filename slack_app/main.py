"""Slack bot entry point (Socket Mode).

Socket Mode means the bot opens an outbound WebSocket to Slack and receives
events through it, so it needs no public URL and runs behind any firewall.

Usage: python -m slack_app.main
"""

import logging
import sys
from functools import partial
from typing import Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent.factory import build_agent
from common.db import DbRole, connect
from common.logs import configure_logging
from common.settings import get_settings
from slack_app.handler import QuestionHandler
from slack_app.history import thread_history_loader

logger = logging.getLogger("slack_app")

EVENT_APP_MENTION = "app_mention"
EVENT_MESSAGE = "message"
CHANNEL_TYPE_DIRECT_MESSAGE = "im"
EXIT_CONFIG_ERROR = 2


def is_direct_question(event: dict[str, Any]) -> bool:
    """A plain message a person typed in a DM with the bot (not an edit, not a bot)."""
    return (
        event.get("channel_type") == CHANNEL_TYPE_DIRECT_MESSAGE
        and event.get("subtype") is None
        and event.get("bot_id") is None
    )


def main() -> int:
    configure_logging()
    settings = get_settings()
    if settings.slack_bot_token is None or settings.slack_app_token is None:
        print(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in .env. See docs/SLACK_SETUP.md.",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    logger.info("Loading models…")
    handler = QuestionHandler(
        build_agent(settings, rerank=settings.slack_rerank),
        thread_history_loader(partial(connect, DbRole.ADMIN, settings)),
    )
    app = App(token=settings.slack_bot_token.get_secret_value())

    # Bolt acknowledges events to Slack right away and runs listeners in a thread
    # pool, so a 30-second agent run does not trigger Slack's 3-second retry.
    @app.event(EVENT_APP_MENTION)
    def on_mention(event: dict[str, Any], client: Any) -> None:
        handler.handle(event, client)

    @app.event(EVENT_MESSAGE)
    def on_message(event: dict[str, Any], client: Any) -> None:
        if is_direct_question(event):
            handler.handle(event, client)

    logger.info("Connected. Mention the bot in a channel or send it a direct message.")
    SocketModeHandler(app, settings.slack_app_token.get_secret_value()).start()  # type: ignore[no-untyped-call]
    return 0


if __name__ == "__main__":
    sys.exit(main())
