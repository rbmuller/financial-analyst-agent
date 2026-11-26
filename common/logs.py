"""Logging setup shared by the command-line entry points."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
NOISY_LIBRARIES = (
    "httpx",
    "httpcore",
    "sentence_transformers",
    "huggingface_hub",
    "transformers",
    "anthropic",
)


class QuietLibraries(logging.Filter):
    """Drop routine messages from model and HTTP libraries (they hide the actual output).

    A handler filter is used because these libraries reset their own logger
    levels when imported, which would undo a plain setLevel().
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR or not record.name.startswith(NOISY_LIBRARIES)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    for handler in logging.getLogger().handlers:
        handler.addFilter(QuietLibraries())
