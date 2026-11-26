"""Split a filing section into chunks that fit the embedding model.

An embedding model reads a limited number of tokens (512 for BGE-small) and
squeezes whatever it reads into a single vector, so a whole MD&A in one vector
would be both truncated and too vague to match a specific question. Chunks are
the retrieval unit: small enough to be about one thing, large enough to make
sense on their own.

Strategy: pack whole paragraphs up to `max_tokens`; consecutive chunks share
`overlap_tokens` of trailing text so a sentence cut at a boundary still appears
with its context in one of them.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

type TokenCounter = Callable[[str], int]

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
PARAGRAPH_SEPARATOR = "\n"

# Exhibit indexes are lists of file names: noise for retrieval.
SKIPPED_ITEMS = frozenset({"Item 15", "Item 16", "Part II, Item 6"})


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    token_count: int


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int


def _split_words(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[_Unit]:
    """Last resort for a sentence longer than a chunk (e.g. a flattened table)."""
    units: list[_Unit] = []
    words: list[str] = []
    tokens = 0
    for word in text.split():
        word_tokens = count_tokens(word)
        if words and tokens + word_tokens > max_tokens:
            units.append(_Unit(" ".join(words), tokens))
            words, tokens = [], 0
        words.append(word)
        tokens += word_tokens
    if words:
        units.append(_Unit(" ".join(words), tokens))
    return units


def _split_units(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[_Unit]:
    """Paragraphs, falling back to sentences and then words when one is too long."""
    units: list[_Unit] = []
    for paragraph in filter(None, (line.strip() for line in text.split(PARAGRAPH_SEPARATOR))):
        tokens = count_tokens(paragraph)
        if tokens <= max_tokens:
            units.append(_Unit(paragraph, tokens))
            continue
        for sentence in SENTENCE_BOUNDARY.split(paragraph):
            sentence_tokens = count_tokens(sentence)
            if sentence_tokens <= max_tokens:
                units.append(_Unit(sentence, sentence_tokens))
            else:
                units.extend(_split_words(sentence, count_tokens, max_tokens))
    return units


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing units of a finished chunk that are repeated at the start of the next."""
    tail: list[_Unit] = []
    tokens = 0
    for unit in reversed(units):
        if tokens + unit.tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        tokens += unit.tokens
    return tail


def chunk_text(
    text: str, count_tokens: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[_Unit] = []

    def emit() -> None:
        chunks.append(
            Chunk(
                index=len(chunks),
                content=PARAGRAPH_SEPARATOR.join(unit.text for unit in current),
                token_count=sum(unit.tokens for unit in current),
            )
        )

    for unit in _split_units(text, count_tokens, max_tokens):
        if current and sum(u.tokens for u in current) + unit.tokens > max_tokens:
            emit()
            tail = _overlap_tail(current, overlap_tokens)
            fits = sum(u.tokens for u in tail) + unit.tokens <= max_tokens
            current = tail if fits else []
        current.append(unit)

    if current:
        emit()
    return chunks
