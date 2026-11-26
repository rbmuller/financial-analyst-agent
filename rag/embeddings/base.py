"""The contract every embedding backend implements.

The rest of the code depends only on this Protocol, so swapping the local model
for a paid API means writing one class and changing one setting.
"""

from typing import Protocol

type Vector = list[float]


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed passages to be stored and searched."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query. Some models encode queries differently from passages."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens the model would read for this text (used to size chunks)."""
        ...


def to_pgvector(vector: Vector) -> str:
    """Text form accepted by pgvector: '[0.1,0.2,...]' (cast with ::vector)."""
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"
