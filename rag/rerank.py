"""Optional second-stage ranking with a cross-encoder.

Retrieval compares a query vector with passage vectors computed independently
(a "bi-encoder"): fast, because passage vectors are precomputed, but the model
never sees query and passage together. A cross-encoder reads the pair jointly
and outputs one relevance score: far more precise, but it must run once per
candidate at query time, so it is only applied to the few dozen passages that
retrieval already selected.
"""

from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query (higher is better)."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        # Imported lazily: loading PyTorch takes seconds and most commands do not need it.
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages], show_progress_bar=False
        )
        return [float(score) for score in scores]
