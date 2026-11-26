"""Local, free embeddings with sentence-transformers (default: BGE-small)."""

from typing import Any

from rag.embeddings.base import Vector

# BGE models are trained with this instruction on the query side only, for
# short-query -> long-passage retrieval. Passages are embedded as they are.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_MODEL_MARKER = "bge-"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        # Imported lazily: loading PyTorch takes seconds and most commands do not need it.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name, device="cpu")
        self._query_prefix = BGE_QUERY_INSTRUCTION if BGE_MODEL_MARKER in model_name else ""

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[Vector]:
        # Normalized vectors have length 1, so cosine similarity is a plain dot product.
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return self._encode(texts)

    def embed_query(self, text: str) -> Vector:
        return self._encode([self._query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.tokenize(text))
