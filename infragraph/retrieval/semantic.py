"""Semantic retrieval — embed query and search Qdrant."""
from __future__ import annotations

from typing import Any

from infragraph.embedding import embed_query


class SemanticRetriever:
    def __init__(self) -> None:
        from infragraph.storage.qdrant import QdrantClient
        self._qdrant = QdrantClient()

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        vector = embed_query(query)
        return self._qdrant.search(
            vector=vector,
            top_k=top_k,
            doc_type_filter=doc_type,
            score_threshold=score_threshold,
        )
