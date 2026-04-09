"""
BM25 lexical retriever.

Complements Qdrant semantic search for exact-match queries:
IPs (192.168.0.50), service names (nginx, folo), port numbers (5020),
error codes, etc. — tokens where semantic embeddings are weak.

Usage:
  index = BM25Index()
  index.build(chunks)            # list of payload dicts from Qdrant
  results = index.search(query)  # returns same format as SemanticRetriever
"""
from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)


def rrf_merge(
    semantic: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion: merge two ranked lists into one.
    Items appearing in both lists are boosted.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(semantic):
        cid = hit["payload"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "payload": hit["payload"], "score": hit.get("score", 0)}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, hit in enumerate(bm25):
        cid = hit["payload"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "payload": hit["payload"], "score": hit.get("score", 0)}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)

    return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)


class BM25Index:
    """In-memory BM25 index built from a list of chunk payload dicts."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of chunk payload dicts.
        Each dict must have at minimum: chunk_id, text.
        """
        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        log.info("BM25 index built from %d chunks", len(chunks))

    def build_from_qdrant(self, qdrant_client: Any) -> None:
        """
        Scroll all points from Qdrant and build the index.
        Called once at startup — no extra storage needed.
        """
        all_chunks: list[dict[str, Any]] = []
        offset = None

        try:
            while True:
                response = qdrant_client._client.scroll(
                    collection_name=qdrant_client._collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = response
                for p in points:
                    if p.payload:
                        all_chunks.append(p.payload)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            log.error("Failed to scroll Qdrant for BM25 build: %s", exc)
            raise

        if not all_chunks:
            log.warning("BM25 index: no chunks found in Qdrant collection")

        self.build(all_chunks)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by BM25 score."""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(raw_scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {"score": float(score), "payload": chunk}
            for score, chunk in ranked[:top_k]
            if score > 0
        ]
