"""Qdrant client — collection bootstrap, upsert, semantic search."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.http import models as qm

from infragraph.config import get_settings

log = logging.getLogger(__name__)


class QdrantClient:
    def __init__(self) -> None:
        cfg = get_settings()
        kwargs: dict[str, Any] = {
            "host": cfg.qdrant_host,
            "port": cfg.qdrant_port,
            "prefer_grpc": False,
        }
        if cfg.qdrant_api_key:
            kwargs["api_key"] = cfg.qdrant_api_key

        self._client = _QdrantClient(**kwargs)
        self._collection = cfg.qdrant_collection
        self._dim = cfg.embed_dim

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def verify(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    # ── collection bootstrap ──────────────────────────────────────────────────

    def bootstrap_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._dim,
                    distance=qm.Distance.COSINE,
                ),
            )
            # Payload indexes for fast filtering
            for field in ("doc_type", "source_path", "entity_ids"):
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            log.info("created Qdrant collection %s (dim=%d)", self._collection, self._dim)
        else:
            log.debug("Qdrant collection %s already exists", self._collection)

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        """
        chunks: list of payload dicts, each must have at minimum:
            chunk_id, text, source_path, doc_type
        vectors: matching list of embedding vectors
        """
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
                vector=v,
                payload=c,
            )
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def delete_by_source(self, source_path: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source_path",
                            match=qm.MatchValue(value=source_path),
                        )
                    ]
                )
            ),
        )

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        doc_type_filter: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        filt = None
        if doc_type_filter:
            filt = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="doc_type",
                        match=qm.MatchValue(value=doc_type_filter),
                    )
                ]
            )
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                query_filter=filt,
                score_threshold=score_threshold,
                with_payload=True,
            )
            results = response.points
        else:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                query_filter=filt,
                score_threshold=score_threshold,
                with_payload=True,
            )
        return [
            {"score": r.score, "payload": r.payload, "id": str(r.id)}
            for r in results
        ]

    def collection_info(self) -> dict[str, Any]:
        info = self._client.get_collection(self._collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
