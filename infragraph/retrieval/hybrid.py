"""
Hybrid retrieval + answer synthesis.

Flow:
  1. Semantic search → Qdrant (top-k chunks)
  2. Entity lookup   → Neo4j (graph context)
  3. Fuse context    → assemble prompt
  4. Answer          → deepseek/deepseek-v3 via OpenRouter
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infragraph.config import get_settings
from infragraph.retrieval.bm25 import BM25Index, rrf_merge
from infragraph.retrieval.graph import GraphRetriever
from infragraph.retrieval.semantic import SemanticRetriever

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert infrastructure assistant. "
    "Answer questions using ONLY the context provided below. "
    "If the context is insufficient, say so clearly. "
    "Be concise and cite specific sources (file paths) when available."
)


class HybridRetriever:
    def __init__(self) -> None:
        self._semantic = SemanticRetriever()
        self._graph    = GraphRetriever()
        self._bm25: BM25Index | None = None
        cfg = get_settings()
        self._model    = cfg.chat_model
        self._base_url = cfg.openrouter_base_url.rstrip("/")
        self._headers  = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DataOpsFusion/infragraph",
            "X-Title": "infragraph",
        }

    def _get_bm25(self) -> BM25Index:
        if self._bm25 is None:
            from infragraph.storage.qdrant import QdrantClient
            self._bm25 = BM25Index()
            self._bm25.build_from_qdrant(QdrantClient())
        return self._bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        graph_depth: int = 2,
    ) -> dict[str, Any]:
        """Return raw context from both sources without calling the LLM."""
        semantic_hits = self._semantic.search(query, top_k=top_k)
        # BM25 lexical search + RRF fusion
        bm25_hits = self._get_bm25().search(query, top_k=top_k)
        fused = rrf_merge(semantic_hits, bm25_hits)
        semantic_hits = [
            {"score": h["rrf_score"], "payload": h["payload"]}
            for h in fused[:top_k]
        ]

        # Extract entity names from top chunks to prime graph lookup
        graph_hits: list[dict] = []
        entity_names: list[str] = []
        for hit in semantic_hits[:3]:
            payload = hit.get("payload", {})
            text = payload.get("text", "")
            # Simple heuristic: capitalised words > 3 chars as entity candidates
            import re
            candidates = re.findall(r"\b[A-Z][a-z]{2,}(?:-[A-Za-z]+)*\b", text)
            entity_names.extend(candidates[:5])

        seen_ids: set[str] = set()
        for name in set(entity_names):
            for match in self._graph.find(name, limit=2):
                node = match.get("node", {})
                eid = node.get("id")
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    nbrs = self._graph.neighbours(eid, depth=graph_depth)
                    graph_hits.append({"entity": node, "neighbours": nbrs})

        return {"semantic": semantic_hits, "graph": graph_hits}

    def _build_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []

        parts.append("## Document chunks (ranked by relevance)")
        for i, hit in enumerate(ctx["semantic"], 1):
            p = hit.get("payload", {})
            score = hit.get("score", 0)
            parts.append(
                f"[{i}] ({p.get('doc_type', '?')}) {p.get('source_path', '')}"
                f" [score={score:.2f}]\n{p.get('text', '')}"
            )

        if ctx["graph"]:
            parts.append("\n## Graph context (entities & relationships)")
            for g in ctx["graph"]:
                e = g["entity"]
                parts.append(
                    f"Entity: {e.get('name')} ({e.get('type', '?')}) "
                    f"— aliases: {e.get('aliases', [])}"
                )
                for nbr_group in g["neighbours"][:3]:
                    for node in nbr_group.get("nodes", []):
                        if node.get("id") != e.get("id"):
                            parts.append(f"  → {node.get('name')} ({node.get('type', '?')})")

        return "\n\n".join(parts)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
    )
    def answer(
        self,
        query: str,
        top_k: int = 20,
        graph_depth: int = 2,
    ) -> dict[str, Any]:
        ctx = self.retrieve(query, top_k=top_k, graph_depth=graph_depth)
        context_str = self._build_context(ctx)

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}",
            },
        ]

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json={"model": self._model, "messages": messages, "temperature": 0.2},
            )
            resp.raise_for_status()

        answer_text = resp.json()["choices"][0]["message"]["content"]
        sources = list({
            hit["payload"].get("source_path", "")
            for hit in ctx["semantic"]
            if hit.get("payload", {}).get("source_path")
        })

        return {
            "answer": answer_text,
            "sources": sources,
            "semantic_hits": len(ctx["semantic"]),
            "graph_entities": len(ctx["graph"]),
        }
