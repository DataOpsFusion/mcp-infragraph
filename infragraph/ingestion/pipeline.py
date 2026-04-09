"""
Ingestion pipeline — orchestrates:
  scan → chunk → embed → upsert Qdrant → extract → normalize → upsert Neo4j
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from infragraph.embedding import embed_texts
from infragraph.extraction.extractor import Extractor
from infragraph.ingestion.chunker import Chunk, chunk_document
from infragraph.ingestion.scanner import scan_paths
from infragraph.normalization.writer import GraphWriter
from infragraph.storage.neo4j import Neo4jClient
from infragraph.storage.qdrant import QdrantClient
from infragraph.ingestion.context_header import ContextHeaderGenerator

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    files: int = 0
    chunks: int = 0
    vectors_written: int = 0
    entities_written: int = 0
    relations_written: int = 0


class IngestionPipeline:
    def __init__(self) -> None:
        self._neo4j   = Neo4jClient()
        self._qdrant  = QdrantClient()
        self._extract = Extractor()
        self._ctx_headers = ContextHeaderGenerator()

    def bootstrap(self) -> None:
        """One-time schema setup on both databases."""
        self._neo4j.bootstrap_schema()
        self._qdrant.bootstrap_collection()
        log.info("schema bootstrap complete")

    def run(
        self,
        paths: list[str],
        include_ocr: bool = True,
        progress_cb: Callable[[str], None] | None = None,
    ) -> IngestStats:
        stats = IngestStats()
        if not paths:
            raise ValueError("paths must not be empty")

        def _progress(msg: str) -> None:
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        # ── 1. scan ───────────────────────────────────────────────────────────
        _progress(f"scanning {len(paths)} path(s)...")
        file_texts = scan_paths(paths, include_ocr=include_ocr)
        stats.files = len(file_texts)
        _progress(f"found {stats.files} files")

        # ── 2. chunk ──────────────────────────────────────────────────────────
        all_chunks: list[Chunk] = []
        for src_path, text in file_texts:
            chunks = chunk_document(text, src_path)
            all_chunks.extend(chunks)
        stats.chunks = len(all_chunks)
        _progress(f"produced {stats.chunks} chunks")

        # ── 3. add context headers, embed + upsert Qdrant (batch) ────────────
        # Group chunks by source so the full document text is available for
        # header generation (provider-side caching benefits same-doc chunks).
        from collections import defaultdict
        by_source: dict[str, list] = defaultdict(list)
        source_texts: dict[str, str] = dict(file_texts)
        for chunk in all_chunks:
            by_source[chunk.source_path].append(chunk)

        contextualised: list[Chunk] = []
        for src_path, chunks in by_source.items():
            doc_text = source_texts.get(src_path, "")
            for chunk in chunks:
                header = self._ctx_headers.get_header(chunk.text, doc_text)
                if header:
                    chunk.text = f"{header}\n\n{chunk.text}"
                contextualised.append(chunk)

        all_chunks = contextualised
        _progress(f"context headers added to {stats.chunks} chunks")

        batch_size = 64
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = embed_texts(texts)
            payloads = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_path": c.source_path,
                    "doc_type": c.doc_type,
                    "index": c.index,
                    **(c.metadata if c.metadata else {}),
                }
                for c in batch
            ]
            self._qdrant.upsert_chunks(payloads, vectors)
            stats.vectors_written += len(batch)

        _progress(f"wrote {stats.vectors_written} vectors to Qdrant")

        # ── 4. extract + normalize → Neo4j ────────────────────────────────────
        writer = GraphWriter(self._neo4j)
        for chunk in all_chunks:
            try:
                result = self._extract.extract(
                    chunk.text,
                    source_doc=chunk.source_path,
                    chunk_id=chunk.chunk_id,
                )
                written = writer.write(result)
                stats.entities_written  += written["entities"]
                stats.relations_written += written["relations"]
            except Exception as exc:
                log.warning("graph extraction skipped for %s: %s", chunk.chunk_id, exc)

        _progress(
            f"graph: {stats.entities_written} entities, "
            f"{stats.relations_written} relations"
        )
        return stats
