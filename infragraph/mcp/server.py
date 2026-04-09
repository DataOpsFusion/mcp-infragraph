"""
Infragraph MCP server.

Exposes tools for:
  - semantic_search  : find relevant document chunks
  - graph_lookup     : look up entities and neighbours in Neo4j
  - hybrid_query     : full RAG answer (Qdrant + Neo4j + deepseek)
  - graph_stats      : entity/relation counts
  - session_*        : exact session events, checkpoints, artifacts, resume packets
  - ocr_extract_upload : OCR a single uploaded PDF/image
  - ocr_ingest_upload  : OCR + ingest a single uploaded PDF/image
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import chardet
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from infragraph.config import get_settings
from infragraph.ingestion.ocr import OCR_ALL_EXTS, is_ocr_candidate
from infragraph.retrieval.graph import GraphRetriever
from infragraph.retrieval.hybrid import HybridRetriever
from infragraph.retrieval.semantic import SemanticRetriever
from infragraph.session_store import SessionStore

log = logging.getLogger(__name__)

app = Server("infragraph")

# Lazy singletons — initialised on first use, protected by a lock for thread safety
_singleton_lock = threading.Lock()
_semantic: SemanticRetriever | None = None
_graph: GraphRetriever | None = None
_hybrid: HybridRetriever | None = None
_session_store: SessionStore | None = None
_pipeline: Any = None           # IngestionPipeline — imported lazily, bootstrapped once
_redis_client: Any = None       # None = not yet attempted; False = unavailable
_redis_available: bool = False  # True only after a successful ping

# ── Redis result cache ─────────────────────────────────────────────────────────
# TTLs for read-only idempotent tools (seconds)
_CACHE_TTL: dict[str, int] = {
    "knowledge_stats":  30,   # Entity/relation counts change frequently during ingestion
    "lookup_entity":    60,   # Graph lookups are stable but neighbours may be updated
    "search_knowledge": 120,  # Semantic results are stable; embeddings rarely change mid-session
    "deep_search":      120,  # Hybrid RAG results are expensive — cache aggressively
}


def _get_redis() -> Any | None:
    """Lazy Redis client. Returns None if REDIS_URL is not set or connection fails."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    url = os.environ.get("REDIS_URL", "")
    if not url:
        return None
    with _singleton_lock:
        if _redis_client is not None:
            return _redis_client if _redis_available else None
        try:
            import redis as _redis_lib
            client = _redis_lib.from_url(
                url, decode_responses=True,
                socket_connect_timeout=5,   # was 2s — too aggressive on slow LANs
                socket_timeout=10,
            )
            client.ping()
            _redis_client = client
            _redis_available = True
            log.info("Redis cache connected: %s", url)
        except Exception as exc:
            log.warning("Redis cache unavailable (%s: %s) — running without cache", type(exc).__name__, exc)
            _redis_client = False   # sentinel: don't retry
            _redis_available = False
    return _redis_client if _redis_available else None


def _cache_key(tool: str, args: dict) -> str:
    payload = tool + ":" + json.dumps(args, sort_keys=True)
    # Use full 64-char SHA-256 hex digest (was truncated to 20 chars — increased collision risk)
    return "infragraph:cache:" + tool + ":" + hashlib.sha256(payload.encode()).hexdigest()


def _rcache_get(tool: str, args: dict) -> str | None:
    if tool not in _CACHE_TTL:
        return None
    r = _get_redis()
    if not r:
        return None
    try:
        return r.get(_cache_key(tool, args))
    except Exception as exc:
        log.warning("Redis cache get failed (%s: %s)", type(exc).__name__, exc)
        return None


def _rcache_set(tool: str, args: dict, value: str) -> None:
    if tool not in _CACHE_TTL:
        return
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(_cache_key(tool, args), _CACHE_TTL[tool], value)
    except Exception as exc:
        log.warning("Redis cache set failed (%s: %s)", type(exc).__name__, exc)


def _get_semantic() -> SemanticRetriever:
    global _semantic
    if _semantic is None:
        with _singleton_lock:
            if _semantic is None:
                _semantic = SemanticRetriever()
    return _semantic


def _get_graph() -> GraphRetriever:
    global _graph
    if _graph is None:
        with _singleton_lock:
            if _graph is None:
                _graph = GraphRetriever()
    return _graph


def _get_hybrid() -> HybridRetriever:
    global _hybrid
    if _hybrid is None:
        with _singleton_lock:
            if _hybrid is None:
                _hybrid = HybridRetriever()
    return _hybrid


def _get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        with _singleton_lock:
            if _session_store is None:
                _session_store = SessionStore()
    return _session_store


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        with _singleton_lock:
            if _pipeline is None:
                from infragraph.ingestion.pipeline import IngestionPipeline
                _pipeline = IngestionPipeline()
                _pipeline.bootstrap()
    return _pipeline


def _text(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=message)]


def _json(payload: object) -> list[TextContent]:
    return _text(json.dumps(payload, indent=2))


def _decode_uploaded_file(filename: str, content_base64: str) -> tuple[str, bytes]:
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise ValueError("filename must not be empty.")
    try:
        return safe_name, base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("content_base64 must be valid base64.") from exc


def _build_source_name(filename: str, source: str | None) -> str:
    supplied = (source or "").strip()
    if not supplied:
        return Path(filename).name
    ext = Path(filename).suffix.lower()
    if ext and Path(supplied).suffix.lower() != ext:
        return f"{supplied}{ext}"
    return supplied


def _decode_uploaded_text_file(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in OCR_ALL_EXTS:
        raise ValueError(f"OCR-supported file type must use OCR upload tools: {filename}")
    sample = file_bytes[:4096]
    non_printable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    if non_printable / max(len(sample), 1) > 0.30:
        raise ValueError(f"Uploaded file appears to be binary, not text: {filename}")
    detected = chardet.detect(file_bytes)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return file_bytes.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return file_bytes.decode("utf-8", errors="replace")


# ── tool definitions ───────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ocr_extract_upload",
            description=(
                "Extract text from a single uploaded PDF or image using OCR. "
                "Upload the file bytes directly to the remote server as base64. "
                "Supports: .pdf, .png, .jpg, .jpeg, .tiff, .bmp"
            ),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Original file name, including extension."},
                    "content_base64": {"type": "string", "description": "Complete file contents encoded as base64."},
                },
                "required": ["filename", "content_base64"],
            },
        ),
        Tool(
            name="ocr_ingest_upload",
            description=(
                "OCR and ingest a single uploaded PDF or image into the knowledge graph. "
                "Use this when the file is uploaded directly to the remote server and you "
                "do not want path mounts or local filesystem references."
            ),
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Original file name, including extension.",
                    },
                    "content_base64": {
                        "type": "string",
                        "description": "Complete file contents encoded as base64.",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Optional stable source identifier. "
                            "If omitted, the filename is used."
                        ),
                    },
                },
                "required": ["filename", "content_base64"],
            },
        ),
        Tool(
            name="ingest_file_upload",
            description=(
                "Ingest a single uploaded text, code, or config file into the knowledge graph. "
                "Use this for .txt, .md, .py, .json, .yaml, and similar text-based files. "
                "Do not use this for PDFs or images."
            ),
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Original file name, including extension.",
                    },
                    "content_base64": {
                        "type": "string",
                        "description": "Complete file contents encoded as base64.",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Optional stable source identifier. "
                            "If omitted, the filename is used."
                        ),
                    },
                },
                "required": ["filename", "content_base64"],
            },
        ),
        Tool(
            name="search_knowledge",
            description=(
                "Search the knowledge base by semantic similarity. "
                "Returns ranked text chunks from runbooks, configs, and repos. "
                "Results are cached for 120s."
            ),
            annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "default": 20, "description": "Max results"},
                    "doc_type": {
                        "type": "string",
                        "description": "Filter by doc type: markdown, code, config, text",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="lookup_entity",
            description=(
                "Look up an infrastructure entity by name and return its "
                "properties and connected neighbours from the knowledge graph. "
                "Results are cached for 60s."
            ),
            annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name to look up"},
                    "depth": {
                        "type": "integer",
                        "default": 2,
                        "description": "Graph traversal depth (1-3)",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="deep_search",
            description=(
                "Answer a question using hybrid retrieval: "
                "semantic doc search + graph context, answered by deepseek. "
                "Results are cached for 120s."
            ),
            annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 20},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="knowledge_stats",
            description="Return entity and relation counts in the knowledge graph. Cached for 30s.",
            annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="remove_document",
            description=(
                "Delete all chunks and graph nodes for a given source identifier "
                "from the knowledge base (Qdrant + Neo4j). Use this to remove "
                "wrong or outdated information. Pass the exact source string used "
                "when ingesting, e.g. 'swarm-ui:overview'."
            ),
            annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Exact source identifier to delete, e.g. 'swarm-ui:overview'.",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="ingest_text",
            description=(
                "Ingest raw text content directly into the knowledge graph — "
                "no file needed. Use this when you have content in memory "
                "(clipboard, API response, generated docs, terminal output, etc.) "
                "that you want indexed without writing it to disk first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to ingest",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "A stable identifier for this content, e.g. "
                            "'k8s-runbook', 'nginx-config', 'incident-2024-03'. "
                            "Re-ingesting the same source name replaces the old chunks."
                        ),
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["markdown", "code", "config", "text"],
                        "description": (
                            "Chunking strategy: markdown=split on headers, "
                            "code=split on def/class, config=YAML/TOML stanzas, "
                            "text=fixed-size overlap. Defaults to 'text'."
                        ),
                    },
                },
                "required": ["content", "source"],
            },
        ),
        Tool(
            name="session_event_append",
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
            description=(
                "Append one exact ordered event to a session timeline. "
                "Use this for structured swarm/session state, not semantic ingestion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "body": {"type": ["object", "array", "string"]},
                    "job_id": {"type": "string"},
                    "worker_id": {"type": "string"},
                    "reply_to": {"type": "string"},
                    "checkpoint_ref": {"type": "string"},
                    "artifact_ref": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "event_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["session_id", "event_type", "summary"],
            },
        ),
        Tool(
            name="session_checkpoint_put",
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
            description=(
                "Store an exact durable checkpoint for a session. "
                "The latest checkpoint is used by session_resume_packet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "content": {"type": "string"},
                    "checkpoint_id": {"type": "string"},
                    "metadata": {"type": "object"},
                    "source_event_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["session_id", "content"],
            },
        ),
        Tool(
            name="session_checkpoint_get",
            annotations={"readOnlyHint": True, "idempotentHint": True},
            description="Load one exact checkpoint by ID, or the latest checkpoint when no ID is provided.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "checkpoint_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="session_artifact_put",
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
            description="Store a durable session artifact such as a summary, patch, or notes blob.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "artifact_id": {"type": "string"},
                    "content": {"type": "string"},
                    "kind": {"type": "string"},
                    "metadata": {"type": "object"},
                    "source_event_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["session_id", "artifact_id", "content"],
            },
        ),
        Tool(
            name="session_artifact_get",
            annotations={"readOnlyHint": True, "idempotentHint": True},
            description="Load one exact session artifact by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "artifact_id": {"type": "string"},
                },
                "required": ["session_id", "artifact_id"],
            },
        ),
        Tool(
            name="session_resume_packet",
            annotations={"readOnlyHint": True, "idempotentHint": True},
            description=(
                "Return a resume packet for a session: latest checkpoint, recent exact events, and artifact refs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "event_limit": {"type": "integer", "default": 50},
                },
                "required": ["session_id"],
            },
        ),
    ]


# ── tool handlers ──────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "session_event_append":
        session_id = (arguments.get("session_id") or "").strip()
        event_type = (arguments.get("event_type") or "").strip()
        summary = (arguments.get("summary") or "").strip()
        if not session_id:
            return _text("session_event_append: 'session_id' must not be empty.")
        if not event_type:
            return _text("session_event_append: 'event_type' must not be empty.")
        if not summary:
            return _text("session_event_append: 'summary' must not be empty.")
        payload = _get_session_store().append_event(
            session_id=session_id,
            event_type=event_type,
            summary=summary,
            body=arguments.get("body"),
            job_id=(arguments.get("job_id") or "").strip(),
            worker_id=(arguments.get("worker_id") or "").strip(),
            reply_to=(arguments.get("reply_to") or "").strip(),
            checkpoint_ref=(arguments.get("checkpoint_ref") or "").strip(),
            artifact_ref=(arguments.get("artifact_ref") or "").strip(),
            tags=arguments.get("tags") or [],
            metadata=arguments.get("metadata") or {},
            event_id=(arguments.get("event_id") or "").strip() or None,
            timestamp=(arguments.get("timestamp") or "").strip() or None,
        )
        return _json(payload)

    if name == "session_checkpoint_put":
        session_id = (arguments.get("session_id") or "").strip()
        content = (arguments.get("content") or "").strip()
        if not session_id:
            return _text("session_checkpoint_put: 'session_id' must not be empty.")
        if not content:
            return _text("session_checkpoint_put: 'content' must not be empty.")
        payload = _get_session_store().put_checkpoint(
            session_id=session_id,
            content=content,
            checkpoint_id=(arguments.get("checkpoint_id") or "").strip() or None,
            metadata=arguments.get("metadata") or {},
            source_event_id=(arguments.get("source_event_id") or "").strip(),
            timestamp=(arguments.get("timestamp") or "").strip() or None,
        )
        return _json(payload)

    if name == "session_checkpoint_get":
        session_id = (arguments.get("session_id") or "").strip()
        if not session_id:
            return _text("session_checkpoint_get: 'session_id' must not be empty.")
        payload = _get_session_store().get_checkpoint(
            session_id=session_id,
            checkpoint_id=(arguments.get("checkpoint_id") or "").strip() or None,
        )
        if payload is None:
            return _text("null")
        return _json(payload)

    if name == "session_artifact_put":
        session_id = (arguments.get("session_id") or "").strip()
        artifact_id = (arguments.get("artifact_id") or "").strip()
        content = (arguments.get("content") or "").strip()
        if not session_id:
            return _text("session_artifact_put: 'session_id' must not be empty.")
        if not artifact_id:
            return _text("session_artifact_put: 'artifact_id' must not be empty.")
        if not content:
            return _text("session_artifact_put: 'content' must not be empty.")
        payload = _get_session_store().put_artifact(
            session_id=session_id,
            artifact_id=artifact_id,
            content=content,
            kind=(arguments.get("kind") or "text").strip() or "text",
            metadata=arguments.get("metadata") or {},
            source_event_id=(arguments.get("source_event_id") or "").strip(),
            timestamp=(arguments.get("timestamp") or "").strip() or None,
        )
        return _json(payload)

    if name == "session_artifact_get":
        session_id = (arguments.get("session_id") or "").strip()
        artifact_id = (arguments.get("artifact_id") or "").strip()
        if not session_id:
            return _text("session_artifact_get: 'session_id' must not be empty.")
        if not artifact_id:
            return _text("session_artifact_get: 'artifact_id' must not be empty.")
        payload = _get_session_store().get_artifact(session_id=session_id, artifact_id=artifact_id)
        if payload is None:
            return _text("null")
        return _json(payload)

    if name == "session_resume_packet":
        session_id = (arguments.get("session_id") or "").strip()
        if not session_id:
            return _text("session_resume_packet: 'session_id' must not be empty.")
        event_limit = max(1, min(int(arguments.get("event_limit", 50)), 500))
        payload = _get_session_store().get_resume_packet(
            session_id=session_id,
            event_limit=event_limit,
        )
        return _json(payload)

    if name == "ocr_extract_upload":
        from infragraph.ingestion.ocr import extract_text_bytes
        try:
            filename, file_bytes = _decode_uploaded_file(
                arguments["filename"],
                arguments["content_base64"],
            )
        except ValueError as exc:
            return _text(f"OCR failed: {exc}")

        if not is_ocr_candidate(filename):
            return _text(f"Not an OCR-supported file type: {filename}")
        try:
            result = extract_text_bytes(filename, file_bytes)
            summary = (
                f"Extracted {len(result.text)} chars "
                f"via {result.method} "
                f"({result.page_count} page(s))"
            )
            if result.warnings:
                summary += "\nWarnings: " + "; ".join(result.warnings)
            return _text(f"{summary}\n\n---\n{result.text[:2000]}")
        except Exception as exc:
            return _text(f"OCR failed: {exc}")

    if name == "ocr_ingest_upload":
        try:
            filename, file_bytes = _decode_uploaded_file(
                arguments["filename"],
                arguments["content_base64"],
            )
        except ValueError as exc:
            return _text(f"Ingestion failed: {exc}")

        def _run_ocr_ingest() -> str:
            from infragraph.embedding import embed_texts
            from infragraph.ingestion.chunker import chunk_document
            from infragraph.ingestion.ocr import extract_text_bytes
            from infragraph.normalization.writer import GraphWriter

            pipeline = _get_pipeline()
            result = extract_text_bytes(filename, file_bytes)
            source_name = _build_source_name(filename, arguments.get("source"))
            chunks = chunk_document(result.text, source_name)

            pipeline._qdrant.delete_by_source(source_name)
            entities_written = relations_written = 0
            if chunks:
                vectors = embed_texts([c.text for c in chunks])
                payloads = [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_path": c.source_path,
                        "doc_type": c.doc_type,
                        "index": c.index,
                    }
                    for c in chunks
                ]
                pipeline._qdrant.upsert_chunks(payloads, vectors)

            writer = GraphWriter(pipeline._neo4j)
            for chunk in chunks:
                try:
                    extracted = pipeline._extract.extract(
                        chunk.text,
                        source_doc=chunk.source_path,
                        chunk_id=chunk.chunk_id,
                    )
                    written = writer.write(extracted)
                    entities_written += written["entities"]
                    relations_written += written["relations"]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "graph extraction skipped for %s: %s", chunk.chunk_id, exc)

            return (
                f"Uploaded file ingested: {filename}\n"
                f"  Source           : {source_name}\n"
                f"  OCR method       : {result.method}\n"
                f"  Chunks produced  : {len(chunks)}\n"
                f"  Entities → Neo4j : {entities_written}\n"
                f"  Relations → Neo4j: {relations_written}"
            )

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, _run_ocr_ingest)
        return _text(msg)

    if name == "ingest_file_upload":
        try:
            filename, file_bytes = _decode_uploaded_file(
                arguments["filename"],
                arguments["content_base64"],
            )
            text = _decode_uploaded_text_file(filename, file_bytes)
        except ValueError as exc:
            return _text(f"Ingestion failed: {exc}")

        def _run_file_ingest() -> str:
            from infragraph.embedding import embed_texts
            from infragraph.ingestion.chunker import chunk_document
            from infragraph.normalization.writer import GraphWriter

            pipeline = _get_pipeline()
            source_name = _build_source_name(filename, arguments.get("source"))
            chunks = chunk_document(text, source_name)

            pipeline._qdrant.delete_by_source(source_name)
            entities_written = relations_written = 0
            if chunks:
                vectors = embed_texts([c.text for c in chunks])
                payloads = [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_path": c.source_path,
                        "doc_type": c.doc_type,
                        "index": c.index,
                    }
                    for c in chunks
                ]
                pipeline._qdrant.upsert_chunks(payloads, vectors)

            writer = GraphWriter(pipeline._neo4j)
            for chunk in chunks:
                try:
                    extracted = pipeline._extract.extract(
                        chunk.text,
                        source_doc=chunk.source_path,
                        chunk_id=chunk.chunk_id,
                    )
                    written = writer.write(extracted)
                    entities_written += written["entities"]
                    relations_written += written["relations"]
                except Exception as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "graph extraction skipped for %s: %s", chunk.chunk_id, exc
                    )

            return (
                f"Uploaded file ingested: {filename}\n"
                f"  Source           : {source_name}\n"
                f"  Chunks produced  : {len(chunks)}\n"
                f"  Entities → Neo4j : {entities_written}\n"
                f"  Relations → Neo4j: {relations_written}"
            )

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, _run_file_ingest)
        return _text(msg)

    if name == "search_knowledge":
        cache_args = {
            "query": arguments["query"],
            "top_k": arguments.get("top_k", 20),
            "doc_type": arguments.get("doc_type"),
        }
        cached = _rcache_get("search_knowledge", cache_args)
        if cached:
            return _text(cached)
        results = _get_semantic().search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 20),
            doc_type=arguments.get("doc_type"),
        )
        out = json.dumps(results, indent=2)
        _rcache_set("search_knowledge", cache_args, out)
        return _text(out)

    if name == "lookup_entity":
        entity_name = arguments["name"]
        depth = max(1, min(int(arguments.get("depth", 2)), 3))
        cache_args = {"name": entity_name, "depth": depth}
        cached = _rcache_get("lookup_entity", cache_args)
        if cached:
            return _text(cached)
        matches = _get_graph().find(entity_name, limit=3)
        results: list[dict] = []
        for m in matches:
            node = m.get("node", {})
            eid = node.get("id")
            if eid:
                nbrs = _get_graph().neighbours(eid, depth=depth)
                results.append({"entity": node, "neighbours": nbrs[:10]})
        out = json.dumps(results, indent=2)
        _rcache_set("lookup_entity", cache_args, out)
        return _text(out)

    if name == "deep_search":
        cache_args = {"question": arguments["question"], "top_k": arguments.get("top_k", 20)}
        cached = _rcache_get("deep_search", cache_args)
        if cached:
            return _text(cached)
        result = _get_hybrid().answer(
            query=arguments["question"],
            top_k=arguments.get("top_k", 20),
        )
        answer = result["answer"]
        _rcache_set("deep_search", cache_args, answer)
        return _text(answer)

    if name == "knowledge_stats":
        cached = _rcache_get("knowledge_stats", {})
        if cached:
            return _text(cached)
        stats = _get_graph().summary_stats()
        out = json.dumps(stats, indent=2)
        _rcache_set("knowledge_stats", {}, out)
        return _text(out)

    if name == "remove_document":
        source = arguments.get("source", "").strip()
        if not source:
            return _text("remove_document: 'source' identifier must not be empty.")

        def _run_delete() -> str:
            pipeline = _get_pipeline()
            pipeline._qdrant.delete_by_source(source)
            try:
                pipeline._neo4j.execute_query(
                    "MATCH (n) WHERE n.source_doc = $src DETACH DELETE n",
                    parameters_={"src": source},
                )
            except Exception as exc:
                return f"Deleted '{source}' from Qdrant. Neo4j cleanup skipped: {exc}"
            return f"Deleted source '{source}' from Qdrant and Neo4j."

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, _run_delete)
        return _text(msg)

    if name == "ingest_text":
        content  = arguments.get("content", "").strip()
        source   = arguments.get("source", "").strip()
        doc_type = arguments.get("doc_type", "text")

        if not content:
            return _text("ingest_text: 'content' must not be empty.")
        if not source:
            return _text("ingest_text: 'source' identifier must not be empty.")

        def _run_text_ingest() -> str:
            from infragraph.embedding import embed_texts
            from infragraph.ingestion.chunker import chunk_document
            from infragraph.normalization.writer import GraphWriter

            pipeline = _get_pipeline()

            # Chunk using source as the path identifier (carries doc_type hint via ext)
            # chunk_document auto-detects doc_type from extension; for raw text we
            # append a fake extension so the right splitter is used.
            _ext_map = {"markdown": ".md", "code": ".py", "config": ".yaml", "text": ".txt"}
            fake_source = f"{source}{_ext_map.get(doc_type, '.txt')}"
            chunks = chunk_document(content, fake_source)

            # Replace any previous vectors for this source
            pipeline._qdrant.delete_by_source(source)
            entities_written = relations_written = 0
            if chunks:
                vectors = embed_texts([c.text for c in chunks])
                payloads = [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_path": source,
                        "doc_type": c.doc_type,
                        "index": c.index,
                    }
                    for c in chunks
                ]
                pipeline._qdrant.upsert_chunks(payloads, vectors)

            # Extract entities/relations per chunk and write to Neo4j
            writer = GraphWriter(pipeline._neo4j)
            for chunk in chunks:
                try:
                    result = pipeline._extract.extract(
                        chunk.text, source_doc=source, chunk_id=chunk.chunk_id
                    )
                    written = writer.write(result)
                    entities_written  += written["entities"]
                    relations_written += written["relations"]
                except Exception as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "graph extraction skipped for %s: %s", chunk.chunk_id, exc
                    )

            return (
                f"Ingested text source '{source}' ({doc_type})\n"
                f"  Chunks → Qdrant : {len(chunks)}\n"
                f"  Entities → Neo4j: {entities_written}\n"
                f"  Relations → Neo4j: {relations_written}"
            )

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, _run_text_ingest)
        return _text(msg)

    return _text(f"Unknown tool: {name}")


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Transport is selected by the MODE env var:
      MODE=stdio  (default) — for local Claude Code integration
      MODE=sse              — for remote HTTP deployment (docker / devops-server)
    """
    import os
    logging.basicConfig(level=logging.INFO)
    mode = os.environ.get("MODE", "stdio").lower()
    if mode == "sse" and not get_settings().mcp_api_key:
        raise RuntimeError("MCP_API_KEY must be set when MODE=sse")

    if mode == "sse":
        _run_sse()
    else:
        asyncio.run(_run_stdio())


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def _run_sse() -> None:
    """HTTP/SSE server — used when deployed as a container."""
    import os

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route

    cfg = get_settings()
    port = int(os.environ.get("PORT", 8080))
    sse_transport = SseServerTransport("/messages/")

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            if request.url.path == "/health":
                return await call_next(request)

            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {cfg.mcp_api_key}":
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    async def handle_sse(request: Request) -> Response:
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read, write):
            await app.run(read, write, app.create_initialization_options())
        return Response()

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": "infragraph"})

    starlette_app = Starlette(
        routes=[
            Route("/health", health),
            Route("/sse", handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

    log.info("infragraph MCP server starting on http://0.0.0.0:%d/sse", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
