"""
Ingest self-describing infragraph documentation directly into the pipeline.
Run inside the infragraph container: python3 /tmp/infragraph_ingest_docs.py
"""
import logging

from infragraph.embedding import embed_texts
from infragraph.ingestion.chunker import chunk_document
from infragraph.ingestion.pipeline import IngestionPipeline
from infragraph.normalization.writer import GraphWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ingest_docs")

pipeline = IngestionPipeline()
pipeline.bootstrap()

DOCS = {}

DOCS["infragraph/architecture-overview"] = ("markdown", """\
# infragraph Architecture Overview

infragraph is a self-hosted hybrid RAG knowledge base system for infrastructure documentation.
It combines vector search, lexical search, and graph-based entity relationships to answer
questions about a homelab/homeserver environment.

Deployed as Docker container infragraph-mcp on devops-server at 192.168.0.50, port 8181.
Exposed as MCP (Model Context Protocol) server over SSE/HTTP.

## Storage Layer

Qdrant is the vector database running on devops-server. It stores chunked text as 1536-dim
dense embeddings produced by text-embedding-3-small via OpenAI. Enables semantic similarity
search. Each point stores: chunk_id, text, source_path, doc_type, index, and metadata.

Neo4j is the graph database running on devops-server. Stores entities (servers, services,
IPs, ports, configs) and typed relationships. Entity types: Server, Service, IP, Port, Config,
Database, Network, User, Script, Package, Unknown. Relation types: RUNS_ON, CONNECTS_TO,
DEPENDS_ON, CONFIGURED_BY, EXPOSES, HOSTS, PART_OF, USES, MANAGES, RELATED_TO.

Redis runs on devops-server at 192.168.0.50:6379, database 1. It caches context headers
keyed by SHA256(chunk_text). Permanent TTL eliminates redundant LLM calls on re-ingestion.

## Deployment

Host: devops-server (192.168.0.50). External port 8181 maps to internal port 8080.
Docker image: docker-local.homeserverlocal.com/infragraph/infragraph:latest.
Compose file: /opt/infragraph/docker-compose.yml. Python 3.11, hatchling package.
Runtime: SSE mode via python -m infragraph.mcp.server.
""")

DOCS["infragraph/ingestion-pipeline"] = ("markdown", """\
# infragraph Ingestion Pipeline

The ingestion pipeline orchestrates: scan, chunk, embed, upsert Qdrant, extract, normalize,
upsert Neo4j. Entry point: IngestionPipeline.run(paths, include_ocr=True).

## Pipeline Stages

Stage 1 Scan: reads files from disk paths using scan_paths(). Supports text files, config
files, and PDFs/images via OCR (Tesseract). Returns list of (path, text) tuples.

Stage 2 Chunk: splits each document into overlapping text segments using chunk_document().
Chunking strategy is inferred from file extension: .md/.rst uses markdown header splitting,
.py/.js/.go/.ts uses code definition splitting, .yaml/.toml/.ini uses config stanza splitting,
all others use fixed-size sliding window with overlap.

Stage 3 Contextual Headers: for each chunk, calls ContextHeaderGenerator.get_header().
Generates a 1-2 sentence summary using Mistral Large 2512 via OpenRouter. The header
is prepended to the chunk text before embedding. Redis cache keyed by SHA256(chunk_text)
prevents repeated LLM calls for unchanged content across re-ingestion runs.

Stage 4 Embed: calls embed_texts() in batches of 64. Uses text-embedding-3-small via
OpenAI API to produce 1536-dimensional dense vectors.

Stage 5 Upsert Qdrant: writes vectors and payloads to Qdrant collection. Payload fields:
chunk_id, text, source_path, doc_type, index, plus any metadata from the scan stage.

Stage 6 Extract: for each chunk, calls Extractor.extract() using Mistral Large 2512.
Returns ExtractionResult with lists of Entity and Relation objects validated by Pydantic.
If extraction fails for a chunk (e.g., validation error on schema-describing content),
the error is logged as a warning and that chunk is skipped. Qdrant vectors are preserved.

Stage 7 Normalize and Write Neo4j: GraphWriter.write() upserts entities with MERGE and
creates relationships. Returns counts of entities and relations written.

## Re-ingestion Behavior

Re-ingesting the same source_path replaces all prior Qdrant vectors for that path
(delete_by_source before upsert). Neo4j uses MERGE so re-ingestion is idempotent for
entities but may add duplicate relations if not cleaned first.

## Context Header Caching

Redis key format: ctx:<sha256hex>. Value: UTF-8 encoded header string. No TTL set.
Provider-side caching also benefits from sending the same document prefix repeatedly
(Mistral caches prompt prefixes). This makes re-ingesting a modified document cheap:
only changed chunks need new header generation.
""")

DOCS["infragraph/retrieval-system"] = ("markdown", """\
# infragraph Retrieval System

Hybrid retrieval fuses semantic vector search, BM25 lexical search, and Neo4j graph context.

## Retrieval Flow

Step 1 Semantic Search: the query is embedded with text-embedding-3-small. Qdrant returns
the top-k most similar chunks by cosine similarity. Default top_k is 20.

Step 2 BM25 Lexical Search: an in-memory BM25Okapi index (rank_bm25 library) is built
at startup by scrolling all points from Qdrant. Returns top-k chunks by BM25 score.
Excels at exact token matches: IP addresses like 192.168.0.50, port numbers like 5020,
service names like nginx or folo, error codes. Semantic embeddings are weak on these tokens.

Step 3 RRF Fusion: Reciprocal Rank Fusion merges the two ranked lists. Score for each
chunk: sum of 1/(k + rank) for each list it appears in, with k=60 (standard constant from
the original RRF paper). Chunks appearing in both lists get boosted. The fused list replaces
the raw semantic hits for downstream use.

Step 4 Graph Lookup: capitalised word candidates are extracted from the top 3 chunks using
a regex heuristic. Each candidate is looked up in Neo4j. For each matching entity, neighbours
are fetched up to depth 2. De-duplication prevents redundant entity fetches.

Step 5 Answer Synthesis: semantic chunks (formatted with score and source path) and graph
context (entity names, types, neighbour names) are assembled into a context string.
A system prompt instructs the LLM to answer using only the provided context and cite sources.
DeepSeek-V3-0324 via OpenRouter generates the final answer. Temperature is 0.2.
HTTP retries use exponential backoff (2-20s) via tenacity, up to 3 attempts.

## BM25 Index Lifecycle

The BM25 index is built once at first query (lazy initialization via _get_bm25()).
It lives in memory for the container lifetime. On container restart the index is rebuilt
from Qdrant scroll (a few seconds for homelab scale). No persistence is needed.

## MCP Tools for Retrieval

semantic_search: returns ranked chunk list with scores, source paths, and text.
hybrid_query: runs the full hybrid pipeline and returns a synthesized LLM answer.
graph_lookup: looks up a named entity and returns its Neo4j neighbours to depth 1-3.
graph_stats: returns total entity and relation counts from Neo4j.
""")

DOCS["infragraph/model-configuration"] = ("markdown", """\
# infragraph Model Configuration

infragraph uses four distinct models via two providers, each optimised for a specific role.

## Models

text-embedding-3-small via OpenAI API. Used for embedding chunks and queries into 1536-dim
vectors. Chosen for: cost efficiency, high quality for RAG retrieval, native OpenAI support.
Embedding dimension is 1536 (embed_dim config). Used in every ingest and every query.

mistralai/mistral-large-2512 via OpenRouter. Used for two distinct tasks:
1. Context header generation (context_model config): generates 1-2 sentence summaries
   prepended to chunks before embedding. Max 120 tokens output, temperature 0.1.
   Provider-side caching on OpenRouter reduces cost when the same document prefix repeats.
2. Entity and relation extraction (extract_model in Extractor): extracts structured
   ExtractionResult from chunk text. Returns JSON with entities and relations arrays.

deepseek/deepseek-v3-0324 via OpenRouter. Used for answer synthesis (chat_model config).
Given fused retrieval context, generates the final user-facing answer. Temperature 0.2.
Strong reasoning, very cost-effective for long-context synthesis. Max 90s HTTP timeout.

## Config Fields (infragraph/config.py via pydantic-settings)

embed_dim: 1536 (matches text-embedding-3-small output dimension)
context_model: mistralai/mistral-large-2512
redis_url: redis://192.168.0.50:6379
context_cache_db: 1
chat_model: deepseek/deepseek-v3-0324
openrouter_base_url: https://openrouter.ai/api/v1
openrouter_api_key: set via OPENROUTER_API_KEY env var
embed_api_key: set via EMBED_API_KEY env var (OpenAI key)

## Docker Environment Variables

Set in /opt/infragraph/docker-compose.yml:
OPENROUTER_API_KEY, EMBED_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
QDRANT_URL, QDRANT_COLLECTION, REDIS_URL, CONTEXT_CACHE_DB, CONTEXT_MODEL.

## Cost Profile

Most expensive operation: context header generation (Mistral Large per chunk on first
ingest). Redis cache makes re-ingestion of unchanged content effectively free.
Query cost is minimal: one embedding call (cheap) plus one DeepSeek call (cheap).
Graph extraction (Mistral Large) is the second most expensive per-ingest cost.
""")

DOCS["infragraph/system-assessment"] = ("markdown", """\
# infragraph System Assessment

Assessment of the infragraph knowledge base system: strengths, known issues, and recommendations.

## Strengths

Hybrid retrieval is the core architectural strength. BM25 covers exact token queries
(IP addresses, port numbers, service names) where semantic embeddings fail. RRF fusion
ensures chunks matching both methods rank highest, reducing false negatives.

Contextual retrieval (Anthropic technique) prepends LLM-generated headers to chunks
before embedding. This gives each chunk document-level context that would otherwise
be lost. The Redis cache makes re-ingestion cheap after the first run.

Graph augmentation surfaces relationship chains that pure vector search misses. For
example: asking about a service returns its host, exposed ports, and dependent services
via Neo4j traversal, even without those details appearing in the matched chunk text.

Graceful degradation: if graph extraction fails for a chunk (Pydantic validation error,
LLM timeout, etc.), the chunk is still embedded and searchable via Qdrant. The system
degrades to semantic-only for that chunk rather than aborting the entire ingestion.

Self-contained deployment: single Docker container, all config via environment variables,
no external dependencies beyond Neo4j, Qdrant, Redis (all self-hosted on devops-server).

## Known Issues

Context header generation adds significant latency to initial ingestion. A 10-chunk
document can take 60-90 seconds on first ingest due to sequential Mistral Large calls.
Re-ingestion is fast (all cached), but first-run latency may cause MCP client timeouts.

MCP tool timeout: the ingest_text and ingest_file_upload tools can timeout for large
documents because header generation runs synchronously in an executor inside the async
MCP handler. Workaround: ingest directly via SSH exec into the container for large docs.

Schema-describing content causes graph extraction failures. When a chunk contains text
that lists infragraph's own entity type names (like EntityType.Server), Mistral may
return those as entity type values, failing Pydantic validation. The fix (try/except
per chunk) is in place and the chunk is skipped for graph extraction.

BM25 index is in-memory only. On container restart, it must be rebuilt by scrolling
all Qdrant points. At homelab scale this is fast (<5 seconds), but grows linearly with
corpus size.

Parallel ingestion causes timeouts. Calling ingest_text for multiple documents
simultaneously overloads the Mistral Large extraction step. Always ingest sequentially.

## Architecture Recommendations

For production use: move context header generation to an async queue (Celery or ARQ)
so ingest returns immediately and headers are filled in asynchronously.

Consider adding a re-ranking step (cross-encoder) after RRF fusion to improve precision
on the final top-10 chunks sent to the LLM.

The Neo4j MERGE-based upsertion is idempotent for entities but can accumulate duplicate
relationships across re-ingestion runs. Adding a relationship deduplication check would
keep the graph clean.

Increase top_k beyond 20 for complex multi-hop questions. The current default of 20
may miss relevant chunks for questions spanning multiple services or time periods.

## Summary

infragraph is well-suited for homelab infrastructure Q&A. The hybrid retrieval pipeline
(semantic + BM25 + graph) covers the main failure modes of pure vector RAG. The main
operational constraint is ingestion latency on large document sets due to synchronous
LLM calls for context headers. For a homelab with a few hundred documents, this is
acceptable.
""")


def ingest_doc(source, doc_type, content):
    log.info("=== Ingesting: %s ===", source)
    pipeline._qdrant.delete_by_source(source)

    fake_ext = {
        "markdown": ".md",
        "code": ".py",
        "config": ".yaml",
        "text": ".txt",
    }.get(doc_type, ".txt")
    fake_source = f"{source}{fake_ext}"

    chunks = chunk_document(content, fake_source)
    log.info("  produced %d chunks", len(chunks))

    # Context headers + embed
    by_source = {}
    by_source[fake_source] = chunks
    contextualised = []
    for src, cks in by_source.items():
        for chunk in cks:
            header = pipeline._ctx_headers.get_header(chunk.text, content)
            if header:
                chunk.text = f"{header}\n\n{chunk.text}"
            contextualised.append(chunk)

    chunks = contextualised

    # Embed + upsert Qdrant
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        vectors = embed_texts(texts)
        payloads = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_path": source,
                "doc_type": c.doc_type,
                "index": c.index,
            }
            for c in batch
        ]
        pipeline._qdrant.upsert_chunks(payloads, vectors)
    log.info("  wrote %d vectors to Qdrant", len(chunks))

    # Graph extraction
    writer = GraphWriter(pipeline._neo4j)
    entities = relations = 0
    for chunk in chunks:
        try:
            result = pipeline._extract.extract(
                chunk.text,
                source_doc=source,
                chunk_id=chunk.chunk_id,
            )
            written = writer.write(result)
            entities += written["entities"]
            relations += written["relations"]
        except Exception as exc:
            log.warning("  graph extraction skipped for %s: %s", chunk.chunk_id, exc)

    log.info("  graph: %d entities, %d relations", entities, relations)
    log.info("  DONE: %s", source)


for source, (doc_type, content) in DOCS.items():
    try:
        ingest_doc(source, doc_type, content)
    except Exception as exc:
        log.error("FAILED %s: %s", source, exc)

log.info("All documents ingested.")
