"""Tests for BM25Index and RRF fusion."""
from __future__ import annotations
import pytest


FAKE_CHUNKS = [
    {"chunk_id": "nginx.md::0", "text": "nginx reverse proxy port 80 docker-server", "source_path": "nginx.md", "doc_type": "markdown"},
    {"chunk_id": "folo.md::0", "text": "folo rss reader port 5020 docker-server", "source_path": "folo.md", "doc_type": "markdown"},
    {"chunk_id": "vpn.md::0",  "text": "openvpn tls-crypt server port 1194 udp",   "source_path": "vpn.md",  "doc_type": "markdown"},
]


def test_bm25_exact_match_ranks_first():
    """An exact keyword match should be ranked first."""
    from infragraph.retrieval.bm25 import BM25Index
    idx = BM25Index()
    idx.build(FAKE_CHUNKS)
    results = idx.search("folo rss", top_k=3)
    assert results[0]["payload"]["chunk_id"] == "folo.md::0"


def test_bm25_ip_match():
    """Searching for a service name should surface the right chunk."""
    from infragraph.retrieval.bm25 import BM25Index
    idx = BM25Index()
    idx.build(FAKE_CHUNKS)
    results = idx.search("openvpn 1194", top_k=3)
    assert results[0]["payload"]["chunk_id"] == "vpn.md::0"


def test_bm25_returns_top_k():
    from infragraph.retrieval.bm25 import BM25Index
    idx = BM25Index()
    idx.build(FAKE_CHUNKS)
    results = idx.search("port docker-server", top_k=2)
    assert len(results) <= 2


def test_rrf_deduplicates_and_merges():
    """RRF should merge two hit lists, dedup by chunk_id, rank by fused score."""
    from infragraph.retrieval.bm25 import rrf_merge

    semantic = [
        {"payload": {"chunk_id": "a::0"}, "score": 0.9},
        {"payload": {"chunk_id": "b::0"}, "score": 0.7},
    ]
    bm25 = [
        {"payload": {"chunk_id": "b::0"}, "score": 0.8},
        {"payload": {"chunk_id": "c::0"}, "score": 0.6},
    ]
    merged = rrf_merge(semantic, bm25)
    ids = [h["payload"]["chunk_id"] for h in merged]
    # b appears in both lists — should rank highest
    assert ids[0] == "b::0"
    assert len(ids) == 3  # a, b, c — no duplicates


def test_rrf_empty_bm25():
    """RRF should handle an empty BM25 result list gracefully."""
    from infragraph.retrieval.bm25 import rrf_merge
    semantic = [{"payload": {"chunk_id": "a::0"}, "score": 0.9}]
    merged = rrf_merge(semantic, [])
    assert len(merged) == 1
