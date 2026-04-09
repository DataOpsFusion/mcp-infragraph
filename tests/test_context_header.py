"""Tests for ContextHeaderGenerator — uses fakes for Redis and HTTP."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

SAMPLE_DOC = "# nginx Service\nNginx runs on Docker-Server at 192.168.0.50 port 80."
SAMPLE_CHUNK = "Nginx runs on Docker-Server at 192.168.0.50 port 80."
EXPECTED_HEADER = "This chunk is from the nginx service doc for Docker-Server (192.168.0.50)."


def _cache_key(chunk_text: str) -> str:
    return "ctx:" + hashlib.sha256(chunk_text.encode()).hexdigest()


def test_cache_hit_skips_llm():
    """If Redis has the key, the LLM should never be called."""
    from infragraph.ingestion.context_header import ContextHeaderGenerator

    fake_redis = MagicMock()
    fake_redis.get.return_value = EXPECTED_HEADER.encode()

    gen = ContextHeaderGenerator.__new__(ContextHeaderGenerator)
    gen._redis = fake_redis

    with patch.object(gen, "_call_llm") as mock_llm:
        result = gen.get_header(SAMPLE_CHUNK, SAMPLE_DOC)

    mock_llm.assert_not_called()
    assert result == EXPECTED_HEADER


def test_cache_miss_calls_llm_and_stores():
    """On a cache miss, LLM is called and result is stored in Redis."""
    from infragraph.ingestion.context_header import ContextHeaderGenerator

    fake_redis = MagicMock()
    fake_redis.get.return_value = None  # cache miss

    gen = ContextHeaderGenerator.__new__(ContextHeaderGenerator)
    gen._redis = fake_redis

    with patch.object(gen, "_call_llm", return_value=EXPECTED_HEADER):
        result = gen.get_header(SAMPLE_CHUNK, SAMPLE_DOC)

    key = _cache_key(SAMPLE_CHUNK)
    fake_redis.set.assert_called_once_with(key, EXPECTED_HEADER)
    assert result == EXPECTED_HEADER


def test_llm_failure_returns_empty_string():
    """If the LLM call fails, return empty string — don't crash ingestion."""
    from infragraph.ingestion.context_header import ContextHeaderGenerator

    fake_redis = MagicMock()
    fake_redis.get.return_value = None

    gen = ContextHeaderGenerator.__new__(ContextHeaderGenerator)
    gen._redis = fake_redis

    with patch.object(gen, "_call_llm", side_effect=Exception("API error")):
        result = gen.get_header(SAMPLE_CHUNK, SAMPLE_DOC)

    assert result == ""
    fake_redis.set.assert_not_called()


def test_empty_chunk_returns_empty_string():
    """Blank chunks should be skipped — no LLM call, no cache write."""
    from infragraph.ingestion.context_header import ContextHeaderGenerator

    gen = ContextHeaderGenerator.__new__(ContextHeaderGenerator)
    gen._redis = MagicMock()

    result = gen.get_header("   ", SAMPLE_DOC)
    assert result == ""
