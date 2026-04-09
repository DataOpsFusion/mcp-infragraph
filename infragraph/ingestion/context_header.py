"""
Context header generator for contextual retrieval.

For each chunk, generates a 1-2 sentence header describing where the chunk
comes from, prepended before embedding. Uses:
  - Redis cache keyed by SHA256(chunk_text) — eliminates cost on re-ingestion
  - Mistral Large 2512 via OpenRouter — provider-side caching reduces cost
    within a single ingestion run (same doc prefix sent repeatedly)
"""
from __future__ import annotations

import hashlib
import logging

import httpx
import redis as redis_lib

from infragraph.config import get_settings

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a documentation indexing assistant. "
    "Your output will be prepended to document chunks to improve search retrieval. "
    "Be concise — one or two sentences maximum."
)

_USER_TMPL = (
    "<document>\n{doc_text}\n</document>\n\n"
    "Write a single sentence (max 80 tokens) describing what this chunk is about "
    "and which document/service it belongs to, so it can be found by search:\n\n"
    "<chunk>\n{chunk_text}\n</chunk>"
)


class ContextHeaderGenerator:
    def __init__(self) -> None:
        cfg = get_settings()
        self._model = cfg.context_model
        self._base_url = cfg.openrouter_base_url.rstrip("/")
        self._http_headers = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DataOpsFusion/infragraph",
            "X-Title": "infragraph",
        }
        # Reuse a single client for connection pooling across all chunks in a run
        self._client = httpx.Client(timeout=30.0)
        self._redis = redis_lib.from_url(
            cfg.redis_url,
            db=cfg.context_cache_db,
            decode_responses=False,
        )

    def get_header(self, chunk_text: str, doc_text: str) -> str:
        """
        Return a context header for chunk_text.
        Checks Redis first; calls the LLM on a miss and stores the result.
        Returns empty string if chunk is blank or LLM fails.
        """
        if not chunk_text.strip():
            return ""

        key = "ctx:" + hashlib.sha256(chunk_text.encode()).hexdigest()

        cached = self._redis.get(key)
        if cached is not None:
            try:
                return cached.decode("utf-8")
            except UnicodeDecodeError as exc:
                log.error("Redis cache corruption for key %s: %s", key, exc)
                # Fall through to regenerate

        try:
            header = self._call_llm(chunk_text, doc_text)
        except httpx.HTTPStatusError as exc:
            log.warning("LLM API HTTP error %s: %s", exc.response.status_code, exc)
            return ""
        except httpx.HTTPError as exc:
            log.warning("LLM API network error: %s", exc)
            return ""
        except (KeyError, IndexError, TypeError) as exc:
            log.error("Unexpected LLM response structure: %s", exc)
            return ""
        except Exception as exc:
            log.error("Unexpected error in context header generation: %s", exc, exc_info=True)
            return ""

        self._redis.set(key, header)
        return header

    def _call_llm(self, chunk_text: str, doc_text: str) -> str:
        doc_snippet = doc_text[:6000]
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TMPL.format(
                    doc_text=doc_snippet,
                    chunk_text=chunk_text,
                ),
            },
        ]
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._http_headers,
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 120,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected API response structure: {exc}") from exc

    def close(self) -> None:
        """Release HTTP and Redis connections."""
        self._client.close()
        self._redis.close()
