"""OpenRouter embedding client — replaces local sentence-transformers."""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from infragraph.config import get_settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the OpenRouter embeddings endpoint."""
    cfg = get_settings()
    resp = httpx.post(
        f"{cfg.openrouter_base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={"model": cfg.embed_model, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # OpenAI-compatible: list of {"index": i, "embedding": [...]}
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
