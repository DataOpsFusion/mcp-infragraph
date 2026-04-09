from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # ── Neo4j (db-docker) ─────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://192.168.0.101:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ── Qdrant (db-docker) ────────────────────────────────────────────────────
    qdrant_host: str = "192.168.0.101"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "infragraph"

    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── MCP server ────────────────────────────────────────────────────────────
    mcp_api_key: str = ""

    # ── Model tiers ───────────────────────────────────────────────────────────
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    context_model: str = "mistralai/mistral-large-2512"
    redis_url: str = "redis://192.168.0.50:6379"
    context_cache_db: int = 1
    extract_model: str = "mistralai/mistral-large-2512"
    chat_model: str = "deepseek/deepseek-v3-0324"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Normalization ─────────────────────────────────────────────────────────
    dedup_threshold: float = 0.92


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
