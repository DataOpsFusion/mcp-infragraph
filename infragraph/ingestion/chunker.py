"""
Per-doctype chunking strategies.

Different document types need different splitting logic:
- Markdown   → split on headers, preserve hierarchy
- Code       → split on top-level functions/classes
- Config     → keep stanzas / YAML top-level keys together
- Plain text → fixed-size with overlap
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from infragraph.config import get_settings

_cfg = get_settings()


@dataclass
class Chunk:
    chunk_id: str          # "{source_path}::{index}"
    text: str
    source_path: str
    doc_type: str
    index: int
    metadata: dict = field(default_factory=dict)


# ── helpers ────────────────────────────────────────────────────────────────────

def _fixed_split(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of approximately `size` characters."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return [c for c in chunks if c.strip()]


def _make_chunks(
    segments: list[str],
    source_path: str,
    doc_type: str,
    max_chars: int = 2000,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cfg = get_settings()
    for seg in segments:
        if not seg.strip():
            continue
        # If segment is still too long, split further
        if len(seg) > max_chars:
            sub = _fixed_split(seg, cfg.chunk_size, cfg.chunk_overlap)
        else:
            sub = [seg]
        for part in sub:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{source_path}::{idx}",
                    text=part.strip(),
                    source_path=source_path,
                    doc_type=doc_type,
                    index=idx,
                )
            )
    return chunks


# ── chunkers per doctype ───────────────────────────────────────────────────────

def chunk_markdown(text: str, source_path: str) -> list[Chunk]:
    """Split on Markdown headers (##, ###) while keeping content with its header."""
    sections = re.split(r"(?m)^(?=#{1,4} )", text)
    return _make_chunks(sections, source_path, "markdown")


def chunk_code(text: str, source_path: str) -> list[Chunk]:
    """Split Python/shell on top-level def/class/function boundaries."""
    ext = Path(source_path).suffix.lower()
    if ext in (".py",):
        pattern = r"(?m)^(?=(?:def |class |async def ))"
    elif ext in (".sh", ".bash"):
        pattern = r"(?m)^(?=[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*\)\s*\{)"
    else:
        # Generic: split on blank lines
        pattern = r"\n{2,}"
    sections = re.split(pattern, text)
    return _make_chunks(sections, source_path, "code")


def chunk_config(text: str, source_path: str) -> list[Chunk]:
    """Keep YAML/TOML/INI top-level stanzas together."""
    ext = Path(source_path).suffix.lower()
    if ext in (".yml", ".yaml"):
        # Split on top-level keys (no leading spaces)
        sections = re.split(r"(?m)^(?=[a-zA-Z_])", text)
    elif ext in (".toml",):
        sections = re.split(r"(?m)^(?=\[)", text)
    elif ext in (".ini", ".cfg", ".conf"):
        sections = re.split(r"(?m)^(?=\[)", text)
    elif ext in (".env",):
        # Group by blank-line-separated blocks
        sections = re.split(r"\n{2,}", text)
    else:
        sections = [text]
    return _make_chunks(sections, source_path, "config")


def chunk_plain(text: str, source_path: str) -> list[Chunk]:
    parts = _fixed_split(text, _cfg.chunk_size, _cfg.chunk_overlap)
    return _make_chunks(parts, source_path, "text")


# ── dispatch ───────────────────────────────────────────────────────────────────

_CODE_EXTS = {".py", ".sh", ".bash", ".js", ".ts", ".go", ".rb", ".rs"}
_CONFIG_EXTS = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".env", ".json"}
_MARKDOWN_EXTS = {".md", ".mdx", ".rst"}


def chunk_document(text: str, source_path: str) -> list[Chunk]:
    ext = Path(source_path).suffix.lower()
    if ext in _MARKDOWN_EXTS:
        return chunk_markdown(text, source_path)
    if ext in _CODE_EXTS:
        return chunk_code(text, source_path)
    if ext in _CONFIG_EXTS:
        return chunk_config(text, source_path)
    return chunk_plain(text, source_path)
