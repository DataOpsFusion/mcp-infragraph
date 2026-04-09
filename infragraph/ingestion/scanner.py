"""
Filesystem scanner — walk paths, detect encoding, return (path, text) pairs.

OCR routing:
  - PDF / image files are handed to ocr.extract_text_safe()
  - All other files are read as UTF-8/detected encoding
  - Files that yield no text from either path are silently skipped
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chardet

from infragraph.ingestion.ocr import OCR_ALL_EXTS, is_ocr_candidate, extract_text_safe

log = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
}

_SKIP_EXTS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".class",
    # Images NOT in OCR_ALL_EXTS (decorative / non-document)
    ".gif", ".svg", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".whl",
    ".lock",
}

_MAX_TEXT_FILE_BYTES = 512 * 1024   # 512 KB for plain text
_MAX_OCR_FILE_BYTES  = 50 * 1024 * 1024  # 50 MB for PDFs/images


@dataclass
class ScanResult:
    path: str
    text: str
    via_ocr: bool = False


def _read_text_file(path: Path) -> str | None:
    if path.stat().st_size > _MAX_TEXT_FILE_BYTES:
        log.debug("skipping large text file: %s", path)
        return None
    raw = path.read_bytes()
    non_printable = sum(1 for b in raw[:4096] if b < 9 or (13 < b < 32))
    if non_printable / max(len(raw[:4096]), 1) > 0.30:
        return None
    detected = chardet.detect(raw)
    enc = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def _handle_file(path: Path) -> ScanResult | None:
    ext = path.suffix.lower()

    if is_ocr_candidate(path):
        if path.stat().st_size > _MAX_OCR_FILE_BYTES:
            log.warning("skipping oversized OCR file (%d MB): %s",
                        path.stat().st_size // (1024*1024), path)
            return None
        text = extract_text_safe(path)
        if not text:
            log.debug("OCR yielded no text: %s", path)
            return None
        return ScanResult(path=str(path), text=text, via_ocr=True)

    text = _read_text_file(path)
    if not text:
        return None
    return ScanResult(path=str(path), text=text, via_ocr=False)


def scan_paths(
    paths: list[str],
    include_ocr: bool = True,
) -> list[tuple[str, str]]:
    """
    Recursively scan all paths.
    Returns list of (absolute_path_str, text_content).

    Args:
        paths:       Filesystem paths to scan (files or directories).
        include_ocr: If False, skip PDFs and images entirely.
    """
    results: list[tuple[str, str]] = []
    ocr_count = 0
    text_count = 0

    for root_str in paths:
        root = Path(root_str).expanduser().resolve()
        if not root.exists():
            log.warning("ingest path does not exist: %s", root)
            continue

        candidates = [root] if root.is_file() else root.rglob("*")

        for path in candidates:
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            ext = path.suffix.lower()
            if ext in _SKIP_EXTS:
                continue
            if not include_ocr and ext in OCR_ALL_EXTS:
                continue

            result = _handle_file(path)
            if result:
                results.append((result.path, result.text))
                if result.via_ocr:
                    ocr_count += 1
                else:
                    text_count += 1

    log.info(
        "scanner: %d text files, %d OCR files across %d path(s)",
        text_count, ocr_count, len(paths),
    )
    return results
