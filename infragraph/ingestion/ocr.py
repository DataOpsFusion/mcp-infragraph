"""
OCR via OpenRouter multimodal API — zero local dependencies.

PDFs:
  - file-parser plugin with 'pdf-text' engine (free, embedded text)
  - auto-upgrades to 'mistral-ocr' if extracted text is too sparse

Images (.png, .jpg, .jpeg, .tiff, .bmp, .webp):
  - sent as base64 image_url to the extract_model (multimodal)
  - model is prompted to transcribe all visible text verbatim

Nothing is installed on this machine. All heavy lifting happens on OpenRouter.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import NamedTuple

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infragraph.config import get_settings

log = logging.getLogger(__name__)

# ── file type routing ──────────────────────────────────────────────────────────

OCR_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
OCR_PDF_EXT    = ".pdf"
OCR_ALL_EXTS   = OCR_IMAGE_EXTS | {OCR_PDF_EXT}

# Chars-per-page threshold below which we upgrade to mistral-ocr
_SPARSE_THRESHOLD = 80


def is_ocr_candidate(path: str | Path) -> bool:
    return Path(path).suffix.lower() in OCR_ALL_EXTS


# ── result type ────────────────────────────────────────────────────────────────

class OcrResult(NamedTuple):
    text: str
    page_count: int   # always 1 for images
    method: str       # "pdf-text" | "mistral-ocr" | "image-multimodal"
    warnings: list[str]


# ── shared HTTP call ───────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
)
def _call_openrouter(payload: dict) -> str:
    cfg = get_settings()
    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/DataOpsFusion/infragraph",
        "X-Title": "infragraph-ocr",
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── PDF extraction ─────────────────────────────────────────────────────────────

def _pdf_payload(file_data: str, engine: str, model: str) -> dict:
    """
    file_data: either a public URL  OR  "data:application/pdf;base64,<b64>"
    engine:    "pdf-text" | "mistral-ocr" | "native"
    """
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text from this document verbatim. "
                            "Preserve structure, headings, and lists. "
                            "Output plain text only — no commentary."
                        ),
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_data": file_data,
                        },
                    },
                ],
            }
        ],
        "plugins": [{"id": "file-parser", "pdf": {"engine": engine}}],
        "temperature": 0.0,
    }


def _extract_pdf(path: Path) -> OcrResult:
    return _extract_pdf_bytes(path.name, path.read_bytes())


def _extract_pdf_bytes(filename: str, file_bytes: bytes) -> OcrResult:
    cfg = get_settings()
    raw_b64 = base64.b64encode(file_bytes).decode()
    file_data = f"data:application/pdf;base64,{raw_b64}"
    warnings: list[str] = []

    # First pass: free pdf-text engine
    text = _call_openrouter(_pdf_payload(file_data, "pdf-text", cfg.extract_model))

    # Estimate page count (rough: every ~2000 chars ≈ 1 page)
    estimated_pages = max(1, len(text) // 2000)

    if len(text.strip()) < _SPARSE_THRESHOLD * estimated_pages:
        warnings.append(
            f"pdf-text returned sparse text ({len(text)} chars) — "
            "retrying with mistral-ocr"
        )
        text = _call_openrouter(_pdf_payload(file_data, "mistral-ocr", cfg.extract_model))
        method = "mistral-ocr"
    else:
        method = "pdf-text"

    return OcrResult(
        text=text,
        page_count=estimated_pages,
        method=method,
        warnings=warnings,
    )


# ── image extraction ───────────────────────────────────────────────────────────

def _extract_image(path: Path) -> OcrResult:
    return _extract_image_bytes(path.name, path.read_bytes())


def _extract_image_bytes(filename: str, file_bytes: bytes) -> OcrResult:
    cfg = get_settings()
    raw_b64 = base64.b64encode(file_bytes).decode()
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    data_url = f"data:{mime};base64,{raw_b64}"

    payload = {
        "model": cfg.extract_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe ALL visible text in this image verbatim. "
                            "Preserve layout as much as possible. "
                            "Output plain text only — no commentary."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "temperature": 0.0,
    }

    text = _call_openrouter(payload)
    return OcrResult(
        text=text,
        page_count=1,
        method="image-multimodal",
        warnings=[] if text.strip() else ["Model returned no text for this image"],
    )


# ── public entrypoint ──────────────────────────────────────────────────────────

def extract_text(path: str | Path) -> OcrResult:
    """
    Extract text from a PDF or image via OpenRouter.
    Raises on API errors or unsupported file type.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == OCR_PDF_EXT:
        result = _extract_pdf(p)
    elif ext in OCR_IMAGE_EXTS:
        result = _extract_image(p)
    else:
        raise ValueError(f"Not an OCR-supported file type: {ext}")

    for w in result.warnings:
        log.warning("[OCR] %s: %s", p.name, w)

    log.info(
        "[OCR] %s → %d chars via %s",
        p.name, len(result.text), result.method,
    )
    return result


def extract_text_bytes(filename: str, file_bytes: bytes) -> OcrResult:
    """
    Extract text from uploaded PDF/image bytes via OpenRouter.
    """
    ext = Path(filename).suffix.lower()

    if ext == OCR_PDF_EXT:
        result = _extract_pdf_bytes(filename, file_bytes)
    elif ext in OCR_IMAGE_EXTS:
        result = _extract_image_bytes(filename, file_bytes)
    else:
        raise ValueError(f"Not an OCR-supported file type: {ext}")

    for warning in result.warnings:
        log.warning("[OCR] %s: %s", filename, warning)

    log.info("[OCR] %s → %d chars via %s", filename, len(result.text), result.method)
    return result


def extract_text_safe(path: str | Path) -> str | None:
    """
    Like extract_text() but returns None on failure instead of raising.
    Used by the scanner so one bad file doesn't abort the whole run.
    """
    try:
        result = extract_text(path)
        return result.text if result.text.strip() else None
    except Exception as exc:
        log.warning("[OCR] failed on %s: %s", path, exc)
        return None
