"""
Extraction service — calls mistral-large-2512 via OpenRouter to extract
entities and relations from a text chunk, returning a validated ExtractionResult.
"""
from __future__ import annotations

import json
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from infragraph.config import get_settings
from infragraph.extraction.schema import EXTRACTION_JSON_SCHEMA, ExtractionResult
from infragraph.ontology import EntityType, RelationType

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""You are an infrastructure knowledge extraction engine.

Given a text chunk from a document (runbook, config file, README, etc.),
extract ALL named infrastructure entities and the relationships between them.

## Entity types you may extract
{", ".join(e.value for e in EntityType)}

## Relation types you may use
{", ".join(r.value for r in RelationType)}

## Rules
- Only extract entities that are explicitly mentioned.
- Do NOT invent entities or relationships.
- For Host/VM/Container entities, capture ip, hostname, os in properties.
- For Service entities, capture port, protocol, version if present.
- Assign confidence between 0.5 (uncertain) and 1.0 (explicit).
- Return ONLY valid JSON matching the schema below, no prose.

## Output JSON schema
{json.dumps(EXTRACTION_JSON_SCHEMA, indent=2)}
"""


class Extractor:
    def __init__(self) -> None:
        cfg = get_settings()
        self._model = cfg.extract_model
        self._base_url = cfg.openrouter_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DataOpsFusion/infragraph",
            "X-Title": "infragraph",
        }

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def extract(self, text: str, source_doc: str = "", chunk_id: str = "") -> ExtractionResult:
        if not text.strip():
            return ExtractionResult(source_doc=source_doc, chunk_id=chunk_id)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract from:\n\n{text}"},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to salvage if model wrapped in markdown code block
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
            if match:
                data = json.loads(match.group(1))
            else:
                raise

        result = ExtractionResult.model_validate(data)
        result.source_doc = source_doc
        result.chunk_id = chunk_id
        result.validate_relations()
        log.debug(
            "extracted %d entities, %d relations from %s",
            len(result.entities),
            len(result.relations),
            chunk_id,
        )
        return result
