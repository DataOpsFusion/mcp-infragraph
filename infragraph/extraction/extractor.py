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

## Relation selection rules
- RUNS_ON: service/container is deployed on a host or VM.
- HOSTS: host/VM contains a service or container.
- PROXIED_BY: traffic to a service/domain goes through a proxy (nginx, traefik).
- RESOLVES_TO: a domain name maps to an IP or host.
- SUBDOMAIN_OF: a domain is a subdomain of a parent domain. Use instead of RELATED_TO_INCIDENT for domain hierarchy.
- CONNECTS_TO: a service calls or communicates with another service over the network.
- STORES_DATA_IN: a service reads/writes to a database or storage system.
- BACKS_UP_TO: data is backed up to a target.
- REPLICATES_TO: a database replicates to a replica.
- DEPENDS_ON: logical or software-level dependency between services.
- PART_OF: entity belongs to a cluster, group, or project. Use instead of RELATED_TO_INCIDENT for membership.
- RELATED_TO_INCIDENT: use ONLY when the relationship is explicitly an incident or outage connection.
- ALIAS_OF: two entities refer to the same real-world thing.

## General rules
- Only extract entities that are explicitly mentioned.
- Do NOT invent entities or relationships.
- For Host/VM/Container entities, capture ip, hostname, os in properties.
- For Service entities, capture port, protocol, version if present.
- Assign confidence between 0.5 (uncertain) and 1.0 (explicit).
- Return ONLY valid JSON matching the schema below, no prose.
- Entity names must be semantic (hostnames, service names), NOT bare IDs.
  If text says "LXC 111 (db-docker)" use "db-docker" as name, store ct_id="111"
  as a property. Never name an entity "LXC 111", "CT 108", "VM 200" etc.
- Prefer hostnames over IP addresses for entity names. Use the hostname as name
  and store the IP as a property. Only use an IP as the name when no hostname
  is available in the surrounding context.

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
