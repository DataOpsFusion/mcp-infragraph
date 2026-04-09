"""
Graph writer — normalises extraction results and writes canonical facts to Neo4j.
"""
from __future__ import annotations

import logging

from infragraph.extraction.schema import ExtractionResult
from infragraph.normalization.dedup import deduplicate, make_entity_id
from infragraph.storage.neo4j import Neo4jClient

log = logging.getLogger(__name__)


class GraphWriter:
    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    def write(self, result: ExtractionResult) -> dict[str, int]:
        """
        Deduplicate, then write entities + relations to Neo4j.
        Returns counts of written nodes and edges.
        """
        if not result.entities:
            return {"entities": 0, "relations": 0}

        # ── deduplicate ───────────────────────────────────────────────────────
        merged = deduplicate(result.entities)

        # Build name → canonical_id map (covers aliases too)
        name_to_id: dict[str, str] = {}
        for entity in merged:
            eid = make_entity_id(entity)
            name_to_id[entity.name] = eid
            for alias in entity.aliases:
                name_to_id[alias] = eid

        # ── write entities ────────────────────────────────────────────────────
        entities_written = 0
        for entity in merged:
            eid = make_entity_id(entity)
            try:
                self._neo4j.upsert_entity(
                    entity_id=eid,
                    entity_type=entity.type,
                    name=entity.name,
                    aliases=entity.aliases,
                    properties=entity.properties,
                    confidence=entity.confidence,
                    source_doc=result.source_doc,
                )
                entities_written += 1
            except Exception as exc:
                log.warning("failed to write entity %s: %s", eid, exc)

        # ── write relations ───────────────────────────────────────────────────
        relations_written = 0
        for rel in result.relations:
            src_id = name_to_id.get(rel.source)
            tgt_id = name_to_id.get(rel.target)
            if not src_id or not tgt_id:
                log.debug("skipping relation — unknown entity: %s → %s", rel.source, rel.target)
                continue
            try:
                self._neo4j.upsert_relation(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=rel.type,
                    properties=rel.properties,
                    confidence=rel.confidence,
                    source_doc=result.source_doc,
                )
                relations_written += 1
            except Exception as exc:
                log.warning("failed to write relation %s→%s: %s", src_id, tgt_id, exc)

        log.debug(
            "wrote %d entities, %d relations from %s",
            entities_written, relations_written, result.source_doc,
        )
        return {"entities": entities_written, "relations": relations_written}
