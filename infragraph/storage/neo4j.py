"""Neo4j client — connection, schema bootstrap, entity/relation CRUD."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

from infragraph.config import get_settings
from infragraph.ontology import (
    FULLTEXT_INDEX,
    UNIQUENESS_CONSTRAINTS,
    EntityType,
    RelationType,
)

log = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._driver: Driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=50,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def verify(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self._driver.session() as s:
            yield s

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        payload = dict(node)
        payload.setdefault("type", next(iter(node.labels), None))
        return payload

    # ── schema bootstrap ──────────────────────────────────────────────────────

    def bootstrap_schema(self) -> None:
        """Create uniqueness constraints and full-text index if missing."""
        with self.session() as s:
            existing = {
                r["name"]
                for r in s.run("SHOW CONSTRAINTS YIELD name").data()
            }
            for label, prop in UNIQUENESS_CONSTRAINTS:
                cname = f"unique_{label.lower()}_{prop}"
                if cname not in existing:
                    s.run(
                        f"CREATE CONSTRAINT {cname} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                    )
                    log.info("created constraint %s", cname)

            ft = FULLTEXT_INDEX
            ft_existing = {
                r["name"]
                for r in s.run("SHOW INDEXES YIELD name").data()
            }
            if ft["name"] not in ft_existing:
                labels = "|".join(ft["labels"])
                props  = ", ".join(f"n.{p}" for p in ft["properties"])
                s.run(
                    f"CREATE FULLTEXT INDEX `{ft['name']}` IF NOT EXISTS "
                    f"FOR (n:{labels}) ON EACH [{props}]"
                )
                log.info("created fulltext index %s", ft["name"])

    # ── entity operations ─────────────────────────────────────────────────────

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        name: str,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source_doc: str = "",
    ) -> dict[str, Any]:
        props = properties or {}
        props.update(
            {
                "id": entity_id,
                "name": name,
                "type": entity_type.value,
                "aliases": aliases or [],
                "confidence": confidence,
                "source_doc": source_doc,
            }
        )
        query = (
            f"MERGE (n:{entity_type.value} {{id: $id}}) "
            "SET n += $props "
            "RETURN n"
        )
        with self.session() as s:
            result = s.run(query, id=entity_id, props=props)
            return self._node_to_dict(result.single()["n"])

    def merge_aliases(self, canonical_id: str, alias_ids: list[str]) -> None:
        """Redirect alias nodes to canonical by setting a redirect property."""
        with self.session() as s:
            for aid in alias_ids:
                s.run(
                    "MATCH (a {id: $aid}), (c {id: $cid}) "
                    "MERGE (a)-[:ALIAS_OF]->(c)",
                    aid=aid,
                    cid=canonical_id,
                )

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            result = s.run("MATCH (n {id: $id}) RETURN n LIMIT 1", id=entity_id)
            rec = result.single()
            return self._node_to_dict(rec["n"]) if rec else None

    def find_entities_by_name(
        self, name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            result = s.run(
                "CALL db.index.fulltext.queryNodes($idx, $q) "
                "YIELD node, score "
                "RETURN node, score ORDER BY score DESC LIMIT $limit",
                idx=FULLTEXT_INDEX["name"],
                q=name,
                limit=limit,
            )
            return [{"node": self._node_to_dict(r["node"]), "score": r["score"]} for r in result]

    # ── relation operations ───────────────────────────────────────────────────

    def upsert_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationType,
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source_doc: str = "",
    ) -> None:
        props = properties or {}
        props.update({"confidence": confidence, "source_doc": source_doc})
        query = (
            "MATCH (a {id: $src}), (b {id: $tgt}) "
            f"MERGE (a)-[r:{rel_type.value}]->(b) "
            "SET r += $props"
        )
        with self.session() as s:
            s.run(query, src=source_id, tgt=target_id, props=props)

    # ── graph traversal ───────────────────────────────────────────────────────

    def neighbours(
        self,
        entity_id: str,
        depth: int = 2,
        rel_types: list[RelationType] | None = None,
    ) -> list[dict[str, Any]]:
        rel_filter = ""
        if rel_types:
            types = "|".join(r.value for r in rel_types)
            rel_filter = f":{types}"

        query = (
            f"MATCH path = (n {{id: $id}})-[{rel_filter}*1..{depth}]-(m) "
            "RETURN nodes(path) AS nodes, relationships(path) AS rels"
        )
        with self.session() as s:
            result = s.run(query, id=entity_id)
            rows = []
            for r in result:
                rows.append(
                    {
                        "nodes": [self._node_to_dict(n) for n in r["nodes"]],
                        "rels": [
                            {
                                "type": rel.type,
                                "props": dict(rel),
                                "start": rel.start_node["id"],
                                "end": rel.end_node["id"],
                            }
                            for rel in r["rels"]
                        ],
                    }
                )
            return rows

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        with self.session() as s:
            return [dict(r) for r in s.run(query, **(params or {}))]

    def delete_entity(self, entity_id: str) -> dict[str, Any]:
        """Delete a specific entity node and all its relationships by ID."""
        with self.session() as s:
            result = s.run(
                "MATCH (n {id: $id}) "
                "WITH n, count { (n)--() } AS rel_count "
                "DETACH DELETE n "
                "RETURN rel_count",
                id=entity_id,
            )
            rec = result.single()
            return {
                "deleted": rec is not None,
                "entity_id": entity_id,
                "relations_removed": rec["rel_count"] if rec else 0,
            }

    def delete_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
    ) -> dict[str, Any]:
        """Delete relation(s) between two entity IDs. Optionally filter by type."""
        if rel_type:
            query = (
                "MATCH (a {id: $src})-[r:" + rel_type + "]->(b {id: $tgt}) "
                "DELETE r RETURN count(r) AS removed"
            )
        else:
            query = (
                "MATCH (a {id: $src})-[r]->(b {id: $tgt}) "
                "DELETE r RETURN count(r) AS removed"
            )
        with self.session() as s:
            result = s.run(query, src=source_id, tgt=target_id)
            rec = result.single()
            return {"relations_removed": rec["removed"] if rec else 0}
