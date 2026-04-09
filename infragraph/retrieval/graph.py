"""Graph retrieval — look up entities and traverse relationships in Neo4j."""
from __future__ import annotations

from typing import Any

from infragraph.ontology import EntityType, RelationType
from infragraph.storage.neo4j import Neo4jClient


class GraphRetriever:
    def __init__(self) -> None:
        self._neo4j = Neo4jClient()

    def find(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text entity search by name."""
        return self._neo4j.find_entities_by_name(name, limit=limit)

    def neighbours(
        self,
        entity_id: str,
        depth: int = 2,
        rel_types: list[RelationType] | None = None,
    ) -> list[dict[str, Any]]:
        return self._neo4j.neighbours(entity_id, depth=depth, rel_types=rel_types)

    def dependencies(self, service_name: str) -> list[dict[str, Any]]:
        """Return everything a service depends on (transitive)."""
        query = (
            "MATCH (s:Service {name: $name})-[:DEPENDS_ON*1..4]->(dep) "
            "RETURN dep.name AS name, dep.type AS type, dep.id AS id"
        )
        return self._neo4j.run_cypher(query, {"name": service_name})

    def runbooks_for(self, entity_name: str) -> list[dict[str, Any]]:
        """Find runbooks that document a given entity."""
        query = (
            "MATCH (r:Runbook)-[:DOCUMENTS]->(e) "
            "WHERE e.name = $name OR $name IN e.aliases "
            "RETURN r.name AS name, r.source_doc AS path, r.id AS id"
        )
        return self._neo4j.run_cypher(query, {"name": entity_name})

    def incidents_for(self, entity_name: str) -> list[dict[str, Any]]:
        query = (
            "MATCH (i:Incident)-[:RELATED_TO_INCIDENT]-(e) "
            "WHERE e.name = $name OR $name IN e.aliases "
            "RETURN i.name AS name, i.id AS id, i.confidence AS confidence "
            "ORDER BY i.confidence DESC LIMIT 10"
        )
        return self._neo4j.run_cypher(query, {"name": entity_name})

    def hosts_on(self, host_name: str) -> list[dict[str, Any]]:
        """What runs on a given host?"""
        query = (
            "MATCH (h:Host {name: $name})<-[:RUNS_ON]-(n) "
            "RETURN n.name AS name, labels(n)[0] AS type "
            "UNION "
            "MATCH (h:Host {name: $name})-[:HOSTS]->(n) "
            "RETURN n.name AS name, labels(n)[0] AS type"
        )
        return self._neo4j.run_cypher(query, {"name": host_name})

    def summary_stats(self) -> dict[str, Any]:
        query = (
            "MATCH (n) "
            "RETURN labels(n)[0] AS type, count(*) AS count "
            "ORDER BY count DESC"
        )
        rows = self._neo4j.run_cypher(query)
        rel_q = "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC"
        rel_rows = self._neo4j.run_cypher(rel_q)
        return {"entities": rows, "relations": rel_rows}
