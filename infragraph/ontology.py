"""Canonical ontology: entity types, relation types, and Neo4j schema constraints."""
from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    HOST         = "Host"
    VM           = "VM"
    CONTAINER    = "Container"
    SERVICE      = "Service"
    PROJECT      = "Project"
    REPOSITORY   = "Repository"
    DOCUMENT     = "Document"
    RUNBOOK      = "Runbook"
    DOMAIN       = "Domain"
    PORT         = "Port"
    INCIDENT     = "Incident"
    PERSON       = "Person"
    CREDENTIAL_REF = "CredentialRef"


class RelationType(StrEnum):
    RUNS_ON            = "RUNS_ON"
    HOSTS              = "HOSTS"
    DEPENDS_ON         = "DEPENDS_ON"
    PROXIED_BY         = "PROXIED_BY"
    RESOLVES_TO        = "RESOLVES_TO"
    DOCUMENTS          = "DOCUMENTS"
    OWNED_BY           = "OWNED_BY"
    MANAGED_BY         = "MANAGED_BY"
    EXPOSES            = "EXPOSES"
    USES_MODEL         = "USES_MODEL"
    STORES_DATA_IN     = "STORES_DATA_IN"
    RELATED_TO_INCIDENT = "RELATED_TO_INCIDENT"


# Neo4j constraints to run on first startup
# Each tuple: (label, property) — creates a uniqueness constraint
UNIQUENESS_CONSTRAINTS: list[tuple[str, str]] = [
    (e.value, "id") for e in EntityType
]

# Full-text search index across entity name + aliases
FULLTEXT_INDEX = {
    "name": "entity_name_ft",
    "labels": [e.value for e in EntityType],
    "properties": ["name", "aliases"],
}

# Relation directionality hints used by graph traversal
RELATION_DIRECTION: dict[str, str] = {
    RelationType.RUNS_ON:            "outbound",
    RelationType.HOSTS:              "inbound",
    RelationType.DEPENDS_ON:         "outbound",
    RelationType.PROXIED_BY:         "outbound",
    RelationType.RESOLVES_TO:        "outbound",
    RelationType.DOCUMENTS:          "outbound",
    RelationType.OWNED_BY:           "outbound",
    RelationType.MANAGED_BY:         "outbound",
    RelationType.EXPOSES:            "outbound",
    RelationType.USES_MODEL:         "outbound",
    RelationType.STORES_DATA_IN:     "outbound",
    RelationType.RELATED_TO_INCIDENT: "both",
}
