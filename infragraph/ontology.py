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
    # Deployment / hosting
    RUNS_ON            = "RUNS_ON"          # Service/Container -> Host/VM
    HOSTS              = "HOSTS"            # Host/VM -> Service/Container
    EXPOSES            = "EXPOSES"          # Service -> Port
    # Networking
    PROXIED_BY         = "PROXIED_BY"       # Service/Domain -> nginx/proxy Service
    RESOLVES_TO        = "RESOLVES_TO"      # Domain -> Host (IP)
    SUBDOMAIN_OF       = "SUBDOMAIN_OF"     # Domain -> parent Domain
    CONNECTS_TO        = "CONNECTS_TO"      # Service -> Service (network call)
    # Data
    STORES_DATA_IN     = "STORES_DATA_IN"   # Service -> DB/storage
    BACKS_UP_TO        = "BACKS_UP_TO"      # Service/Host -> backup target
    REPLICATES_TO      = "REPLICATES_TO"    # DB primary -> replica
    # Ownership / management
    OWNED_BY           = "OWNED_BY"         # entity -> Person/Project
    MANAGED_BY         = "MANAGED_BY"       # entity -> tool/system
    PART_OF            = "PART_OF"          # entity -> group/cluster/project
    # Dependencies / docs
    DEPENDS_ON         = "DEPENDS_ON"       # Service -> Service (logical dep)
    USES_MODEL         = "USES_MODEL"       # Service -> AI model
    DOCUMENTS          = "DOCUMENTS"        # Document/Runbook -> entity
    # Incidents / aliases
    RELATED_TO_INCIDENT = "RELATED_TO_INCIDENT"  # entity -> Incident
    ALIAS_OF           = "ALIAS_OF"         # duplicate entity -> canonical


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
    RelationType.SUBDOMAIN_OF:        "outbound",
    RelationType.CONNECTS_TO:         "outbound",
    RelationType.BACKS_UP_TO:         "outbound",
    RelationType.REPLICATES_TO:       "outbound",
    RelationType.PART_OF:             "outbound",
    RelationType.ALIAS_OF:            "outbound",
}
