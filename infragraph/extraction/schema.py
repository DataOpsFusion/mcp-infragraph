"""Pydantic models for structured extraction output from Mistral."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from infragraph.ontology import EntityType, RelationType


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical name of the entity")
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Key facts: ip, version, os, port, url, etc.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)

    @model_validator(mode="after")
    def normalise_name(self) -> "ExtractedEntity":
        self.name = self.name.strip()
        self.aliases = [a.strip() for a in self.aliases if a.strip()]
        return self


class ExtractedRelation(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: RelationType
    properties: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    source_doc: str = ""
    chunk_id: str = ""

    def entity_names(self) -> set[str]:
        names: set[str] = set()
        for e in self.entities:
            names.add(e.name)
            names.update(e.aliases)
        return names

    def validate_relations(self) -> "ExtractionResult":
        """Drop relations whose source/target don't appear in entity list."""
        known = self.entity_names()
        self.relations = [
            r for r in self.relations
            if r.source in known and r.target in known
        ]
        return self


# JSON schema injected into the Mistral prompt
EXTRACTION_JSON_SCHEMA = ExtractionResult.model_json_schema()
