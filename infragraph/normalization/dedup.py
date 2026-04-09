"""
Entity deduplication.

Strategy (layered):
  1. Exact name/alias match → same entity
  2. Normalised string match (lowercase, strip punctuation) → same entity
  3. Embedding cosine similarity above threshold → candidate duplicate
     (requires embedder, so only used when called explicitly)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from infragraph.extraction.schema import ExtractedEntity


def _normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^a-z0-9\-\._]", "", s)
    return s.strip("-_.")


@dataclass
class EntityGroup:
    canonical: ExtractedEntity
    members: list[ExtractedEntity] = field(default_factory=list)

    def all_names(self) -> set[str]:
        names: set[str] = {self.canonical.name}
        names.update(self.canonical.aliases)
        for m in self.members:
            names.add(m.name)
            names.update(m.aliases)
        return names

    def merged(self) -> ExtractedEntity:
        """Return a single merged entity with all aliases and highest confidence."""
        all_aliases = list(self.all_names() - {self.canonical.name})
        merged_props = dict(self.canonical.properties)
        for m in self.members:
            merged_props.update(m.properties)
        return ExtractedEntity(
            name=self.canonical.name,
            type=self.canonical.type,
            aliases=all_aliases,
            properties=merged_props,
            confidence=max(
                self.canonical.confidence,
                *(m.confidence for m in self.members),
                0.0,
            ),
        )


def deduplicate(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """
    Group entities by exact and normalised name overlap.
    Returns one merged entity per group.
    """
    groups: list[EntityGroup] = []
    # Map normalised name → group index
    norm_index: dict[str, int] = {}

    for entity in entities:
        all_names = {entity.name} | set(entity.aliases)
        norm_names = {_normalise(n) for n in all_names}

        # Find any existing group that overlaps
        matched_idx: int | None = None
        for nn in norm_names:
            if nn in norm_index:
                matched_idx = norm_index[nn]
                break

        if matched_idx is not None:
            group = groups[matched_idx]
            group.members.append(entity)
            # Register new norm names → same group
            for nn in norm_names:
                norm_index.setdefault(nn, matched_idx)
        else:
            idx = len(groups)
            groups.append(EntityGroup(canonical=entity))
            for nn in norm_names:
                norm_index[nn] = idx

    return [g.merged() for g in groups]


def make_entity_id(entity: ExtractedEntity) -> str:
    """Stable ID: type + normalised name."""
    return f"{entity.type.value}::{_normalise(entity.name)}"
