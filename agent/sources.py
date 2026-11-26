"""Sources an agent run may cite, and the check that it only cites those.

Tools hand every retrieved passage or query result to the registry and get back
a short id (S1, S2 for filing passages; D1 for data queries). The model cites
ids inline, and the final source list is built here from the registry, never
from text the model wrote. An id the registry does not know is a fabricated
citation and is reported as such.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

CITATION_GROUP = re.compile(r"\[((?:[SD]\d+)(?:\s*[,;]\s*[SD]\d+)*)\]")
CITATION_ID = re.compile(r"[SD]\d+")


class SourceKind(StrEnum):
    FILING = "S"  # a passage from a 10-K / 10-Q
    DATA = "D"  # a SQL query over XBRL financial data


@dataclass(frozen=True)
class Source:
    id: str
    kind: SourceKind
    label: str  # human-readable citation
    url: str | None = None
    detail: str | None = None  # the SQL, for data sources


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._ids_by_key: dict[tuple[SourceKind, str], str] = {}

    def add(
        self,
        kind: SourceKind,
        key: str,
        label: str,
        url: str | None = None,
        detail: str | None = None,
    ) -> str:
        """Register a source and return its id. The same key always gets the same id."""
        existing = self._ids_by_key.get((kind, key))
        if existing:
            return existing
        count = sum(1 for source in self._sources.values() if source.kind is kind)
        source_id = f"{kind.value}{count + 1}"
        self._sources[source_id] = Source(source_id, kind, label, url, detail)
        self._ids_by_key[(kind, key)] = source_id
        return source_id

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)


def cited_ids(answer: str) -> list[str]:
    """Citation ids in order of first appearance: "[S1]" and "[S1, D2]" forms."""
    ids: list[str] = []
    for group in CITATION_GROUP.findall(answer):
        for source_id in CITATION_ID.findall(group):
            if source_id not in ids:
                ids.append(source_id)
    return ids


def resolve_citations(answer: str, registry: SourceRegistry) -> tuple[list[Source], list[str]]:
    """Split the answer's citations into (known sources, unknown ids)."""
    sources: list[Source] = []
    unknown: list[str] = []
    for source_id in cited_ids(answer):
        source = registry.get(source_id)
        if source is None:
            unknown.append(source_id)
        else:
            sources.append(source)
    return sources, unknown


def strip_citations(answer: str) -> str:
    """Remove citation markers, e.g. before replaying an old answer as context.

    Source ids are numbered per run, so an id from an earlier answer would point
    at a different source (or none) in the next one.
    """
    return CITATION_GROUP.sub("", answer).replace("  ", " ").strip()
