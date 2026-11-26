"""Eval cases: the questions, what a correct answer contains, and what retrieval must find."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from rag.search import SearchFilters, SearchResult

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


class CaseKind(StrEnum):
    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    MIXED = "mixed"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class RetrievalTarget:
    query: str
    filters: SearchFilters
    ticker: str
    phrases: tuple[str, ...]
    form: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None

    def is_relevant(self, result: SearchResult) -> bool:
        """Right filing (as far as specified) and contains one of the key phrases."""
        normalized = " ".join(result.content.split()).lower()
        return (
            result.ticker == self.ticker
            and self.form in (None, result.form)
            and self.fiscal_year in (None, result.fiscal_year)
            and self.fiscal_period in (None, result.fiscal_period)
            and any(phrase.lower() in normalized for phrase in self.phrases)
        )

    def matches_citation(self, label: str) -> bool:
        """Whether a cited source label points at the expected filing."""
        period = " ".join(
            part
            for part in (
                self.form,
                f"FY{self.fiscal_year}" if self.fiscal_year else None,
                self.fiscal_period,
            )
            if part
        )
        return f"({self.ticker})" in label and (not self.form or f", {period}" in label)


@dataclass(frozen=True)
class EvalCase:
    id: str
    kind: CaseKind
    question: str
    expected_facts: tuple[str, ...]
    retrieval: RetrievalTarget | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def answerable(self) -> bool:
        return self.kind is not CaseKind.UNANSWERABLE


def _retrieval_target(raw: dict[str, Any]) -> RetrievalTarget:
    filters = raw.get("filters", {})
    return RetrievalTarget(
        query=raw["query"],
        filters=SearchFilters(
            tickers=list(filters.get("tickers", [])),
            fiscal_years=list(filters.get("fiscal_years", [])),
            forms=list(filters.get("forms", [])),
        ),
        ticker=raw["ticker"],
        phrases=tuple(raw["phrases"]),
        form=raw.get("form"),
        fiscal_year=raw.get("fiscal_year"),
        fiscal_period=raw.get("fiscal_period"),
    )


def load_cases(path: Path = DATASET_PATH) -> list[EvalCase]:
    payload = yaml.safe_load(path.read_text())
    cases = [
        EvalCase(
            id=raw["id"],
            kind=CaseKind(raw["kind"]),
            question=raw["question"],
            expected_facts=tuple(raw["expected_facts"]),
            retrieval=_retrieval_target(raw["retrieval"]) if "retrieval" in raw else None,
        )
        for raw in payload["cases"]
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ids in the eval dataset.")
    return cases
