from datetime import date
from decimal import Decimal
from typing import Any

from agent.loop import AgentResult
from agent.pricing import ModelPrice, Pricing, TokenUsage
from agent.sources import Source, SourceKind
from agent.tracing import RunOrigin, RunStatus, Step, StepType
from evals.answers import STATUS_ERROR, STATUS_GRADED, evaluate_case, summarize
from evals.cases import CaseKind, EvalCase, RetrievalTarget, load_cases
from evals.judge import Judgment, Verdict, build_judge_prompt
from evals.retrieval import summarize as summarize_retrieval
from rag.search import SearchFilters, SearchResult

JUDGE_MODEL = "judge-model"
PRICING = Pricing(
    {JUDGE_MODEL: ModelPrice(Decimal(1), Decimal(1), Decimal(1))}, cache_write_multiplier=Decimal(1)
)
TARGET = RetrievalTarget(
    query="gross margin",
    filters=SearchFilters(tickers=["NKE"]),
    ticker="NKE",
    phrases=("300 basis points lower",),
    form="10-Q",
    fiscal_year=2026,
    fiscal_period="Q2",
)
EXPECTED_LABEL = "NIKE, Inc. (NKE), 10-Q FY2026 Q2, filed 2025-12-30, Part I, Item 2: MD&A"


def passage(content: str, **overrides: Any) -> SearchResult:
    values: dict[str, Any] = {
        "chunk_id": 1, "score": 0.5, "vector_rank": 1, "keyword_rank": 1, "ticker": "NKE",
        "company": "NIKE, Inc.", "form": "10-Q", "fiscal_year": 2026, "fiscal_period": "Q2",
        "filing_date": date(2025, 12, 30), "item": "Part I, Item 2", "title": "MD&A",
        "source_url": "https://sec.test/doc", "accession_number": "a-1", "content": content,
    }  # fmt: skip
    return SearchResult(**{**values, **overrides})


def make_case(
    kind: CaseKind = CaseKind.NARRATIVE, target: RetrievalTarget | None = TARGET
) -> EvalCase:
    return EvalCase("case-1", kind, "Why?", ("Margin fell 300 bps.",), target)


class FakeAgent:
    def __init__(self, result: AgentResult | Exception) -> None:
        self._result = result

    def run(self, question: str, origin: RunOrigin, on_step: Any = None) -> AgentResult:
        if isinstance(self._result, Exception):
            raise self._result
        on_step(Step(StepType.LLM, 1))
        on_step(Step(StepType.TOOL, 1, tool_name="search_filings"))
        return self._result


class FakeJudge:
    model = JUDGE_MODEL

    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    def grade(self, case: EvalCase, answer: str) -> tuple[Judgment, TokenUsage]:
        judgment = Judgment(
            reasoning="because", missing_facts=[], contradicted_facts=[], verdict=self._verdict
        )
        return judgment, TokenUsage(input_tokens=1_000_000)


def agent_result(**overrides: Any) -> AgentResult:
    values: dict[str, Any] = {
        "run_id": "run-1", "status": RunStatus.OK, "answer": "Margin fell [S1].",
        "sources": [Source("S1", SourceKind.FILING, EXPECTED_LABEL)],
        "cost_usd": Decimal("0.05"), "latency_ms": 4000, "steps": 3,
    }  # fmt: skip
    return AgentResult(**{**values, **overrides})


def test_dataset_loads_and_is_well_formed() -> None:
    cases = load_cases()

    assert len(cases) >= 20
    assert {case.kind for case in cases} == set(CaseKind)
    for case in cases:
        assert case.expected_facts, case.id
        needs_text = case.kind in (CaseKind.NARRATIVE, CaseKind.MIXED)
        assert (case.retrieval is not None) == needs_text, case.id


def test_relevance_needs_the_right_filing_and_a_key_phrase() -> None:
    text = "Consolidated gross margin was 300  basis points\nLOWER than the prior year."

    assert TARGET.is_relevant(passage(text))
    assert not TARGET.is_relevant(passage("Gross margin was discussed."))
    assert not TARGET.is_relevant(passage(text, fiscal_period="Q3"))
    assert not TARGET.is_relevant(passage(text, ticker="AAPL"))


def test_citation_match_follows_the_specified_filing_fields() -> None:
    any_nike_filing = RetrievalTarget("q", SearchFilters(), "NKE", ("x",))

    assert TARGET.matches_citation(EXPECTED_LABEL)
    assert not TARGET.matches_citation(EXPECTED_LABEL.replace("Q2", "Q3"))
    assert not TARGET.matches_citation("SEC XBRL financial data: NKE 10-Q filed 2025-12-30")
    assert any_nike_filing.matches_citation(EXPECTED_LABEL.replace("Q2", "Q3"))


def test_retrieval_metrics() -> None:
    metrics = summarize_retrieval("hybrid", [1, 2, None, 6])

    assert metrics.hit_rate == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}
    assert metrics.mrr == (1 + 0.5 + 0 + 1 / 6) / 4


def test_judge_prompt_marks_unanswerable_cases() -> None:
    answerable = build_judge_prompt(make_case(), "an answer")
    unanswerable = build_judge_prompt(make_case(CaseKind.UNANSWERABLE, None), "an answer")

    assert "UNANSWERABLE" not in answerable
    assert unanswerable.startswith("This question is UNANSWERABLE")
    assert "<answer>\nan answer\n</answer>" in answerable
    assert "- Margin fell 300 bps." in answerable


def test_evaluate_case_combines_run_checks_and_judgment() -> None:
    result = evaluate_case(
        make_case(), FakeAgent(agent_result()), FakeJudge(Verdict.CORRECT), PRICING
    )

    assert (result.status, result.verdict) == (STATUS_GRADED, "correct")
    assert result.cited_expected_filing is True
    assert result.tools_used == ["search_filings"]
    assert result.answer_words == 3
    assert (result.agent_cost_usd, result.judge_cost_usd) == (0.05, 1.0)


def test_failures_are_errors_not_wrong_answers() -> None:
    crashed = evaluate_case(
        make_case(), FakeAgent(RuntimeError("network")), FakeJudge(Verdict.CORRECT), PRICING
    )
    cut_off = evaluate_case(
        make_case(),
        FakeAgent(agent_result(status=RunStatus.STEP_LIMIT)),
        FakeJudge(Verdict.CORRECT),
        PRICING,
    )

    assert (crashed.status, crashed.verdict) == (STATUS_ERROR, None)
    assert "network" in (crashed.error or "")
    assert (cut_off.status, cut_off.verdict) == (STATUS_ERROR, None)
    assert "step_limit" in (cut_off.error or "")


def test_summary_separates_answerable_unanswerable_and_errors() -> None:
    def graded(kind: CaseKind, verdict: Verdict, **agent_overrides: Any) -> Any:
        target = TARGET if kind is CaseKind.NARRATIVE else None
        return evaluate_case(
            make_case(kind, target),
            FakeAgent(agent_result(**agent_overrides)),
            FakeJudge(verdict),
            PRICING,
        )

    results = [
        graded(CaseKind.NARRATIVE, Verdict.CORRECT),
        graded(CaseKind.NARRATIVE, Verdict.PARTIALLY_CORRECT, sources=[]),
        graded(CaseKind.NUMERIC, Verdict.INCORRECT, unknown_citations=["S9"]),
        graded(CaseKind.UNANSWERABLE, Verdict.CORRECT, sources=[]),
        graded(CaseKind.UNANSWERABLE, Verdict.INCORRECT, sources=[]),
        evaluate_case(
            make_case(), FakeAgent(RuntimeError("x")), FakeJudge(Verdict.CORRECT), PRICING
        ),
    ]

    metrics = summarize(results)

    assert (metrics.cases, metrics.errors) == (6, 1)
    assert (metrics.answerable_graded, metrics.correct) == (3, 1)
    assert (metrics.partially_correct, metrics.incorrect) == (1, 1)
    assert (metrics.unanswerable_graded, metrics.abstained) == (2, 1)
    assert metrics.cited == 2
    assert metrics.with_unknown_citations == 1
    assert (metrics.expected_filing_cases, metrics.cited_expected_filing) == (2, 1)
    assert metrics.median_latency_s == 4.0
    assert (metrics.median_answer_words, metrics.max_answer_words) == (3, 3)
    assert metrics.total_cost_usd == 5 * 0.05 + 5 * 1.0
