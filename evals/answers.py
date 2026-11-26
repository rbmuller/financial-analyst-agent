"""Answer eval: run the whole agent on each case and grade what comes back.

Two kinds of check per case:

* programmatic, read from the run itself: did it cite sources, were all cited
  ids real, did it cite the expected filing, which tools did it use;
* judged: is the answer correct against the expected facts (see evals/judge.py).

A case whose run or grading raised is recorded as an error, not as a wrong
answer: a network failure says nothing about answer quality.
"""

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from statistics import median
from typing import Any, Protocol

from agent.loop import AgentResult
from agent.pricing import Pricing
from agent.tracing import RunOrigin, RunStatus, Step, StepType
from evals.cases import EvalCase
from evals.judge import Judge, Verdict

ORIGIN_EVAL = "eval"
STATUS_GRADED = "graded"
STATUS_ERROR = "error"


class EvaluatedAgent(Protocol):
    def run(self, question: str, origin: RunOrigin, on_step: Any = None) -> AgentResult: ...


@dataclass
class CaseResult:
    case_id: str
    kind: str
    question: str
    status: str = STATUS_GRADED
    error: str | None = None
    answer: str = ""
    answer_words: int = 0
    verdict: str | None = None
    judge_reasoning: str = ""
    missing_facts: list[str] = field(default_factory=list)
    contradicted_facts: list[str] = field(default_factory=list)
    run_status: str | None = None
    cited_sources: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    cited_expected_filing: bool | None = None  # None when the case has no expected filing
    tools_used: list[str] = field(default_factory=list)
    steps: int = 0
    latency_ms: int = 0
    agent_cost_usd: float | None = None
    judge_cost_usd: float | None = None
    run_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_case(
    case: EvalCase, agent: EvaluatedAgent, judge: Judge, pricing: Pricing
) -> CaseResult:
    result = CaseResult(case_id=case.id, kind=case.kind.value, question=case.question)
    tools_used: list[str] = []

    def record_tool(step: Step) -> None:
        if step.type is StepType.TOOL and step.tool_name:
            tools_used.append(step.tool_name)

    try:
        run = agent.run(case.question, RunOrigin(source=ORIGIN_EVAL), on_step=record_tool)
        result.answer = run.answer
        result.answer_words = len(run.answer.split())
        result.run_status = run.status.value
        result.run_id = run.run_id
        result.cited_sources = [source.label for source in run.sources]
        result.unknown_citations = run.unknown_citations
        result.tools_used = tools_used
        result.steps = run.steps
        result.latency_ms = run.latency_ms
        result.agent_cost_usd = float(run.cost_usd) if run.cost_usd is not None else None
        if case.retrieval is not None:
            result.cited_expected_filing = any(
                case.retrieval.matches_citation(label) for label in result.cited_sources
            )
        if run.status is not RunStatus.OK:
            raise RuntimeError(f"agent run ended with status '{run.status.value}'")

        judgment, judge_usage = judge.grade(case, run.answer)
        judge_cost = pricing.cost_usd(judge.model, judge_usage)
        result.verdict = judgment.verdict.value
        result.judge_reasoning = judgment.reasoning
        result.missing_facts = judgment.missing_facts
        result.contradicted_facts = judgment.contradicted_facts
        result.judge_cost_usd = float(judge_cost) if judge_cost is not None else None
    except Exception as error:
        result.status = STATUS_ERROR
        result.error = f"{type(error).__name__}: {error}"
    return result


@dataclass(frozen=True)
class AnswerMetrics:
    cases: int
    errors: int
    answerable_graded: int
    correct: int  # answerable cases judged fully correct
    partially_correct: int
    incorrect: int
    unanswerable_graded: int
    abstained: int  # unanswerable cases where the agent said it was not available
    cited: int  # answerable cases with at least one resolved source
    with_unknown_citations: int
    expected_filing_cases: int
    cited_expected_filing: int
    median_answer_words: float
    max_answer_words: int
    median_latency_s: float
    median_steps: float
    median_agent_cost_usd: float | None
    total_cost_usd: float | None


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "n/a"


def summarize(results: list[CaseResult]) -> AnswerMetrics:
    graded = [r for r in results if r.status == STATUS_GRADED]
    answerable = [r for r in graded if r.kind != "unanswerable"]
    unanswerable = [r for r in graded if r.kind == "unanswerable"]
    with_target = [r for r in graded if r.cited_expected_filing is not None]
    agent_costs = [r.agent_cost_usd for r in graded if r.agent_cost_usd is not None]
    all_costs = [
        Decimal(str(cost))
        for r in results
        for cost in (r.agent_cost_usd, r.judge_cost_usd)
        if cost is not None
    ]

    def count(rows: list[CaseResult], verdict: Verdict) -> int:
        return sum(1 for r in rows if r.verdict == verdict.value)

    return AnswerMetrics(
        cases=len(results),
        errors=len(results) - len(graded),
        answerable_graded=len(answerable),
        correct=count(answerable, Verdict.CORRECT),
        partially_correct=count(answerable, Verdict.PARTIALLY_CORRECT),
        incorrect=count(answerable, Verdict.INCORRECT),
        unanswerable_graded=len(unanswerable),
        abstained=count(unanswerable, Verdict.CORRECT),
        cited=sum(1 for r in answerable if r.cited_sources),
        with_unknown_citations=sum(1 for r in graded if r.unknown_citations),
        expected_filing_cases=len(with_target),
        cited_expected_filing=sum(1 for r in with_target if r.cited_expected_filing),
        median_answer_words=median(r.answer_words for r in graded) if graded else 0.0,
        max_answer_words=max((r.answer_words for r in graded), default=0),
        median_latency_s=median(r.latency_ms for r in graded) / 1000 if graded else 0.0,
        median_steps=median(r.steps for r in graded) if graded else 0.0,
        median_agent_cost_usd=median(agent_costs) if agent_costs else None,
        total_cost_usd=float(sum(all_costs)) if all_costs else None,
    )


def format_summary(metrics: AnswerMetrics, agent_model: str, judge_model: str) -> str:
    def cost(value: float | None) -> str:
        return f"${value:.2f}" if value is not None else "n/a"

    lines = [
        f"Agent model: {agent_model}   Judge model: {judge_model}",
        f"Cases: {metrics.cases} ({metrics.errors} errored and excluded from the rates below)",
        "",
        f"Correct (answerable cases):        {_share(metrics.correct, metrics.answerable_graded)}"
        f"   partially correct: {metrics.partially_correct}   incorrect: {metrics.incorrect}",
        f"Abstained (unanswerable cases):    "
        f"{_share(metrics.abstained, metrics.unanswerable_graded)}",
        f"Cited a source (answerable cases): {_share(metrics.cited, metrics.answerable_graded)}",
        f"Cited the expected filing:         "
        f"{_share(metrics.cited_expected_filing, metrics.expected_filing_cases)}",
        f"Runs citing an id no tool returned: {metrics.with_unknown_citations}",
        "",
        f"Answer length: median {metrics.median_answer_words:.0f} words, "
        f"longest {metrics.max_answer_words}",
        f"Median per question: {metrics.median_latency_s:.1f}s, {metrics.median_steps:.0f} steps, "
        f"{cost(metrics.median_agent_cost_usd)}",
        f"Total cost of this eval run (agent + judge): {cost(metrics.total_cost_usd)}",
    ]
    return "\n".join(lines)
