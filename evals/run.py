"""Run the evals.

Usage:
    python -m evals.run retrieval                 # free: no LLM calls
    python -m evals.run answers                   # paid: agent + judge on every case
    python -m evals.run answers --case num-aapl-revenue-fy26 txt-msft-openai-stake
    python -m evals.run answers --rerank          # same cases with the re-ranker on
    python -m evals.run judge-check               # paid, tiny: sanity-check the judge

Results are written to data/evals/<timestamp>-<name>/ (results.jsonl + summary.txt).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent.factory import build_agent
from agent.pricing import Pricing
from common.db import DbRole, connect
from common.logs import configure_logging
from common.settings import Settings, get_settings
from evals.answers import CaseResult, evaluate_case, format_summary, summarize
from evals.cases import CaseKind, EvalCase, load_cases
from evals.judge import ClaudeJudge, Verdict
from evals.retrieval import HIT_CUTOFFS, RetrievalMetrics, run_retrieval_eval
from rag.embeddings.local import SentenceTransformerEmbedder
from rag.rerank import CrossEncoderReranker

RESULTS_DIR = "evals"
COMMAND_RETRIEVAL = "retrieval"
COMMAND_ANSWERS = "answers"
COMMAND_JUDGE_CHECK = "judge-check"
EXIT_FAILED_CHECK = 1


def output_dir(settings: Settings, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = settings.data_dir / RESULTS_DIR / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_retrieval(
    metrics: list[RetrievalMetrics], ranks: dict[str, dict[str, int | None]]
) -> str:
    header = f"{'setup':<18}" + "".join(f"{f'hit@{k}':>8}" for k in HIT_CUTOFFS) + f"{'MRR':>8}"
    lines = [f"Retrieval eval on {metrics[0].cases} cases (rank of the first relevant passage)", ""]
    lines.append(header)
    for row in metrics:
        hits = "".join(f"{row.hit_rate[k]:>8.0%}" for k in HIT_CUTOFFS)
        lines.append(f"{row.setup:<18}{hits}{row.mrr:>8.2f}")
    lines += ["", "Rank per case ('-' = not in the top 10):"]
    setups = [row.setup for row in metrics]
    lines.append(f"{'case':<38}" + "".join(f"{name:>17}" for name in setups))
    for case_id, by_setup in ranks.items():
        cells = "".join(f"{by_setup[name] or '-':>17}" for name in setups)
        lines.append(f"{case_id:<38}{cells}")
    return "\n".join(lines)


def run_retrieval(settings: Settings, cases: list[EvalCase]) -> int:
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model)
    with connect(DbRole.ADMIN, settings) as conn:
        metrics, ranks = run_retrieval_eval(
            conn, embedder, reranker, cases, settings.search_candidates
        )
    report = format_retrieval(metrics, ranks)
    path = output_dir(settings, COMMAND_RETRIEVAL)
    (path / "summary.txt").write_text(report + "\n")
    (path / "ranks.json").write_text(json.dumps(ranks, indent=2))
    print(f"{report}\n\nSaved to {path}")
    return 0


def run_answers(settings: Settings, cases: list[EvalCase], rerank: bool) -> int:
    agent = build_agent(settings, rerank=rerank)
    judge = ClaudeJudge(settings.eval_judge_model)
    pricing = Pricing.load(settings.config_dir)
    path = output_dir(settings, COMMAND_ANSWERS + ("-rerank" if rerank else ""))

    results: list[CaseResult] = []
    with (path / "results.jsonl").open("w") as output:
        for index, case in enumerate(cases, start=1):
            result = evaluate_case(case, agent, judge, pricing)
            results.append(result)
            output.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")
            output.flush()
            outcome = result.verdict or f"ERROR {result.error}"
            print(f"[{index}/{len(cases)}] {case.id:<36} {outcome}", file=sys.stderr)

    summary = format_summary(summarize(results), settings.agent_model, settings.eval_judge_model)
    (path / "summary.txt").write_text(summary + "\n")
    print(f"\n{summary}\n\nSaved to {path}")
    return 0


def run_judge_check(settings: Settings, cases: list[EvalCase]) -> int:
    """The judge must pass a perfect answer and fail three kinds of bad answer."""
    judge = ClaudeJudge(settings.eval_judge_model)
    answerable = next(case for case in cases if case.kind is CaseKind.NUMERIC)
    unanswerable = next(case for case in cases if not case.answerable)
    probes = [
        ("oracle: the expected facts", answerable, " ".join(answerable.expected_facts), True),
        ("empty answer", answerable, "", False),
        ("'I could not find it'", answerable, "I could not find this information.", False),
        (
            "confident answer to another question",
            answerable,
            "Tesla delivered 1.8 million vehicles and its revenue was $12.3 billion.",
            False,
        ),
        (
            "abstention on an unanswerable case",
            unanswerable,
            "That is not available in the SEC filings data I can access.",
            True,
        ),
        (
            "fabricated answer on an unanswerable case",
            unanswerable,
            "The figure was $637.9 billion.",
            False,
        ),
    ]
    failures = 0
    for name, case, answer, should_pass in probes:
        judgment, _ = judge.grade(case, answer)
        passed = judgment.verdict is Verdict.CORRECT
        ok = passed == should_pass
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:<42} verdict={judgment.verdict.value}")
    print(f"\n{len(probes) - failures}/{len(probes)} probes behaved as expected.")
    return EXIT_FAILED_CHECK if failures else 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=[COMMAND_RETRIEVAL, COMMAND_ANSWERS, COMMAND_JUDGE_CHECK]
    )
    parser.add_argument("--case", nargs="+", help="run only these case ids")
    parser.add_argument("--rerank", action="store_true", help="answers: turn the re-ranker on")
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_cases()
    if args.case:
        unknown = set(args.case) - {case.id for case in cases}
        if unknown:
            parser.error(f"unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in args.case]

    if args.command == COMMAND_RETRIEVAL:
        return run_retrieval(settings, cases)
    if args.command == COMMAND_JUDGE_CHECK:
        return run_judge_check(settings, cases)
    return run_answers(settings, cases, args.rerank)


if __name__ == "__main__":
    sys.exit(main())
