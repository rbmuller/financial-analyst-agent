"""Ask the agent a question from the terminal.

Usage:
    python -m agent.cli "Why did Nike's gross margin fall in the last 3 quarters?"
    python -m agent.cli "..." --rerank
"""

import argparse
import json
import sys

import anthropic

from agent.factory import build_agent
from agent.loop import AgentResult
from agent.tracing import RunOrigin, Step, StepType
from common.logs import configure_logging
from common.settings import get_settings

ORIGIN_CLI = "cli"
ARGUMENT_PREVIEW_CHARS = 160
EXIT_API_ERROR = 1


def print_step(step: Step) -> None:
    """Live trace of what the agent is doing."""
    if step.type is StepType.LLM:
        usage = step.usage
        tokens = f"{usage.total_input_tokens} in / {usage.output_tokens} out" if usage else ""
        print(f"  model  {step.latency_ms:>6} ms  {tokens}", file=sys.stderr)
        return
    arguments = json.dumps(step.tool_input, ensure_ascii=False)[:ARGUMENT_PREVIEW_CHARS]
    status = "ERROR " if step.is_error else ""
    print(
        f"  tool   {step.latency_ms:>6} ms  {status}{step.tool_name} {arguments}", file=sys.stderr
    )


def print_result(result: AgentResult) -> None:
    print(f"\n{result.answer}\n")
    if result.sources:
        print("Sources:")
        for source in result.sources:
            print(f"  [{source.id}] {source.label}")
            if source.url:
                print(f"        {source.url}")
            if source.detail:
                print(f"        SQL: {source.detail}")
    if result.unknown_citations:
        print(f"\nWARNING: cited ids that no tool returned: {', '.join(result.unknown_citations)}")
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "n/a (no price configured)"
    print(
        f"\nstatus={result.status.value}  steps={result.steps}  "
        f"tokens={result.usage.total_input_tokens} in / {result.usage.output_tokens} out  "
        f"cost={cost}  time={result.latency_ms / 1000:.1f}s  run_id={result.run_id}"
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--rerank", action="store_true", help="re-rank search results")
    args = parser.parse_args(argv)

    agent = build_agent(get_settings(), rerank=args.rerank)
    try:
        result = agent.run(args.question, RunOrigin(source=ORIGIN_CLI), on_step=print_step)
    except anthropic.AuthenticationError:
        print("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.", file=sys.stderr)
        return EXIT_API_ERROR
    except anthropic.RateLimitError:
        print("Rate limited by the Anthropic API. Try again in a minute.", file=sys.stderr)
        return EXIT_API_ERROR
    except anthropic.APIStatusError as error:
        print(f"Anthropic API error {error.status_code}: {error.message}", file=sys.stderr)
        return EXIT_API_ERROR
    except anthropic.APIConnectionError:
        print("Could not reach the Anthropic API. Check the network.", file=sys.stderr)
        return EXIT_API_ERROR

    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
