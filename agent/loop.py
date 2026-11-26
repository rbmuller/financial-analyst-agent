"""The agent loop, written by hand.

An "agent" is this loop and nothing more:

    1. send the conversation and the tool definitions to the model;
    2. if the model answers with tool calls, run them and append the results;
    3. repeat until the model answers with text instead of tool calls.

The model never executes anything. It only writes "call search_filings with
these arguments"; this code decides whether and how to run it. The conversation
is append-only: every model turn is stored exactly as returned, because the API
checks that earlier turns (including the model's reasoning blocks) come back
unmodified.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agent.llm import LLM, usage_of
from agent.pricing import Pricing, TokenUsage
from agent.sources import Source, SourceRegistry, resolve_citations, strip_citations
from agent.tools.base import Tool, ToolInputError, ToolOutput, to_api_definition
from agent.tracing import RunOrigin, RunStatus, Step, StepType, Tracer

STOP_TOOL_USE = "tool_use"
STOP_END_TURN = "end_turn"
STOP_PAUSE_TURN = "pause_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"

MESSAGE_REFUSED = "The model declined to answer this question."
MESSAGE_STEP_LIMIT = "I could not finish the analysis within the allowed number of steps."
MESSAGE_TRUNCATED = "The answer was cut off before it was complete."

type StepListener = Callable[[Step], None]
type Exchange = tuple[str, str]  # an earlier (question, answer) of the same conversation


@dataclass
class AgentResult:
    run_id: str
    status: RunStatus
    answer: str
    sources: list[Source] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: Decimal | None = None
    latency_ms: int = 0
    steps: int = 0


def _milliseconds_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _text_of(response: Any) -> str:
    return "\n".join(block.text for block in response.content if block.type == BLOCK_TEXT).strip()


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        system_prompt: str,
        tracer: Tracer,
        pricing: Pricing,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._tool_definitions = [to_api_definition(tool) for tool in tools]
        self._system_prompt = system_prompt
        self._tracer = tracer
        self._pricing = pricing
        self._max_steps = max_steps

    def run(
        self,
        question: str,
        origin: RunOrigin,
        on_step: StepListener | None = None,
        history: list[Exchange] | None = None,
    ) -> AgentResult:
        started = time.monotonic()
        run_id = self._tracer.start_run(question, self._llm.model, origin)
        result = AgentResult(run_id=run_id, status=RunStatus.RUNNING, answer="")
        try:
            self._loop(question, history or [], result, on_step)
        except Exception as error:
            result.status = RunStatus.ERROR
            self._finish(result, started, error=f"{type(error).__name__}: {error}")
            raise
        self._finish(result, started)
        return result

    def _finish(self, result: AgentResult, started: float, error: str | None = None) -> None:
        result.latency_ms = _milliseconds_since(started)
        self._tracer.finish_run(
            result.run_id,
            result.status,
            result.answer or None,
            error,
            result.usage,
            result.cost_usd,
            result.latency_ms,
        )

    def _record(self, result: AgentResult, step: Step, on_step: StepListener | None) -> None:
        self._tracer.record_step(result.run_id, result.steps, step)
        result.steps += 1
        if on_step:
            on_step(step)

    def _loop(
        self,
        question: str,
        history: list[Exchange],
        result: AgentResult,
        on_step: StepListener | None,
    ) -> None:
        sources = SourceRegistry()
        messages: list[dict[str, Any]] = []
        for earlier_question, earlier_answer in history:
            messages.append({"role": "user", "content": earlier_question})
            messages.append({"role": "assistant", "content": strip_citations(earlier_answer)})
        messages.append({"role": "user", "content": question})

        for _ in range(self._max_steps):
            response = self._call_model(messages, result, on_step)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == STOP_TOOL_USE:
                tool_results = [
                    self._run_tool(block, sources, result, on_step)
                    for block in response.content
                    if block.type == BLOCK_TOOL_USE
                ]
                # All results of one turn go back together, in a single user message.
                messages.append({"role": "user", "content": tool_results})
                continue
            if response.stop_reason == STOP_PAUSE_TURN:
                continue

            self._conclude(response, sources, result)
            return

        result.status = RunStatus.STEP_LIMIT
        result.answer = MESSAGE_STEP_LIMIT

    def _call_model(
        self, messages: list[dict[str, Any]], result: AgentResult, on_step: StepListener | None
    ) -> Any:
        started = time.monotonic()
        response = self._llm.complete(self._system_prompt, self._tool_definitions, messages)
        usage = usage_of(response)
        cost = self._pricing.cost_usd(response.model, usage)

        result.usage += usage
        if cost is not None:
            result.cost_usd = (result.cost_usd or Decimal(0)) + cost
        step = Step(
            type=StepType.LLM,
            latency_ms=_milliseconds_since(started),
            output=f"[{response.stop_reason}] {_text_of(response)}",
            usage=usage,
            cost_usd=cost,
        )
        self._record(result, step, on_step)
        return response

    def _run_tool(
        self,
        block: Any,
        sources: SourceRegistry,
        result: AgentResult,
        on_step: StepListener | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        tool = self._tools.get(block.name)
        try:
            if tool is None:
                raise ToolInputError(f"Unknown tool '{block.name}'.")
            output = tool.run(dict(block.input), sources)
        except ToolInputError as error:
            # The model gets the error back and can correct its call.
            output = ToolOutput(str(error), is_error=True)

        step = Step(
            type=StepType.TOOL,
            latency_ms=_milliseconds_since(started),
            tool_name=block.name,
            tool_input=dict(block.input),
            output=output.content,
            is_error=output.is_error,
        )
        self._record(result, step, on_step)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output.content,
            "is_error": output.is_error,
        }

    def _conclude(self, response: Any, sources: SourceRegistry, result: AgentResult) -> None:
        if response.stop_reason == STOP_REFUSAL:
            result.status, result.answer = RunStatus.REFUSED, MESSAGE_REFUSED
            return

        result.answer = _text_of(response)
        result.sources, result.unknown_citations = resolve_citations(result.answer, sources)
        if response.stop_reason == STOP_MAX_TOKENS:
            result.status = RunStatus.TRUNCATED
            result.answer = f"{result.answer}\n\n{MESSAGE_TRUNCATED}".strip()
        else:
            result.status = RunStatus.OK
