"""The agent loop with a scripted model: no network, no database."""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from agent.loop import MESSAGE_REFUSED, MESSAGE_STEP_LIMIT, Agent
from agent.pricing import ModelPrice, Pricing, TokenUsage
from agent.sources import SourceKind, SourceRegistry
from agent.tools.base import Tool, ToolInputError, ToolOutput
from agent.tracing import RunOrigin, RunStatus, Step, StepType

MODEL = "test-model"
ORIGIN = RunOrigin(source="test")
PRICING = Pricing(
    {MODEL: ModelPrice(Decimal(1), Decimal(1), Decimal(1))}, cache_write_multiplier=Decimal(1)
)


def text(content: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=content)


def tool_call(call_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def turn(*blocks: SimpleNamespace, stop_reason: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage, model=MODEL)


class ScriptedLLM:
    model = MODEL

    def __init__(self, *turns: SimpleNamespace) -> None:
        self._turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []

    def complete(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        self.requests.append(list(messages))
        return self._turns.pop(0) if len(self._turns) > 1 else self._turns[0]


class LookupTool:
    name = "lookup"
    description = "Looks something up."
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any], sources: SourceRegistry) -> ToolOutput:
        if arguments.get("fail"):
            raise ToolInputError("bad argument, try again")
        source_id = sources.add(SourceKind.FILING, str(arguments["key"]), "A filing")
        return ToolOutput(f"found {arguments['key']} as {source_id}")


class MemoryTracer:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.finished: dict[str, Any] = {}

    def start_run(self, question: str, model: str, origin: RunOrigin) -> str:
        return "run-1"

    def record_step(self, run_id: str, index: int, step: Step) -> None:
        assert index == len(self.steps)
        self.steps.append(step)

    def finish_run(self, run_id: str, status: RunStatus, *details: Any) -> None:
        self.finished = {"status": status, "details": details}


def make_agent(llm: ScriptedLLM, tracer: MemoryTracer, max_steps: int = 5) -> Agent:
    tools: list[Tool] = [LookupTool()]
    return Agent(llm, tools, "system prompt", tracer, PRICING, max_steps)


def test_runs_tools_until_the_model_answers_and_resolves_citations() -> None:
    llm = ScriptedLLM(
        turn(
            tool_call("call-1", "lookup", key="a"),
            tool_call("call-2", "lookup", key="b"),
            stop_reason="tool_use",
        ),
        turn(text("Supported claim [S2]. Invented claim [S9].")),
    )
    tracer = MemoryTracer()

    result = make_agent(llm, tracer).run("question", ORIGIN)

    assert result.status is RunStatus.OK
    assert [source.id for source in result.sources] == ["S2"]
    assert result.unknown_citations == ["S9"]
    assert [step.type for step in tracer.steps] == [
        StepType.LLM, StepType.TOOL, StepType.TOOL, StepType.LLM,
    ]  # fmt: skip
    assert result.steps == 4
    assert result.usage == TokenUsage(input_tokens=200, output_tokens=20)
    assert result.cost_usd == Decimal("0.00022")
    assert tracer.finished["status"] is RunStatus.OK


def test_tool_results_go_back_together_after_the_unmodified_model_turn() -> None:
    first_turn = turn(
        tool_call("call-1", "lookup", key="a"),
        tool_call("call-2", "lookup", key="b"),
        stop_reason="tool_use",
    )
    llm = ScriptedLLM(first_turn, turn(text("done")))

    make_agent(llm, MemoryTracer()).run("question", ORIGIN)

    question, assistant, results = llm.requests[1]
    assert question == {"role": "user", "content": "question"}
    assert assistant == {"role": "assistant", "content": first_turn.content}
    assert results["role"] == "user"
    assert [(r["tool_use_id"], r["content"], r["is_error"]) for r in results["content"]] == [
        ("call-1", "found a as S1", False),
        ("call-2", "found b as S2", False),
    ]


def test_tool_errors_are_returned_to_the_model_instead_of_crashing() -> None:
    llm = ScriptedLLM(
        turn(
            tool_call("call-1", "lookup", fail=True),
            tool_call("call-2", "no_such_tool"),
            stop_reason="tool_use",
        ),
        turn(text("I could not find it.")),
    )
    tracer = MemoryTracer()

    result = make_agent(llm, tracer).run("question", ORIGIN)

    results = llm.requests[1][2]["content"]
    assert [(r["is_error"], r["content"]) for r in results] == [
        (True, "bad argument, try again"),
        (True, "Unknown tool 'no_such_tool'."),
    ]
    assert [step.is_error for step in tracer.steps] == [False, True, True, False]
    assert result.status is RunStatus.OK


def test_step_limit_stops_a_model_that_never_answers() -> None:
    llm = ScriptedLLM(turn(tool_call("call-1", "lookup", key="a"), stop_reason="tool_use"))

    result = make_agent(llm, MemoryTracer(), max_steps=3).run("question", ORIGIN)

    assert result.status is RunStatus.STEP_LIMIT
    assert result.answer == MESSAGE_STEP_LIMIT
    assert len(llm.requests) == 3


def test_refusal_and_truncation_are_reported_as_such() -> None:
    refused = make_agent(ScriptedLLM(turn(stop_reason="refusal")), MemoryTracer())
    truncated = make_agent(
        ScriptedLLM(turn(text("Half an ans"), stop_reason="max_tokens")), MemoryTracer()
    )

    result = refused.run("q", ORIGIN)
    assert (result.status, result.answer) == (RunStatus.REFUSED, MESSAGE_REFUSED)
    result = truncated.run("q", ORIGIN)
    assert result.status is RunStatus.TRUNCATED
    assert result.answer.startswith("Half an ans")


def test_a_failed_model_call_is_traced_and_reraised() -> None:
    class BrokenLLM(ScriptedLLM):
        def complete(self, *args: Any) -> Any:
            raise RuntimeError("network down")

    tracer = MemoryTracer()

    with pytest.raises(RuntimeError, match="network down"):
        make_agent(BrokenLLM(turn()), tracer).run("question", ORIGIN)

    assert tracer.finished["status"] is RunStatus.ERROR
    assert "network down" in tracer.finished["details"][1]


def test_steps_are_reported_live_to_the_listener() -> None:
    llm = ScriptedLLM(
        turn(tool_call("call-1", "lookup", key="a"), stop_reason="tool_use"), turn(text("done"))
    )
    seen: list[str] = []

    make_agent(llm, MemoryTracer()).run(
        "question", ORIGIN, on_step=lambda step: seen.append(step.tool_name or step.type.value)
    )

    assert seen == ["llm", "lookup", "llm"]
