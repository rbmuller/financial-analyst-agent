"""What a tool is, from the agent loop's point of view.

A tool has two faces. The model sees `name`, `description` and `input_schema`:
that text is all it knows about when and how to call the tool. The loop sees
`run`, a plain Python function that receives the arguments the model chose.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from agent.sources import SourceRegistry


class ToolInputError(ValueError):
    """The model called the tool with arguments it can fix and retry."""


@dataclass(frozen=True)
class ToolOutput:
    content: str  # what the model reads back (JSON text)
    is_error: bool = False


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def run(self, arguments: dict[str, Any], sources: SourceRegistry) -> ToolOutput: ...


def to_api_definition(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
