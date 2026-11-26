"""Deterministic arithmetic, so the model never does it "in its head".

LLMs predict plausible digits; they do not compute. Anything numeric in an
answer should come out of this tool (or out of SQL), not out of the model.
"""

import json
from collections.abc import Callable
from typing import Any

from agent.sources import SourceRegistry
from agent.tools.base import ToolInputError, ToolOutput

PERCENT = 100.0
RESULT_DECIMALS = 4


def _require(values: list[float], count: int, operation: str) -> None:
    if len(values) != count:
        raise ToolInputError(f"'{operation}' needs exactly {count} values, got {len(values)}.")


def percent_change(values: list[float], periods: float | None) -> float:
    """[old, new] -> (new - old) / |old| * 100."""
    _require(values, 2, "percent_change")
    old, new = values
    if old == 0:
        raise ToolInputError("percent_change is undefined when the old value is 0.")
    return (new - old) / abs(old) * PERCENT


def cagr(values: list[float], periods: float | None) -> float:
    """[start, end] over `periods` years -> compound annual growth rate in %."""
    _require(values, 2, "cagr")
    start, end = values
    if periods is None or periods <= 0:
        raise ToolInputError("cagr needs 'periods' (number of years) greater than 0.")
    if start <= 0 or end <= 0:
        raise ToolInputError("cagr needs positive start and end values.")
    return float(((end / start) ** (1 / periods) - 1) * PERCENT)


def average(values: list[float], periods: float | None) -> float:
    if not values:
        raise ToolInputError("average needs at least one value.")
    return sum(values) / len(values)


def ratio_percent(values: list[float], periods: float | None) -> float:
    """[numerator, denominator] -> numerator / denominator * 100 (e.g. a margin)."""
    _require(values, 2, "ratio_percent")
    numerator, denominator = values
    if denominator == 0:
        raise ToolInputError("ratio_percent is undefined when the denominator is 0.")
    return numerator / denominator * PERCENT


def difference(values: list[float], periods: float | None) -> float:
    """[a, b] -> b - a (e.g. percentage points between two margins)."""
    _require(values, 2, "difference")
    return values[1] - values[0]


OPERATIONS: dict[str, Callable[[list[float], float | None], float]] = {
    "percent_change": percent_change,
    "cagr": cagr,
    "average": average,
    "ratio_percent": ratio_percent,
    "difference": difference,
}


class CalculatorTool:
    name = "calculator"
    description = (
        "Exact arithmetic on numbers you already retrieved. Call this for every calculation "
        "that appears in your answer instead of computing it yourself. Operations: "
        "percent_change [old, new] -> % change; "
        "cagr [start, end] with periods = number of years -> compound annual growth in %; "
        "average [v1, v2, ...] -> mean; "
        "ratio_percent [numerator, denominator] -> ratio in % (e.g. gross profit / revenue); "
        "difference [a, b] -> b - a (e.g. percentage points between two margins)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": sorted(OPERATIONS)},
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Input numbers, in the order the operation expects.",
            },
            "periods": {
                "type": "number",
                "description": "Number of years between start and end. Only for cagr.",
            },
        },
        "required": ["operation", "values"],
    }

    def run(self, arguments: dict[str, Any], sources: SourceRegistry) -> ToolOutput:
        operation = arguments.get("operation")
        values = arguments.get("values")
        if operation not in OPERATIONS:
            raise ToolInputError(f"Unknown operation. Use one of: {', '.join(sorted(OPERATIONS))}.")
        if not isinstance(values, list) or not all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in values
        ):
            raise ToolInputError("'values' must be a list of numbers.")

        result = OPERATIONS[operation]([float(value) for value in values], arguments.get("periods"))
        return ToolOutput(
            json.dumps({"operation": operation, "result": round(result, RESULT_DECIMALS)})
        )
