"""The single place that talks to the Anthropic API."""

from typing import Any, Protocol

import anthropic

from agent.pricing import TokenUsage

# If the model declines a request on safety grounds, the API retries it on a
# fallback model inside the same call instead of returning an empty answer.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODE = "default"


class LLM(Protocol):
    model: str

    def complete(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        """One model turn. Returns the SDK message (content, stop_reason, usage, model)."""
        ...


class ClaudeLLM:
    def __init__(self, model: str, effort: str, max_tokens: int) -> None:
        self.model = model
        self._effort = effort
        self._max_tokens = max_tokens
        # Credentials come from the environment (ANTHROPIC_API_KEY); never from code.
        self._client = anthropic.Anthropic()

    def complete(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        # The request is built from plain dicts (the loop stores turns exactly as the
        # API returned them), which the SDK's typed overloads cannot check.
        return self._client.beta.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self._effort},
            # Cache the growing prefix: each loop iteration resends the whole
            # conversation, and cached tokens cost a fraction of fresh ones.
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks=FALLBACK_MODE,
        )


def usage_of(response: Any) -> TokenUsage:
    usage = response.usage
    return TokenUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )
