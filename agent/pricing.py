"""Estimate the cost of an LLM call from its token usage."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml

PRICING_FILE = "pricing.yaml"
TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0  # not cached
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    output: Decimal
    cache_read: Decimal


class Pricing:
    def __init__(self, prices: dict[str, ModelPrice], cache_write_multiplier: Decimal) -> None:
        self._prices = prices
        self._cache_write_multiplier = cache_write_multiplier

    @classmethod
    def load(cls, config_dir: Path) -> "Pricing":
        payload = yaml.safe_load((config_dir / PRICING_FILE).read_text())
        prices = {
            model: ModelPrice(
                input=Decimal(str(price["input"])),
                output=Decimal(str(price["output"])),
                cache_read=Decimal(str(price["cache_read"])),
            )
            for model, price in payload["models"].items()
        }
        return cls(prices, Decimal(str(payload["cache_write_multiplier"])))

    def cost_usd(self, model: str, usage: TokenUsage) -> Decimal | None:
        """Cost in USD, or None for a model with no configured price (never a guess)."""
        price = self._prices.get(model)
        if price is None:
            return None
        total = (
            usage.input_tokens * price.input
            + usage.output_tokens * price.output
            + usage.cache_read_tokens * price.cache_read
            + usage.cache_write_tokens * price.input * self._cache_write_multiplier
        )
        return total / TOKENS_PER_PRICE_UNIT
