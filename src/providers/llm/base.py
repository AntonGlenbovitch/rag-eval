from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMGeneration:
    provider: str
    model: str
    text: str
    prompt: str
    usage: dict[str, int | None]
    raw_response: Any


class LLMProvider(ABC):
    """Base abstraction for async text generation providers."""

    @abstractmethod
    async def generate_response(self, prompt: str) -> LLMGeneration:
        """Generate text and return normalized metadata."""

    async def generate(self, prompt: str) -> str:
        """Generate text only (required interface)."""
        result = await self.generate_response(prompt)
        return result.text
