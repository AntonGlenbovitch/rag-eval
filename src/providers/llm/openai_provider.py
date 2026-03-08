from __future__ import annotations

from typing import TYPE_CHECKING

from src.providers.llm.base import LLMGeneration, LLMProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        client: "AsyncOpenAI",
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_response(self, prompt: str) -> LLMGeneration:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        text = message.content if message and message.content else ""

        usage: dict[str, int | None] = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

        return LLMGeneration(
            provider="openai",
            model=self._model,
            text=text,
            prompt=prompt,
            usage=usage,
            raw_response=response,
        )
