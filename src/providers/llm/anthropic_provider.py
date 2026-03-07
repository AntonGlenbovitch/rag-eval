from __future__ import annotations

from typing import TYPE_CHECKING

from src.providers.llm.base import LLMGeneration, LLMProvider

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        client: "AsyncAnthropic",
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_response(self, prompt: str) -> LLMGeneration:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        text = "\n".join(text_blocks).strip()

        usage = {
            "prompt_tokens": getattr(response.usage, "input_tokens", None),
            "completion_tokens": getattr(response.usage, "output_tokens", None),
            "total_tokens": (
                (getattr(response.usage, "input_tokens", 0) or 0)
                + (getattr(response.usage, "output_tokens", 0) or 0)
            )
            if getattr(response, "usage", None)
            else None,
        }

        return LLMGeneration(
            provider="anthropic",
            model=self._model,
            text=text,
            prompt=prompt,
            usage=usage,
            raw_response=response,
        )
