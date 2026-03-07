from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib import request

from src.providers.llm.base import LLMGeneration, LLMProvider


class LocalModelProvider(LLMProvider):
    """Provider for local Ollama-compatible models via HTTP API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def generate_response(self, prompt: str) -> LLMGeneration:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        response = await asyncio.to_thread(self._post_generate, payload)
        text = str(response.get("response", "")).strip()

        usage = {
            "prompt_tokens": response.get("prompt_eval_count"),
            "completion_tokens": response.get("eval_count"),
            "total_tokens": _total_tokens(
                response.get("prompt_eval_count"),
                response.get("eval_count"),
            ),
        }

        return LLMGeneration(
            provider="ollama",
            model=self._model,
            text=text,
            prompt=prompt,
            usage=usage,
            raw_response=response,
        )

    def _post_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/api/generate"
        raw_payload = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=raw_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=self._timeout_seconds) as response:
            body = response.read().decode("utf-8")

        return json.loads(body)


def _total_tokens(input_tokens: Any, output_tokens: Any) -> int | None:
    if input_tokens is None and output_tokens is None:
        return None
    return int(input_tokens or 0) + int(output_tokens or 0)
