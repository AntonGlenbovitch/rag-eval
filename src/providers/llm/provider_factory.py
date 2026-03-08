from __future__ import annotations

from typing import Any

from src.providers.llm.anthropic_provider import AnthropicProvider
from src.providers.llm.base import LLMProvider
from src.providers.llm.local_model_provider import LocalModelProvider
from src.providers.llm.openai_provider import OpenAIProvider


class ProviderFactory:
    @staticmethod
    def get_provider(provider: str, **kwargs: Any) -> LLMProvider:
        name = provider.strip().lower()

        if name == "openai":
            client = kwargs["client"]
            model = kwargs["model"]
            max_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.0)
            return OpenAIProvider(client=client, model=model, max_tokens=max_tokens, temperature=temperature)

        if name == "anthropic":
            client = kwargs["client"]
            model = kwargs["model"]
            max_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.0)
            return AnthropicProvider(client=client, model=model, max_tokens=max_tokens, temperature=temperature)

        if name in {"ollama", "local"}:
            model = kwargs["model"]
            base_url = kwargs.get("base_url", "http://localhost:11434")
            timeout_seconds = kwargs.get("timeout_seconds", 60)
            return LocalModelProvider(model=model, base_url=base_url, timeout_seconds=timeout_seconds)

        raise ValueError(f"Unsupported provider: {provider}")
