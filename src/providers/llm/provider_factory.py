from __future__ import annotations

import uuid
from typing import Any, Callable

from src.models.model import Model
from src.providers.llm.anthropic_provider import AnthropicProvider
from src.providers.llm.base import LLMProvider
from src.providers.llm.local_model_provider import LocalModelProvider
from src.providers.llm.openai_provider import OpenAIProvider


class ProviderFactory:
    @staticmethod
    def _build_provider(provider: str, **kwargs: Any) -> LLMProvider:
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

    @staticmethod
    def get_provider(model_id_or_provider: uuid.UUID | str, **kwargs: Any) -> LLMProvider:
        if isinstance(model_id_or_provider, uuid.UUID):
            model_resolver: Callable[[uuid.UUID], Model | None] = kwargs["model_resolver"]
            model_record = model_resolver(model_id_or_provider)
            if model_record is None:
                raise ValueError(f"Model not found: {model_id_or_provider}")
            kwargs.setdefault("model", model_record.name)
            return ProviderFactory._build_provider(model_record.provider, **kwargs)

        return ProviderFactory._build_provider(str(model_id_or_provider), **kwargs)


provider_factory = ProviderFactory()
