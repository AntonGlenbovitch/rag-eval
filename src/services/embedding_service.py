import hashlib
import json
from typing import Any

from redis import Redis


class EmbeddingService:
    CACHE_TTL_SECONDS = 24 * 60 * 60

    def __init__(self, openai_client: Any, redis_client: Redis, model: str) -> None:
        self._openai_client = openai_client
        self._redis_client = redis_client
        self._model = model

    @staticmethod
    def _cache_key(model: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{model}:{digest}"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        ordered_embeddings: list[list[float] | None] = [None] * len(texts)
        misses_by_text: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            cache_key = self._cache_key(self._model, text)
            cached_value = self._redis_client.get(cache_key)

            if cached_value is None:
                misses_by_text.setdefault(text, []).append(index)
                continue

            if isinstance(cached_value, bytes):
                decoded = cached_value.decode("utf-8")
            elif isinstance(cached_value, str):
                decoded = cached_value
            else:
                misses_by_text.setdefault(text, []).append(index)
                continue

            ordered_embeddings[index] = json.loads(decoded)

        missing_texts = list(misses_by_text.keys())
        if missing_texts:
            response = self._openai_client.embeddings.create(model=self._model, input=missing_texts)
            for row_index, item in enumerate(response.data):
                text = missing_texts[row_index]
                embedding = item.embedding
                cache_key = self._cache_key(self._model, text)
                self._redis_client.setex(cache_key, self.CACHE_TTL_SECONDS, json.dumps(embedding))

                for original_index in misses_by_text[text]:
                    ordered_embeddings[original_index] = embedding

        return [embedding for embedding in ordered_embeddings if embedding is not None]
