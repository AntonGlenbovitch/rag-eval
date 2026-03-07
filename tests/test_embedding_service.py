import hashlib
import json
import unittest

from src.services.embedding_service import EmbeddingService


class FakeRedis:
    def __init__(self) -> None:
        self.store = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls = []

    def create(self, model: str, input: list[str]):
        self.calls.append({"model": model, "input": input})

        class Item:
            def __init__(self, embedding):
                self.embedding = embedding

        class Response:
            def __init__(self, data):
                self.data = data

        data = [Item([float(len(text))]) for text in input]
        return Response(data)


class FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis_client = FakeRedis()
        self.openai_client = FakeOpenAI()
        self.service = EmbeddingService(
            openai_client=self.openai_client,
            redis_client=self.redis_client,
            model="text-embedding-3-small",
        )

    def test_cache_key_format(self) -> None:
        text = "hello"
        expected = f"emb:text-embedding-3-small:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        self.assertEqual(self.service._cache_key("text-embedding-3-small", text), expected)

    def test_embed_text_uses_cache_after_first_call(self) -> None:
        first = self.service.embed_text("cached")
        second = self.service.embed_text("cached")

        self.assertEqual(first, [6.0])
        self.assertEqual(second, [6.0])
        self.assertEqual(len(self.openai_client.embeddings.calls), 1)

    def test_embed_batch_fetches_only_cache_misses(self) -> None:
        cached_text = "cached"
        key = self.service._cache_key("text-embedding-3-small", cached_text)
        self.redis_client.store[key] = json.dumps([123.0])

        result = self.service.embed_batch([cached_text, "new", cached_text])

        self.assertEqual(result, [[123.0], [3.0], [123.0]])
        self.assertEqual(len(self.openai_client.embeddings.calls), 1)
        self.assertEqual(self.openai_client.embeddings.calls[0]["input"], ["new"])


if __name__ == "__main__":
    unittest.main()
