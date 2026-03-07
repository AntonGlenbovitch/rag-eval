import unittest
import uuid

from src.services.retrieval_service import RetrievalService


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeAsyncSession:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []
        self.last_query = None
        self.last_params = None

    async def execute(self, query, params):
        self.last_query = str(query)
        self.last_params = params
        return FakeResult(self.rows)


class RetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_applies_dataset_filter_threshold_and_ordering(self) -> None:
        dataset_id = uuid.uuid4()
        row = {
            "id": uuid.uuid4(),
            "dataset_id": dataset_id,
            "content": "chunk content",
            "metadata": {"source": "doc1"},
            "similarity": 0.91,
        }
        fake_session = FakeAsyncSession(rows=[row])
        service = RetrievalService(fake_session)

        results = await service.search(
            dataset_id=dataset_id,
            query_embedding=[0.1, 0.2, 0.3],
            k=5,
            similarity_threshold=0.8,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "chunk content")
        self.assertEqual(results[0].similarity, 0.91)

        self.assertIn("WHERE dataset_id = :dataset_id", fake_session.last_query)
        self.assertIn("ORDER BY embedding <=> CAST(:query_vector AS vector)", fake_session.last_query)
        self.assertIn("LIMIT :k", fake_session.last_query)

        self.assertEqual(fake_session.last_params["dataset_id"], dataset_id)
        self.assertEqual(fake_session.last_params["query_vector"], "[0.1,0.2,0.3]")
        self.assertEqual(fake_session.last_params["k"], 5)
        self.assertEqual(fake_session.last_params["similarity_threshold"], 0.8)

    async def test_search_returns_empty_for_invalid_input(self) -> None:
        fake_session = FakeAsyncSession(rows=[])
        service = RetrievalService(fake_session)

        empty_embedding = await service.search(uuid.uuid4(), [], 5, 0.5)
        zero_k = await service.search(uuid.uuid4(), [0.1], 0, 0.5)

        self.assertEqual(empty_embedding, [])
        self.assertEqual(zero_k, [])
        self.assertIsNone(fake_session.last_query)


if __name__ == "__main__":
    unittest.main()
