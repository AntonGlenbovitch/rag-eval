import unittest
import uuid

from src.services.model_selector_service import ModelSelectorService


class _FakeExecuteResult:
    def __init__(self, model_id):
        self._model_id = model_id

    def scalar_one_or_none(self):
        return self._model_id


class FakeAsyncSession:
    def __init__(self, model_id=None):
        self.model_id = model_id
        self.last_query = None

    async def execute(self, query):
        self.last_query = query
        return _FakeExecuteResult(self.model_id)


class ModelSelectorServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_select_best_model_returns_highest_scoring_model_id(self):
        dataset_id = uuid.uuid4()
        expected_model_id = uuid.uuid4()
        session = FakeAsyncSession(model_id=expected_model_id)
        service = ModelSelectorService(session)

        model_id = await service.select_best_model(dataset_id)

        self.assertEqual(model_id, expected_model_id)
        self.assertIsNotNone(session.last_query)

    async def test_select_best_model_returns_none_when_no_rankings(self):
        dataset_id = uuid.uuid4()
        session = FakeAsyncSession(model_id=None)
        service = ModelSelectorService(session)

        model_id = await service.select_best_model(dataset_id)

        self.assertIsNone(model_id)


if __name__ == "__main__":
    unittest.main()
