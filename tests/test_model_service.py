import unittest
import uuid
from datetime import datetime, timezone

from src.models.model import Model
from src.services.model_service import ModelService


class _FakeScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _FakeExecuteResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _FakeScalarResult(self._values)


class FakeAsyncSession:
    def __init__(self) -> None:
        self.rows: list[Model] = []
        self.commit_called = 0
        self.refresh_called = 0

    def add(self, obj: Model) -> None:
        if obj.id is None:
            obj.id = uuid.uuid4()
        if obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc)
        self.rows.append(obj)

    async def commit(self) -> None:
        self.commit_called += 1

    async def refresh(self, _obj: Model) -> None:
        self.refresh_called += 1

    async def get(self, _model_cls, model_id: uuid.UUID):
        for row in self.rows:
            if row.id == model_id:
                return row
        return None

    async def execute(self, query):
        params = query.compile().params
        provider = params.get("provider_1")

        values = [row for row in self.rows if provider is None or row.provider == provider]
        values.sort(key=lambda row: row.created_at, reverse=True)
        return _FakeExecuteResult(values)

    async def delete(self, obj: Model) -> None:
        self.rows = [row for row in self.rows if row.id != obj.id]


class ModelServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_register_and_get_model(self) -> None:
        session = FakeAsyncSession()
        service = ModelService(session)

        created = await service.register_model(
            name="gpt-4.1",
            provider="openai",
            context_window=128000,
            cost_per_1k_tokens=0.01,
        )

        fetched = await service.get_model(created.id)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "gpt-4.1")
        self.assertEqual(session.commit_called, 1)
        self.assertEqual(session.refresh_called, 1)

    async def test_list_models_and_filter_by_provider(self) -> None:
        session = FakeAsyncSession()
        service = ModelService(session)

        older = Model(
            id=uuid.uuid4(),
            name="claude-3-haiku",
            provider="anthropic",
            context_window=200000,
            cost_per_1k_tokens=0.003,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = Model(
            id=uuid.uuid4(),
            name="gpt-4.1-mini",
            provider="openai",
            context_window=128000,
            cost_per_1k_tokens=0.002,
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        session.rows.extend([older, newer])

        all_models = await service.list_models()
        openai_models = await service.list_models(provider="openai")

        self.assertEqual([model.name for model in all_models], ["gpt-4.1-mini", "claude-3-haiku"])
        self.assertEqual([model.name for model in openai_models], ["gpt-4.1-mini"])

    async def test_delete_model(self) -> None:
        session = FakeAsyncSession()
        service = ModelService(session)

        item = Model(
            id=uuid.uuid4(),
            name="gpt-4o",
            provider="openai",
            context_window=128000,
            cost_per_1k_tokens=0.005,
            created_at=datetime.now(timezone.utc),
        )
        session.rows.append(item)

        deleted = await service.delete_model(item.id)
        missing_delete = await service.delete_model(uuid.uuid4())

        self.assertTrue(deleted)
        self.assertFalse(missing_delete)
        self.assertEqual(session.commit_called, 1)


if __name__ == "__main__":
    unittest.main()
