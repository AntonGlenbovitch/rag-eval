import unittest
import uuid

from src.models.evaluation import EvaluationRun, PipelineConfig
from src.models.model import Model
from src.models.model_ranking import ModelRanking
from src.services.model_ranking_service import ModelRankingService


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeAsyncSession:
    def __init__(self, rows=None, models=None):
        self.rows = rows or []
        self.models = {model.id: model for model in (models or [])}
        self.added = []
        self.commit_called = 0
        self.delete_called = 0

    async def execute(self, query):
        query_text = str(query)
        if "DELETE FROM model_rankings" in query_text:
            self.delete_called += 1
            return _FakeExecuteResult([])
        return _FakeExecuteResult(self.rows)

    async def get(self, _model_cls, model_id):
        return self.models.get(model_id)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commit_called += 1


class ModelRankingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_rank_models_computes_weighted_scores_and_persists_rankings(self):
        dataset_id = uuid.uuid4()
        model_1_id = uuid.uuid4()
        model_2_id = uuid.uuid4()

        model_1 = Model(id=model_1_id, name="gpt-4.1", provider="openai", context_window=128000, cost_per_1k_tokens=0.01)
        model_2 = Model(
            id=model_2_id,
            name="claude-3-5-sonnet",
            provider="anthropic",
            context_window=200000,
            cost_per_1k_tokens=0.015,
        )

        pipeline_1 = PipelineConfig(id=uuid.uuid4(), name="p1", provider="openai", config={"model_id": str(model_1_id)})
        pipeline_2 = PipelineConfig(id=uuid.uuid4(), name="p2", provider="anthropic", config={"model_id": str(model_2_id)})

        run_1 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_1.id,
            status="completed",
            metrics={"score": 0.8, "retrieval_metrics": {"precision_at_k": 0.7}},
        )
        run_2 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_1.id,
            status="completed",
            metrics={"score": 0.9, "retrieval_metrics": {"precision_at_k": 0.8}},
        )
        run_3 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_2.id,
            status="completed",
            metrics={"score": 0.75, "retrieval_metrics": {"precision_at_k": 0.6}},
        )

        session = FakeAsyncSession(rows=[(run_1, pipeline_1), (run_2, pipeline_1), (run_3, pipeline_2)], models=[model_1, model_2])
        service = ModelRankingService(
            session,
            metric_weights={"score": 0.7, "retrieval_metrics.precision_at_k": 0.3},
        )

        rankings = await service.rank_models(dataset_id)

        self.assertEqual(len(rankings), 2)
        self.assertEqual(rankings[0].model_id, model_1_id)
        self.assertGreater(rankings[0].weighted_score, rankings[1].weighted_score)
        self.assertEqual(rankings[0].rank, 1)
        self.assertEqual(rankings[1].rank, 2)

        saved_rankings = [item for item in session.added if isinstance(item, ModelRanking)]
        self.assertEqual(len(saved_rankings), 2)
        self.assertEqual(session.delete_called, 1)
        self.assertEqual(session.commit_called, 1)

    async def test_rank_models_skips_rows_without_valid_model_id(self):
        dataset_id = uuid.uuid4()
        pipeline = PipelineConfig(id=uuid.uuid4(), name="p", provider="openai", config={})
        run = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline.id,
            status="completed",
            metrics={"score": 0.5},
        )

        session = FakeAsyncSession(rows=[(run, pipeline)], models=[])
        service = ModelRankingService(session)

        rankings = await service.rank_models(dataset_id)

        self.assertEqual(rankings, [])
        self.assertEqual(session.delete_called, 1)
        self.assertEqual(session.commit_called, 1)


if __name__ == "__main__":
    unittest.main()
