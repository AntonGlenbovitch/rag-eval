import unittest
import uuid

from src.models.evaluation import EvaluationRun, PipelineConfig
from src.models.model import Model
from src.models.model_ranking import ModelRanking
from src.services.evaluation_service import EvaluationService
from src.services.query_analyzer import QueryFeatures
from src.services.retrieval_service import RetrievalResult
from src.services.routing_policy_service import RoutedPipeline, RoutingPolicyService


class _ScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def first(self):
        return self._items[0] if self._items else None


class _ExecResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _ScalarResult(self._items)


class FakeRoutingSession:
    def __init__(self, execute_results, models):
        self._execute_results = list(execute_results)
        self._models = models

    async def execute(self, _query):
        return _ExecResult(self._execute_results.pop(0))

    async def get(self, model_cls, model_id):
        return self._models.get(model_id)


class RoutingPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_factual_query_selects_cheaper_model(self):
        dataset_id = uuid.uuid4()
        cheap_model_id = uuid.uuid4()
        expensive_model_id = uuid.uuid4()
        cheap_pipeline_id = uuid.uuid4()
        expensive_pipeline_id = uuid.uuid4()

        cheap_model = Model(id=cheap_model_id, name="cheap", provider="openai", context_window=8000, cost_per_1k_tokens=0.2)
        expensive_model = Model(
            id=expensive_model_id,
            name="expensive",
            provider="openai",
            context_window=8000,
            cost_per_1k_tokens=20.0,
        )

        rankings = [
            ModelRanking(dataset_id=dataset_id, model_id=cheap_model_id, weighted_score=0.8, rank=1),
            ModelRanking(dataset_id=dataset_id, model_id=expensive_model_id, weighted_score=0.8, rank=2),
        ]
        pipelines = [
            PipelineConfig(id=cheap_pipeline_id, name="cheap-p", provider="openai", config={"model_id": str(cheap_model_id)}),
            PipelineConfig(
                id=expensive_pipeline_id,
                name="exp-p",
                provider="openai",
                config={"model_id": str(expensive_model_id)},
            ),
        ]
        # latest run per pipeline
        runs = [
            [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=cheap_pipeline_id, status="completed", metrics={"faithfulness": 0.7, "relevance": 0.7, "recall_at_k": 0.7, "reciprocal_rank": 0.7})],
            [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=expensive_pipeline_id, status="completed", metrics={"faithfulness": 0.7, "relevance": 0.7, "recall_at_k": 0.7, "reciprocal_rank": 0.7})],
        ]

        session = FakeRoutingSession([rankings, pipelines, runs[0], runs[1]], {cheap_model_id: cheap_model, expensive_model_id: expensive_model})
        selected = await RoutingPolicyService(db_session=session).select_pipeline(
            dataset_id=dataset_id,
            features=QueryFeatures(10, "factual", False, False, "low"),
        )

        self.assertEqual(selected.model_id, cheap_model_id)

    async def test_reasoning_query_selects_higher_reasoning_model(self):
        dataset_id = uuid.uuid4()
        low_id, high_id = uuid.uuid4(), uuid.uuid4()
        low_pipe, high_pipe = uuid.uuid4(), uuid.uuid4()
        models = {
            low_id: Model(id=low_id, name="low", provider="openai", context_window=8000, cost_per_1k_tokens=1.0),
            high_id: Model(id=high_id, name="high", provider="openai", context_window=8000, cost_per_1k_tokens=1.0),
        }
        rankings = [ModelRanking(dataset_id=dataset_id, model_id=low_id, weighted_score=0.7, rank=2), ModelRanking(dataset_id=dataset_id, model_id=high_id, weighted_score=0.8, rank=1)]
        pipelines = [
            PipelineConfig(id=low_pipe, name="low-p", provider="openai", config={"model_id": str(low_id)}),
            PipelineConfig(id=high_pipe, name="high-p", provider="openai", config={"model_id": str(high_id)}),
        ]
        session = FakeRoutingSession(
            [
                rankings,
                pipelines,
                [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=low_pipe, status="completed", metrics={"faithfulness": 0.3, "relevance": 0.8, "recall_at_k": 0.8, "reciprocal_rank": 0.8})],
                [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=high_pipe, status="completed", metrics={"faithfulness": 0.95, "relevance": 0.8, "recall_at_k": 0.8, "reciprocal_rank": 0.8})],
            ],
            models,
        )
        selected = await RoutingPolicyService(db_session=session).select_pipeline(
            dataset_id=dataset_id,
            features=QueryFeatures(200, "analytical", True, False, "high"),
        )
        self.assertEqual(selected.model_id, high_id)

    async def test_long_context_query_selects_large_context_model(self):
        dataset_id = uuid.uuid4()
        small_id, large_id = uuid.uuid4(), uuid.uuid4()
        small_pipe, large_pipe = uuid.uuid4(), uuid.uuid4()
        models = {
            small_id: Model(id=small_id, name="small", provider="openai", context_window=8000, cost_per_1k_tokens=1.0),
            large_id: Model(id=large_id, name="large", provider="openai", context_window=200000, cost_per_1k_tokens=1.0),
        }
        rankings = [ModelRanking(dataset_id=dataset_id, model_id=small_id, weighted_score=0.8, rank=1), ModelRanking(dataset_id=dataset_id, model_id=large_id, weighted_score=0.8, rank=2)]
        pipelines = [
            PipelineConfig(id=small_pipe, name="small-p", provider="openai", config={"model_id": str(small_id)}),
            PipelineConfig(id=large_pipe, name="large-p", provider="openai", config={"model_id": str(large_id)}),
        ]
        same_metrics = {"faithfulness": 0.7, "relevance": 0.7, "recall_at_k": 0.7, "reciprocal_rank": 0.7}
        session = FakeRoutingSession([rankings, pipelines, [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=small_pipe, status="completed", metrics=same_metrics)], [EvaluationRun(dataset_id=dataset_id, pipeline_config_id=large_pipe, status="completed", metrics=same_metrics)]], models)
        selected = await RoutingPolicyService(db_session=session).select_pipeline(
            dataset_id=dataset_id,
            features=QueryFeatures(220, "analytical", False, True, "medium"),
        )
        self.assertEqual(selected.model_id, large_id)


class FakeEmbeddingService:
    def embed_text(self, _text):
        return [0.1]


class FakeRetrievalService:
    async def search(self, **_kwargs):
        return [RetrievalResult(id=uuid.uuid4(), dataset_id=uuid.uuid4(), content="ctx", similarity=1.0, metadata={})]


class FakeJudgeService:
    async def evaluate(self, question: str, answer: str, context_chunks: list[str]):
        return {"faithfulness": 1.0, "relevance": 1.0, "hallucination": 1.0, "confidence": 1.0}


class FakeEvalSession:
    def __init__(self, run):
        self.run = run
        self.added = []

    async def get(self, model_cls, row_id):
        if model_cls is EvaluationRun and row_id == self.run.id:
            return self.run
        return None

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        return None


class RoutingDisabledTests(unittest.IsolatedAsyncioTestCase):
    async def test_routing_disabled_uses_default_pipeline(self):
        run_id = uuid.uuid4()
        default_pipeline_id = uuid.uuid4()
        run = EvaluationRun(id=run_id, dataset_id=uuid.uuid4(), pipeline_config_id=default_pipeline_id, status="pending")
        session = FakeEvalSession(run)

        service = EvaluationService(session, FakeEmbeddingService(), FakeRetrievalService(), FakeJudgeService())
        service._route_pipeline = lambda **_kwargs: RoutedPipeline(uuid.uuid4(), uuid.uuid4(), 0.0, "mock")  # type: ignore[method-assign]

        await service.evaluate(
            evaluation_run_id=run_id,
            dataset_id=run.dataset_id,
            pipeline_config_id=default_pipeline_id,
            question="What is RAG?",
            answer="retrieval augmented generation",
            routing_enabled=False,
        )

        experiment = session.added[0]
        self.assertEqual(experiment.pipeline_config_id, default_pipeline_id)


if __name__ == "__main__":
    unittest.main()
