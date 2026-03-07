import json
import unittest
import uuid

from src.models.evaluation import EvaluationRun, PipelineExperiment
from src.services.optimization_service import OptimizationService


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
    def __init__(self, runs=None):
        self.runs = runs or []
        self.added = []
        self.flush_called = 0
        self.commit_called = 0

    async def execute(self, _query):
        return _FakeExecuteResult(self.runs)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flush_called += 1

    async def commit(self):
        self.commit_called += 1


class FakeEmbeddingService:
    def embed_batch(self, texts):
        mapping = {
            "timeout on legal doc query": [1.0, 0.0],
            "timeout while retrieving policy": [1.1, 0.0],
            "irrelevant response for pricing": [0.0, 1.0],
            "hallucinated citation answer": [0.0, 1.1],
        }
        return [mapping[text] for text in texts]


class _FakeClaudeMessages:
    async def create(self, **kwargs):
        user_content = kwargs["messages"][0]["content"]
        label = "Retrieval Timeout Errors" if "timeout" in user_content else "Relevance/Hallucination Issues"

        class _Block:
            text = json.dumps({"label": label})

        class _Response:
            content = [_Block()]

        return _Response()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeClaudeMessages()


class OptimizationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_evaluation_runs_summarizes_scores(self):
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()

        run1 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_config_id,
            status="completed",
            metrics={"score": 0.8},
        )
        run2 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_config_id,
            status="completed",
            metrics={"score": 0.6},
        )
        run3 = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_config_id,
            status="failed",
            metrics={"error": "boom"},
        )

        service = OptimizationService(db_session=FakeAsyncSession([run1, run2, run3]))
        summary = await service.analyze_evaluation_runs(dataset_id=dataset_id, pipeline_config_id=pipeline_config_id)

        self.assertEqual(summary.total_runs, 3)
        self.assertEqual(summary.completed_runs, 2)
        self.assertEqual(summary.failed_runs, 1)
        self.assertAlmostEqual(summary.average_score, 0.7)
        self.assertEqual(summary.best_run_id, run1.id)
        self.assertEqual(summary.best_score, 0.8)
        self.assertEqual(summary.failure_clusters, [])

    async def test_analyze_evaluation_runs_builds_failure_clusters(self):
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()
        runs = [
            EvaluationRun(
                id=uuid.uuid4(),
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config_id,
                status="failed",
                metrics={"question": "timeout on legal doc query"},
            ),
            EvaluationRun(
                id=uuid.uuid4(),
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config_id,
                status="failed",
                metrics={"question": "timeout while retrieving policy"},
            ),
            EvaluationRun(
                id=uuid.uuid4(),
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config_id,
                status="failed",
                metrics={"question": "irrelevant response for pricing"},
            ),
            EvaluationRun(
                id=uuid.uuid4(),
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config_id,
                status="failed",
                metrics={"question": "hallucinated citation answer"},
            ),
        ]

        service = OptimizationService(
            db_session=FakeAsyncSession(runs),
            embedding_service=FakeEmbeddingService(),
            anthropic_client=FakeAnthropicClient(),
        )

        summary = await service.analyze_evaluation_runs(dataset_id=dataset_id, pipeline_config_id=pipeline_config_id)

        self.assertEqual(len(summary.failure_clusters), 2)
        labels = {cluster.label for cluster in summary.failure_clusters}
        self.assertIn("Retrieval Timeout Errors", labels)
        self.assertIn("Relevance/Hallucination Issues", labels)
        self.assertEqual(sum(cluster.size for cluster in summary.failure_clusters), 4)

    def test_generate_pipeline_candidates_cartesian_product(self):
        candidates = OptimizationService.generate_pipeline_candidates(
            base_config={"temperature": 0.0},
            search_space={"k": [3, 5], "reranker": ["none", "cross-encoder"]},
        )

        self.assertEqual(len(candidates), 4)
        self.assertIn({"temperature": 0.0, "k": 3, "reranker": "none"}, candidates)
        self.assertIn({"temperature": 0.0, "k": 5, "reranker": "cross-encoder"}, candidates)

    async def test_schedule_evaluation_experiments_creates_runs_and_enqueues(self):
        enqueued = []
        session = FakeAsyncSession()
        service = OptimizationService(db_session=session, enqueue_evaluation_run=enqueued.append)

        runs = await service.schedule_evaluation_experiments(
            dataset_id=uuid.uuid4(),
            provider="openai",
            pipeline_candidates=[{"k": 4}, {"k": 8}],
        )

        self.assertEqual(len(runs), 2)
        self.assertEqual(session.flush_called, 2)
        self.assertEqual(session.commit_called, 1)
        self.assertEqual(len(enqueued), 2)

    def test_compare_metrics_and_select_best_pipeline(self):
        exp1 = PipelineExperiment(
            id=uuid.uuid4(),
            evaluation_run_id=uuid.uuid4(),
            pipeline_config_id=uuid.uuid4(),
            score=0.75,
            results={"judge_scores": {"faithfulness": 0.7}, "retrieval_metrics": {"precision_at_k": 0.8}},
        )
        exp2 = PipelineExperiment(
            id=uuid.uuid4(),
            evaluation_run_id=uuid.uuid4(),
            pipeline_config_id=uuid.uuid4(),
            score=0.65,
            results={"judge_scores": {"faithfulness": 0.6}, "retrieval_metrics": {"precision_at_k": 0.9}},
        )

        service = OptimizationService(db_session=FakeAsyncSession())
        ranked = service.compare_metrics(
            experiments=[exp1, exp2],
            metric_weights={"judge_scores.faithfulness": 0.7, "retrieval_metrics.precision_at_k": 0.3},
        )

        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0].score, ranked[1].score)

        best = service.select_best_pipeline(
            experiments=[exp1, exp2],
            metric_weights={"judge_scores.faithfulness": 0.7, "retrieval_metrics.precision_at_k": 0.3},
        )

        self.assertIsNotNone(best)
        self.assertEqual(best.pipeline_config_id, ranked[0].pipeline_config_id)


if __name__ == "__main__":
    unittest.main()
