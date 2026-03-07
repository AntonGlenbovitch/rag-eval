import unittest
import uuid

from src.models.evaluation import EvaluationRun
from src.services.evaluation_service import EvaluationService
from src.services.retrieval_service import RetrievalResult


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeRetrievalService:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class FakeJudgeService:
    def __init__(self) -> None:
        self.calls = []

    async def evaluate(self, question: str, answer: str, context_chunks: list[str]):
        self.calls.append(
            {
                "question": question,
                "answer": answer,
                "context_chunks": context_chunks,
            }
        )
        return {
            "faithfulness": 0.8,
            "relevance": 0.7,
            "hallucination": 0.9,
            "confidence": 0.6,
        }


class FakeAsyncSession:
    def __init__(self, run: EvaluationRun | None) -> None:
        self.run = run
        self.added = []
        self.commit_called = False

    async def get(self, model, row_id):
        if model is not EvaluationRun:
            return None
        if self.run and self.run.id == row_id:
            return self.run
        return None

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commit_called = True


class EvaluationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_runs_pipeline_and_persists_results(self) -> None:
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()
        run_id = uuid.uuid4()

        retrieved_chunk_id = uuid.uuid4()
        retrieval_results = [
            RetrievalResult(
                id=retrieved_chunk_id,
                dataset_id=dataset_id,
                content="A retrieval chunk",
                similarity=0.94,
                metadata={"source": "doc"},
            )
        ]

        run = EvaluationRun(id=run_id, dataset_id=dataset_id, pipeline_config_id=pipeline_config_id, status="pending")
        session = FakeAsyncSession(run=run)
        embedding_service = FakeEmbeddingService()
        retrieval_service = FakeRetrievalService(results=retrieval_results)
        judge_service = FakeJudgeService()

        service = EvaluationService(
            db_session=session,
            embedding_service=embedding_service,
            retrieval_service=retrieval_service,
            judge_service=judge_service,
            retrieval_k=5,
            similarity_threshold=0.75,
        )

        result = await service.evaluate(
            evaluation_run_id=run_id,
            dataset_id=dataset_id,
            pipeline_config_id=pipeline_config_id,
            question="What is RAG?",
            answer="RAG combines retrieval and generation.",
            relevant_chunk_ids=[retrieved_chunk_id],
        )

        self.assertEqual(embedding_service.calls, ["What is RAG?"])
        self.assertEqual(retrieval_service.calls[0]["dataset_id"], dataset_id)
        self.assertEqual(retrieval_service.calls[0]["query_embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(retrieval_service.calls[0]["k"], 5)
        self.assertEqual(retrieval_service.calls[0]["similarity_threshold"], 0.75)

        self.assertEqual(judge_service.calls[0]["question"], "What is RAG?")
        self.assertEqual(judge_service.calls[0]["context_chunks"], ["A retrieval chunk"])

        self.assertEqual(result.retrieval_metrics["precision_at_k"], 1.0)
        self.assertEqual(result.retrieval_metrics["recall_at_k"], 1.0)
        self.assertEqual(result.retrieval_metrics["reciprocal_rank"], 1.0)

        self.assertEqual(run.status, "completed")
        self.assertIsNotNone(run.completed_at)
        self.assertIn("judge_scores", run.metrics)
        self.assertTrue(session.commit_called)
        self.assertEqual(len(session.added), 1)

    async def test_evaluate_raises_when_run_not_found(self) -> None:
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()
        run_id = uuid.uuid4()
        session = FakeAsyncSession(run=None)

        service = EvaluationService(
            db_session=session,
            embedding_service=FakeEmbeddingService(),
            retrieval_service=FakeRetrievalService(results=[]),
            judge_service=FakeJudgeService(),
        )

        with self.assertRaises(ValueError):
            await service.evaluate(
                evaluation_run_id=run_id,
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config_id,
                question="Q",
                answer="A",
                relevant_chunk_ids=[],
            )


if __name__ == "__main__":
    unittest.main()
