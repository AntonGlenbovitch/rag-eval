import unittest
import uuid

from src.models.evaluation import QAPair
from src.services.evaluation_service import EvaluationResult
from src.services.model_benchmark_service import ModelBenchmarkService


class FakeExecuteResult:
    def __init__(self, qa_pair: QAPair | None) -> None:
        self._qa_pair = qa_pair

    def scalars(self):
        return self

    def first(self):
        return self._qa_pair


class FakeAsyncSession:
    def __init__(self, qa_pair: QAPair | None) -> None:
        self.qa_pair = qa_pair
        self.added = []
        self.commit_count = 0

    async def execute(self, _query):
        return FakeExecuteResult(self.qa_pair)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commit_count += 1


class FakeEvaluationService:
    def __init__(self) -> None:
        self.calls = []

    async def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        model_score = float((kwargs["pipeline_config_id"].int % 100) / 100)
        return EvaluationResult(
            question=kwargs["question"],
            query_embedding=[0.1, 0.2],
            retrieved_chunk_ids=kwargs["relevant_chunk_ids"],
            retrieval_metrics={"precision_at_k": 0.5},
            judge_scores={"faithfulness": 0.8},
            score=model_score,
        )


class ModelBenchmarkServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_benchmark_dataset_runs_models_collects_metrics_and_stores_runs(self) -> None:
        dataset_id = uuid.uuid4()
        model_ids = [uuid.uuid4(), uuid.uuid4()]
        relevant_chunk_id = uuid.uuid4()

        qa_pair = QAPair(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            question="What is retrieval augmented generation?",
            answer="It combines retrieval and generation.",
            extra_metadata={"relevant_chunk_ids": [str(relevant_chunk_id)]},
        )

        session = FakeAsyncSession(qa_pair=qa_pair)
        evaluation_service = FakeEvaluationService()
        service = ModelBenchmarkService(db_session=session, evaluation_service=evaluation_service)

        result = await service.benchmark_dataset(dataset_id=dataset_id, model_ids=model_ids)

        self.assertEqual(result["dataset_id"], str(dataset_id))
        self.assertEqual(len(result["model_results"]), 2)
        self.assertEqual(len(evaluation_service.calls), 2)
        self.assertEqual(session.commit_count, 2)
        self.assertEqual(len(session.added), 2)

        for index, call in enumerate(evaluation_service.calls):
            self.assertEqual(call["dataset_id"], dataset_id)
            self.assertEqual(call["pipeline_config_id"], model_ids[index])
            self.assertEqual(call["question"], qa_pair.question)
            self.assertEqual(call["answer"], qa_pair.answer)
            self.assertEqual(call["relevant_chunk_ids"], [relevant_chunk_id])

    async def test_benchmark_dataset_raises_when_qa_pair_is_missing(self) -> None:
        session = FakeAsyncSession(qa_pair=None)
        evaluation_service = FakeEvaluationService()
        service = ModelBenchmarkService(db_session=session, evaluation_service=evaluation_service)

        with self.assertRaises(ValueError):
            await service.benchmark_dataset(dataset_id=uuid.uuid4(), model_ids=[uuid.uuid4()])


if __name__ == "__main__":
    unittest.main()
