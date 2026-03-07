import unittest
import uuid
from types import SimpleNamespace

from src.models.evaluation import EvaluationRun, QAPair
from src.tasks.jobs import _run_evaluation_async


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _ExecuteResult:
    def __init__(self, qa_pair):
        self._qa_pair = qa_pair

    def scalars(self):
        return _ScalarResult(self._qa_pair)


class FakeSession:
    def __init__(self, run: EvaluationRun | None, qa_pair: QAPair | None):
        self._run = run
        self._qa_pair = qa_pair
        self.commits = 0

    async def get(self, model, row_id):
        if self._run and self._run.id == row_id:
            return self._run
        return None

    async def execute(self, _statement):
        return _ExecuteResult(self._qa_pair)

    async def commit(self):
        self.commits += 1


class FakeSessionFactory:
    def __init__(self, session: FakeSession):
        self._session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeEvaluationResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class FakeEvaluationService:
    def __init__(self, payload=None, exc: Exception | None = None):
        self.payload = payload or {"status": "ok"}
        self.exc = exc
        self.calls = []

    async def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return FakeEvaluationResult(self.payload)


class CeleryRunEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_evaluation_executes_service_and_returns_payload(self):
        run_id = uuid.uuid4()
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()
        relevant_chunk_id = uuid.uuid4()

        run = EvaluationRun(id=run_id, dataset_id=dataset_id, pipeline_config_id=pipeline_config_id, status="pending")
        qa_pair = QAPair(
            dataset_id=dataset_id,
            question="What is RAG?",
            answer="Retrieval augmented generation.",
            extra_metadata={"relevant_chunk_ids": [str(relevant_chunk_id)]},
        )

        session = FakeSession(run=run, qa_pair=qa_pair)
        session_factory = FakeSessionFactory(session)
        evaluation_service = FakeEvaluationService(payload={"score": 0.9})

        result = await _run_evaluation_async(
            evaluation_run_id=run_id,
            session_factory=session_factory,
            service_factory=lambda _: evaluation_service,
        )

        self.assertEqual(result, {"score": 0.9})
        self.assertEqual(run.status, "running")
        self.assertEqual(session.commits, 1)
        self.assertEqual(len(evaluation_service.calls), 1)
        self.assertEqual(evaluation_service.calls[0]["question"], "What is RAG?")
        self.assertEqual(evaluation_service.calls[0]["relevant_chunk_ids"], [relevant_chunk_id])

    async def test_run_evaluation_raises_when_run_not_found(self):
        run_id = uuid.uuid4()
        session = FakeSession(run=None, qa_pair=None)

        with self.assertRaises(ValueError):
            await _run_evaluation_async(
                evaluation_run_id=run_id,
                session_factory=FakeSessionFactory(session),
                service_factory=lambda _: FakeEvaluationService(),
            )

    async def test_run_evaluation_marks_run_failed_when_service_errors(self):
        run_id = uuid.uuid4()
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()

        run = EvaluationRun(id=run_id, dataset_id=dataset_id, pipeline_config_id=pipeline_config_id, status="pending")
        qa_pair = QAPair(dataset_id=dataset_id, question="Q", answer="A")

        session = FakeSession(run=run, qa_pair=qa_pair)
        session_factory = FakeSessionFactory(session)
        evaluation_service = FakeEvaluationService(exc=RuntimeError("judge unavailable"))

        with self.assertRaises(RuntimeError):
            await _run_evaluation_async(
                evaluation_run_id=run_id,
                session_factory=session_factory,
                service_factory=lambda _: evaluation_service,
            )

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.metrics, {"error": "judge unavailable"})
        self.assertIsNotNone(run.completed_at)
        self.assertEqual(session.commits, 2)

    async def test_run_evaluation_raises_when_no_qa_pair_found(self):
        run_id = uuid.uuid4()
        dataset_id = uuid.uuid4()
        pipeline_config_id = uuid.uuid4()

        run = EvaluationRun(id=run_id, dataset_id=dataset_id, pipeline_config_id=pipeline_config_id, status="pending")
        session = FakeSession(run=run, qa_pair=None)

        with self.assertRaises(ValueError):
            await _run_evaluation_async(
                evaluation_run_id=run_id,
                session_factory=FakeSessionFactory(session),
                service_factory=lambda _: SimpleNamespace(),
            )


if __name__ == "__main__":
    unittest.main()
