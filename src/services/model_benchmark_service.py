from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.evaluation import EvaluationRun, QAPair
from src.services.evaluation_service import EvaluationService


@dataclass(slots=True)
class ModelBenchmarkResult:
    model_id: uuid.UUID
    evaluation_run_id: uuid.UUID
    score: float
    retrieval_metrics: dict[str, float]
    judge_scores: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "model_id": str(self.model_id),
            "evaluation_run_id": str(self.evaluation_run_id),
            "score": self.score,
            "retrieval_metrics": self.retrieval_metrics,
            "judge_scores": self.judge_scores,
        }


class ModelBenchmarkService:
    def __init__(self, db_session: AsyncSession, evaluation_service: EvaluationService) -> None:
        self._db_session = db_session
        self._evaluation_service = evaluation_service

    @staticmethod
    def _extract_relevant_chunk_ids(qa_pair: QAPair) -> list[uuid.UUID]:
        metadata = qa_pair.extra_metadata or {}
        raw_ids = metadata.get("relevant_chunk_ids", [])

        relevant_chunk_ids: list[uuid.UUID] = []
        for raw_id in raw_ids:
            try:
                relevant_chunk_ids.append(uuid.UUID(str(raw_id)))
            except (TypeError, ValueError):
                continue

        return relevant_chunk_ids

    async def benchmark_dataset(self, dataset_id: uuid.UUID, model_ids: list[uuid.UUID]) -> dict:
        qa_result = await self._db_session.execute(
            select(QAPair).where(QAPair.dataset_id == dataset_id).order_by(QAPair.created_at.asc()).limit(1)
        )
        qa_pair = qa_result.scalars().first()
        if qa_pair is None:
            raise ValueError(f"No QA pair found for dataset: {dataset_id}")

        model_results: list[ModelBenchmarkResult] = []

        for model_id in model_ids:
            evaluation_run = EvaluationRun(dataset_id=dataset_id, pipeline_config_id=model_id, status="pending")
            self._db_session.add(evaluation_run)
            await self._db_session.commit()

            evaluation_result = await self._evaluation_service.evaluate(
                evaluation_run_id=evaluation_run.id,
                dataset_id=dataset_id,
                pipeline_config_id=model_id,
                question=qa_pair.question,
                answer=qa_pair.answer,
                relevant_chunk_ids=self._extract_relevant_chunk_ids(qa_pair),
            )

            model_results.append(
                ModelBenchmarkResult(
                    model_id=model_id,
                    evaluation_run_id=evaluation_run.id,
                    score=evaluation_result.score,
                    retrieval_metrics=evaluation_result.retrieval_metrics,
                    judge_scores=evaluation_result.judge_scores,
                )
            )

        average_score = sum(result.score for result in model_results) / len(model_results) if model_results else 0.0

        return {
            "dataset_id": str(dataset_id),
            "average_score": average_score,
            "model_results": [result.to_dict() for result in model_results],
        }
