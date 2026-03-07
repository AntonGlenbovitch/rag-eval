from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.models.evaluation import EvaluationRun, PipelineExperiment
from src.services.claude_judge_service import ClaudeJudgeService
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import RetrievalService


@dataclass(slots=True)
class EvaluationResult:
    query_embedding: list[float]
    retrieved_chunk_ids: list[uuid.UUID]
    retrieval_metrics: dict[str, float]
    judge_scores: dict[str, float]
    score: float

    def to_dict(self) -> dict:
        return {
            "query_embedding": self.query_embedding,
            "retrieved_chunk_ids": [str(chunk_id) for chunk_id in self.retrieved_chunk_ids],
            "retrieval_metrics": self.retrieval_metrics,
            "judge_scores": self.judge_scores,
            "score": self.score,
        }


class EvaluationService:
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        judge_service: ClaudeJudgeService,
        *,
        retrieval_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> None:
        self._db_session = db_session
        self._embedding_service = embedding_service
        self._retrieval_service = retrieval_service
        self._judge_service = judge_service
        self._retrieval_k = retrieval_k
        self._similarity_threshold = similarity_threshold

    @staticmethod
    def _compute_score(retrieval_metrics: dict[str, float], judge_scores: dict[str, float]) -> float:
        metric_values = [*retrieval_metrics.values(), *judge_scores.values()]
        if not metric_values:
            return 0.0
        return sum(metric_values) / len(metric_values)

    async def evaluate(
        self,
        *,
        evaluation_run_id: uuid.UUID,
        dataset_id: uuid.UUID,
        pipeline_config_id: uuid.UUID,
        question: str,
        answer: str,
        relevant_chunk_ids: list[uuid.UUID] | None = None,
    ) -> EvaluationResult:
        # Step 1: Embed query
        query_embedding = self._embedding_service.embed_text(question)

        # Step 2: Retrieve documents
        retrieval_results = await self._retrieval_service.search(
            dataset_id=dataset_id,
            query_embedding=query_embedding,
            k=self._retrieval_k,
            similarity_threshold=self._similarity_threshold,
        )

        retrieved_chunk_ids = [item.id for item in retrieval_results]
        context_chunks = [item.content for item in retrieval_results]

        # Step 3: Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids or [],
            k=self._retrieval_k,
        )

        # Step 4: Run judge
        judge_scores = await self._judge_service.evaluate(
            question=question,
            answer=answer,
            context_chunks=context_chunks,
        )

        score = self._compute_score(retrieval_metrics, judge_scores)
        evaluation_result = EvaluationResult(
            query_embedding=query_embedding,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieval_metrics=retrieval_metrics,
            judge_scores=judge_scores,
            score=score,
        )

        # Step 5: Store evaluation results
        run = await self._db_session.get(EvaluationRun, evaluation_run_id)
        if run is None:
            raise ValueError(f"EvaluationRun not found: {evaluation_run_id}")

        run.status = "completed"
        run.metrics = evaluation_result.to_dict()
        run.completed_at = datetime.now(timezone.utc)

        experiment = PipelineExperiment(
            evaluation_run_id=evaluation_run_id,
            pipeline_config_id=pipeline_config_id,
            score=score,
            results=evaluation_result.to_dict(),
        )
        self._db_session.add(experiment)

        await self._db_session.commit()
        return evaluation_result
