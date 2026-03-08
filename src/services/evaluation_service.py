from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.models.evaluation import EvaluationRun, PipelineExperiment
from src.models.model import Model
from src.models.routing_decision import RoutingDecision
from src.providers.llm.provider_factory import provider_factory
from src.services.claude_judge_service import ClaudeJudgeService
from src.services.embedding_service import EmbeddingService
from src.services.query_analyzer import QueryAnalyzer
from src.services.retrieval_service import RetrievalService
from src.services.routing_policy_service import RoutedPipeline, RoutingPolicyService


@dataclass(slots=True)
class EvaluationResult:
    question: str
    query_embedding: list[float]
    retrieved_chunk_ids: list[uuid.UUID]
    retrieval_metrics: dict[str, float]
    judge_scores: dict[str, float]
    score: float

    def to_dict(self) -> dict:
        return {
            "question": self.question,
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
        self._query_analyzer = QueryAnalyzer()
        self._routing_policy_service = RoutingPolicyService(db_session=db_session)

    @staticmethod
    def _compute_score(retrieval_metrics: dict[str, float], judge_scores: dict[str, float]) -> float:
        metric_values = [*retrieval_metrics.values(), *judge_scores.values()]
        if not metric_values:
            return 0.0
        return sum(metric_values) / len(metric_values)

    async def _route_pipeline(self, dataset_id: uuid.UUID, question: str) -> RoutedPipeline:
        features = self._query_analyzer.analyze_query(question)
        selected = await self._routing_policy_service.select_pipeline(dataset_id=dataset_id, features=features)

        self._db_session.add(
            RoutingDecision(
                dataset_id=dataset_id,
                query=question,
                query_features=asdict(features),
                model_id=selected.model_id,
                pipeline_config_id=selected.pipeline_config_id,
                score=selected.score,
            )
        )
        return selected

    async def _maybe_generate_answer(
        self,
        *,
        model_id: uuid.UUID,
        question: str,
        context_chunks: list[str],
        fallback_answer: str,
    ) -> str:
        model = await self._db_session.get(Model, model_id)
        if model is None:
            return fallback_answer

        try:
            provider = provider_factory.get_provider(
                model_id,
                model_resolver=lambda candidate: model if candidate == model.id else None,
                model=model.name,
            )
            prompt = (
                "Use the context to answer the question.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{' '.join(context_chunks)}"
            )
            return await provider.generate(prompt)
        except Exception:
            return fallback_answer

    async def evaluate(
        self,
        *,
        evaluation_run_id: uuid.UUID,
        dataset_id: uuid.UUID,
        pipeline_config_id: uuid.UUID,
        question: str,
        answer: str,
        relevant_chunk_ids: list[uuid.UUID] | None = None,
        routing_enabled: bool = False,
    ) -> EvaluationResult:
        selected_pipeline_config_id = pipeline_config_id
        selected_model_id: uuid.UUID | None = None

        if routing_enabled:
            selected = await self._route_pipeline(dataset_id=dataset_id, question=question)
            selected_pipeline_config_id = selected.pipeline_config_id
            selected_model_id = selected.model_id

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

        if routing_enabled and selected_model_id is not None:
            answer = await self._maybe_generate_answer(
                model_id=selected_model_id,
                question=question,
                context_chunks=context_chunks,
                fallback_answer=answer,
            )

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
            question=question,
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
            pipeline_config_id=selected_pipeline_config_id,
            score=score,
            results=evaluation_result.to_dict(),
        )
        self._db_session.add(experiment)

        await self._db_session.commit()
        return evaluation_result
