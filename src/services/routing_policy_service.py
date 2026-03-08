from __future__ import annotations

import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.evaluation import EvaluationRun, PipelineConfig
from src.models.model import Model
from src.models.model_ranking import ModelRanking
from src.services.query_analyzer import QueryFeatures


@dataclass(slots=True)
class RoutedPipeline:
    model_id: uuid.UUID
    pipeline_config_id: uuid.UUID
    score: float
    reason: str


class RoutingPolicyService:
    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    @staticmethod
    def _to_uuid(value: Any) -> uuid.UUID | None:
        if isinstance(value, uuid.UUID):
            return value
        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _metric_from_run(run: EvaluationRun | None, key: str, fallback: float = 0.0) -> float:
        if run is None or not isinstance(run.metrics, dict):
            return fallback
        if key in run.metrics and isinstance(run.metrics[key], (int, float)):
            return float(run.metrics[key])

        for nested in run.metrics.values():
            if isinstance(nested, dict) and key in nested and isinstance(nested[key], (int, float)):
                return float(nested[key])
        return fallback

    @staticmethod
    def _cost_efficiency(model: Model) -> float:
        if model.cost_per_1k_tokens <= 0:
            return 1.0
        return min(1.0, 1.0 / model.cost_per_1k_tokens)

    async def select_pipeline(self, dataset_id: uuid.UUID, features: QueryFeatures) -> RoutedPipeline:
        rankings_result = await self._db_session.execute(
            select(ModelRanking).where(ModelRanking.dataset_id == dataset_id).order_by(ModelRanking.rank.asc())
        )
        rankings = list(rankings_result.scalars().all())

        pipelines_result = await self._db_session.execute(select(PipelineConfig).order_by(PipelineConfig.created_at.desc()))
        pipelines = list(pipelines_result.scalars().all())

        if not pipelines:
            raise ValueError("No pipeline configurations found")

        best: RoutedPipeline | None = None
        reasons: list[str] = []

        for pipeline in pipelines:
            config = pipeline.config if isinstance(pipeline.config, dict) else {}
            model_id = self._to_uuid(config.get("model_id"))
            if model_id is None:
                continue

            model = await self._db_session.get(Model, model_id)
            if model is None:
                continue

            latest_run_result = await self._db_session.execute(
                select(EvaluationRun)
                .where(
                    EvaluationRun.dataset_id == dataset_id,
                    EvaluationRun.pipeline_config_id == pipeline.id,
                    EvaluationRun.status == "completed",
                )
                .order_by(EvaluationRun.completed_at.desc())
                .limit(1)
            )
            latest_run = latest_run_result.scalars().first()

            faithfulness = self._metric_from_run(latest_run, "faithfulness")
            relevance = self._metric_from_run(latest_run, "relevance")
            recall_at_k = self._metric_from_run(latest_run, "recall_at_k")
            mrr = self._metric_from_run(latest_run, "reciprocal_rank")
            cost_efficiency = self._cost_efficiency(model)

            ranking_bonus = 0.0
            if rankings:
                rank_lookup = {item.model_id: item for item in rankings}
                ranked = rank_lookup.get(model_id)
                if ranked is not None:
                    ranking_bonus = max(0.0, 1.0 - ((ranked.rank - 1) / max(1, len(rankings)))) * 0.1

            score = (
                0.40 * faithfulness
                + 0.25 * relevance
                + 0.15 * recall_at_k
                + 0.10 * mrr
                + 0.10 * cost_efficiency
                + ranking_bonus
            )

            if features.requires_reasoning:
                score += 0.10 * faithfulness
                reasons.append(f"boosted for reasoning ({model.name})")

            if features.requires_long_context:
                score += min(0.15, model.context_window / 1_000_000)
                reasons.append(f"boosted for long context ({model.name})")

            routed = RoutedPipeline(
                model_id=model_id,
                pipeline_config_id=pipeline.id,
                score=score,
                reason=f"Selected {model.name} based on evaluation metrics and ranking.",
            )
            if best is None or routed.score > best.score:
                best = routed

        if best is None:
            raise ValueError("No routable pipelines with model_id found")

        return best

    async def get_routing_stats(self, dataset_id: uuid.UUID) -> dict[str, Any]:
        from src.models.routing_decision import RoutingDecision

        result = await self._db_session.execute(
            select(RoutingDecision).where(RoutingDecision.dataset_id == dataset_id).order_by(RoutingDecision.created_at.desc())
        )
        decisions = list(result.scalars().all())

        if not decisions:
            return {
                "dataset_id": str(dataset_id),
                "number_of_routed_queries": 0,
                "average_model_score": 0.0,
                "most_selected_model": None,
            }

        scores = [item.score for item in decisions]
        by_model: dict[uuid.UUID, int] = {}
        for item in decisions:
            by_model[item.model_id] = by_model.get(item.model_id, 0) + 1

        most_selected_model = str(max(by_model.items(), key=lambda pair: pair[1])[0])

        return {
            "dataset_id": str(dataset_id),
            "number_of_routed_queries": len(decisions),
            "average_model_score": float(mean(scores)),
            "most_selected_model": most_selected_model,
        }
