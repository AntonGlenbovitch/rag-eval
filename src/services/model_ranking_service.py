from __future__ import annotations

import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.evaluation import EvaluationRun, PipelineConfig
from src.models.model import Model
from src.models.model_ranking import ModelRanking


@dataclass(slots=True)
class RankedModel:
    model_id: uuid.UUID
    model_name: str
    provider: str
    weighted_score: float
    rank: int


class ModelRankingService:
    def __init__(
        self,
        db_session: AsyncSession,
        *,
        metric_weights: dict[str, float] | None = None,
    ) -> None:
        self._db_session = db_session
        self._metric_weights = metric_weights or {"score": 1.0}

    @staticmethod
    def _flatten_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
        if not metrics:
            return {}

        flattened: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flattened[key] = float(value)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        flattened[f"{key}.{nested_key}"] = float(nested_value)

        return flattened

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

    def _compute_weighted_score(self, flattened_metrics: dict[str, float]) -> float:
        weighted_values = [
            flattened_metrics[metric] * weight
            for metric, weight in self._metric_weights.items()
            if metric in flattened_metrics and isinstance(weight, (int, float))
        ]
        normalizer = sum(weight for weight in self._metric_weights.values() if isinstance(weight, (int, float)))
        if weighted_values and normalizer:
            return sum(weighted_values) / normalizer

        return mean(flattened_metrics.values()) if flattened_metrics else 0.0

    async def rank_models(self, dataset_id: uuid.UUID) -> list[RankedModel]:
        result = await self._db_session.execute(
            select(EvaluationRun, PipelineConfig)
            .join(PipelineConfig, PipelineConfig.id == EvaluationRun.pipeline_config_id)
            .where(EvaluationRun.dataset_id == dataset_id, EvaluationRun.status == "completed")
        )
        rows = list(result.all())

        scores_by_model: dict[uuid.UUID, list[float]] = {}
        for run, pipeline_config in rows:
            run_metrics = run.metrics if isinstance(run.metrics, dict) else {}
            pipeline_data = pipeline_config.config if isinstance(pipeline_config.config, dict) else {}

            model_id = self._to_uuid(run_metrics.get("model_id"))
            if model_id is None:
                model_id = self._to_uuid(pipeline_data.get("model_id"))
            if model_id is None:
                continue

            score = self._compute_weighted_score(self._flatten_metrics(run_metrics))
            scores_by_model.setdefault(model_id, []).append(score)

        ranked_scores = sorted(
            ((model_id, mean(scores)) for model_id, scores in scores_by_model.items() if scores),
            key=lambda item: item[1],
            reverse=True,
        )

        await self._db_session.execute(delete(ModelRanking).where(ModelRanking.dataset_id == dataset_id))

        ranked_models: list[RankedModel] = []
        for index, (model_id, weighted_score) in enumerate(ranked_scores, start=1):
            model = await self._db_session.get(Model, model_id)
            if model is None:
                continue

            ranked_models.append(
                RankedModel(
                    model_id=model.id,
                    model_name=model.name,
                    provider=model.provider,
                    weighted_score=float(weighted_score),
                    rank=index,
                )
            )
            self._db_session.add(
                ModelRanking(
                    dataset_id=dataset_id,
                    model_id=model.id,
                    weighted_score=float(weighted_score),
                    rank=index,
                )
            )

        await self._db_session.commit()
        return ranked_models
