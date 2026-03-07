from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.evaluation import EvaluationRun, PipelineConfig, PipelineExperiment


@dataclass(slots=True)
class EvaluationRunAnalysis:
    total_runs: int
    completed_runs: int
    failed_runs: int
    average_score: float
    best_run_id: uuid.UUID | None
    best_score: float | None


@dataclass(slots=True)
class PipelineComparison:
    pipeline_config_id: uuid.UUID
    evaluation_run_id: uuid.UUID | None
    score: float
    metrics: dict[str, float]


class OptimizationService:
    def __init__(
        self,
        db_session: AsyncSession,
        *,
        enqueue_evaluation_run: Callable[[str], Any] | None = None,
    ) -> None:
        self._db_session = db_session
        self._enqueue_evaluation_run = enqueue_evaluation_run

    @staticmethod
    def _extract_score(run: EvaluationRun) -> float | None:
        if run.metrics is None:
            return None

        raw_score = run.metrics.get("score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)

        return None

    async def analyze_evaluation_runs(
        self,
        *,
        dataset_id: uuid.UUID | None = None,
        pipeline_config_id: uuid.UUID | None = None,
    ) -> EvaluationRunAnalysis:
        query = select(EvaluationRun)
        if dataset_id is not None:
            query = query.where(EvaluationRun.dataset_id == dataset_id)
        if pipeline_config_id is not None:
            query = query.where(EvaluationRun.pipeline_config_id == pipeline_config_id)

        result = await self._db_session.execute(query)
        runs = list(result.scalars().all())

        completed_runs = [run for run in runs if run.status == "completed"]
        failed_runs = [run for run in runs if run.status == "failed"]

        scored_runs = [(run.id, score) for run in completed_runs if (score := self._extract_score(run)) is not None]

        average_score = mean([score for _, score in scored_runs]) if scored_runs else 0.0
        best_run_id: uuid.UUID | None = None
        best_score: float | None = None
        if scored_runs:
            best_run_id, best_score = max(scored_runs, key=lambda item: item[1])

        return EvaluationRunAnalysis(
            total_runs=len(runs),
            completed_runs=len(completed_runs),
            failed_runs=len(failed_runs),
            average_score=average_score,
            best_run_id=best_run_id,
            best_score=best_score,
        )

    @staticmethod
    def generate_pipeline_candidates(
        *,
        base_config: dict[str, Any],
        search_space: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        if not search_space:
            return [dict(base_config)]

        keys = sorted(search_space.keys())
        values_product = itertools.product(*(search_space[key] for key in keys))

        candidates: list[dict[str, Any]] = []
        for values in values_product:
            candidate = dict(base_config)
            candidate.update(dict(zip(keys, values, strict=True)))
            candidates.append(candidate)

        return candidates

    async def schedule_evaluation_experiments(
        self,
        *,
        dataset_id: uuid.UUID,
        provider: str,
        pipeline_candidates: Sequence[dict[str, Any]],
    ) -> list[EvaluationRun]:
        created_runs: list[EvaluationRun] = []

        for index, candidate in enumerate(pipeline_candidates, start=1):
            pipeline_config = PipelineConfig(
                name=f"candidate-{index}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                provider=provider,
                config=dict(candidate),
            )
            self._db_session.add(pipeline_config)
            await self._db_session.flush()

            run = EvaluationRun(
                dataset_id=dataset_id,
                pipeline_config_id=pipeline_config.id,
                status="pending",
            )
            self._db_session.add(run)
            created_runs.append(run)

        await self._db_session.commit()

        if self._enqueue_evaluation_run is not None:
            for run in created_runs:
                self._enqueue_evaluation_run(str(run.id))

        return created_runs

    @staticmethod
    def _flatten_metrics(results: dict[str, Any] | None) -> dict[str, float]:
        if not results:
            return {}

        flattened: dict[str, float] = {}

        for key, value in results.items():
            if isinstance(value, (int, float)):
                flattened[key] = float(value)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        flattened[f"{key}.{nested_key}"] = float(nested_value)

        return flattened

    def compare_metrics(
        self,
        *,
        experiments: Sequence[PipelineExperiment],
        metric_weights: dict[str, float] | None = None,
    ) -> list[PipelineComparison]:
        comparisons: list[PipelineComparison] = []

        for experiment in experiments:
            metrics = self._flatten_metrics(experiment.results)
            if metric_weights:
                weighted_values = [
                    metrics[metric] * weight
                    for metric, weight in metric_weights.items()
                    if metric in metrics and isinstance(weight, (int, float))
                ]
                normalizer = sum(weight for weight in metric_weights.values() if isinstance(weight, (int, float)))
                score = sum(weighted_values) / normalizer if weighted_values and normalizer else 0.0
            else:
                score = experiment.score if experiment.score is not None else (mean(metrics.values()) if metrics else 0.0)

            comparisons.append(
                PipelineComparison(
                    pipeline_config_id=experiment.pipeline_config_id,
                    evaluation_run_id=experiment.evaluation_run_id,
                    score=float(score),
                    metrics=metrics,
                )
            )

        return sorted(comparisons, key=lambda item: item.score, reverse=True)

    def select_best_pipeline(
        self,
        *,
        experiments: Sequence[PipelineExperiment],
        metric_weights: dict[str, float] | None = None,
    ) -> PipelineComparison | None:
        ranked = self.compare_metrics(experiments=experiments, metric_weights=metric_weights)
        if not ranked:
            return None

        return ranked[0]
