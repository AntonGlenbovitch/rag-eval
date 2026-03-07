from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import TYPE_CHECKING, Any, Callable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.evaluation import EvaluationRun, PipelineConfig, PipelineExperiment
from src.services.embedding_service import EmbeddingService

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


@dataclass(slots=True)
class FailureCluster:
    label: str
    size: int
    queries: list[str]
    run_ids: list[uuid.UUID]


@dataclass(slots=True)
class EvaluationRunAnalysis:
    total_runs: int
    completed_runs: int
    failed_runs: int
    average_score: float
    best_run_id: uuid.UUID | None
    best_score: float | None
    failure_clusters: list[FailureCluster]


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
        embedding_service: EmbeddingService | None = None,
        anthropic_client: "AsyncAnthropic | None" = None,
        anthropic_model: str = "claude-3-5-sonnet-latest",
    ) -> None:
        self._db_session = db_session
        self._enqueue_evaluation_run = enqueue_evaluation_run
        self._embedding_service = embedding_service
        self._anthropic_client = anthropic_client
        self._anthropic_model = anthropic_model

    @staticmethod
    def _extract_score(run: EvaluationRun) -> float | None:
        if run.metrics is None:
            return None

        raw_score = run.metrics.get("score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)

        return None

    @staticmethod
    def _extract_failed_query(run: EvaluationRun) -> str | None:
        if not run.metrics:
            return None

        raw_query = run.metrics.get("question") or run.metrics.get("query")
        if isinstance(raw_query, str):
            query = raw_query.strip()
            if query:
                return query

        return None

    @staticmethod
    def _distance_sq(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=True))

    @staticmethod
    def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
        if not vectors:
            return []

        dimensions = len(vectors[0])
        totals = [0.0] * dimensions
        for vector in vectors:
            for index, value in enumerate(vector):
                totals[index] += value

        return [value / len(vectors) for value in totals]

    @classmethod
    def _kmeans_cluster(cls, embeddings: Sequence[Sequence[float]], k: int, max_iterations: int = 20) -> list[int]:
        if not embeddings:
            return []

        centroids = [list(vector) for vector in embeddings[:k]]
        labels = [0] * len(embeddings)

        for _ in range(max_iterations):
            changed = False

            for index, vector in enumerate(embeddings):
                closest_index = min(range(k), key=lambda c: cls._distance_sq(vector, centroids[c]))
                if labels[index] != closest_index:
                    labels[index] = closest_index
                    changed = True

            grouped: list[list[Sequence[float]]] = [[] for _ in range(k)]
            for label, vector in zip(labels, embeddings, strict=True):
                grouped[label].append(vector)

            for centroid_index in range(k):
                if grouped[centroid_index]:
                    centroids[centroid_index] = cls._mean_vector(grouped[centroid_index])

            if not changed:
                break

        return labels

    async def _label_failure_cluster(self, queries: Sequence[str], cluster_index: int) -> str:
        if self._anthropic_client is None:
            sample = "; ".join(queries[:2])
            return f"Cluster {cluster_index + 1}: {sample}" if sample else f"Cluster {cluster_index + 1}"

        prompt = (
            "You are labeling clusters of failed RAG queries.\n"
            "Given the queries, produce a concise label (3-8 words) for the shared failure theme.\n"
            "Return JSON only with schema: {\"label\": string}.\n\n"
            "Queries:\n"
            + "\n".join(f"- {query}" for query in queries[:12])
        )

        response = await self._anthropic_client.messages.create(
            model=self._anthropic_model,
            max_tokens=80,
            temperature=0.0,
            system="Respond with strict JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        payload = json.loads("\n".join(text_blocks).strip())
        label = payload.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
        return f"Cluster {cluster_index + 1}"

    async def _build_failure_clusters(self, failed_runs: Sequence[EvaluationRun]) -> list[FailureCluster]:
        if self._embedding_service is None:
            return []

        query_rows = [
            (run.id, query)
            for run in failed_runs
            if (query := self._extract_failed_query(run)) is not None
        ]
        if len(query_rows) < 2:
            return []

        queries = [query for _, query in query_rows]
        embeddings = self._embedding_service.embed_batch(queries)

        k = min(5, max(2, round(len(queries) ** 0.5)))
        labels = self._kmeans_cluster(embeddings, k=k)

        grouped: dict[int, list[tuple[uuid.UUID, str]]] = {}
        for label, row in zip(labels, query_rows, strict=True):
            grouped.setdefault(label, []).append(row)

        clusters: list[FailureCluster] = []
        for cluster_index in sorted(grouped.keys()):
            rows = grouped[cluster_index]
            cluster_queries = [query for _, query in rows]
            cluster_label = await self._label_failure_cluster(cluster_queries, cluster_index)
            clusters.append(
                FailureCluster(
                    label=cluster_label,
                    size=len(rows),
                    queries=cluster_queries,
                    run_ids=[run_id for run_id, _ in rows],
                )
            )

        return sorted(clusters, key=lambda item: item.size, reverse=True)

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

        failure_clusters = await self._build_failure_clusters(failed_runs)

        return EvaluationRunAnalysis(
            total_runs=len(runs),
            completed_runs=len(completed_runs),
            failed_runs=len(failed_runs),
            average_score=average_score,
            best_run_id=best_run_id,
            best_score=best_score,
            failure_clusters=failure_clusters,
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
