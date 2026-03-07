from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from src.models.evaluation import Dataset, EvaluationRun


class ReportGenerator:
    """Generate markdown reports for evaluation runs and datasets."""

    _METRIC_RECOMMENDATIONS: dict[str, str] = {
        "precision_at_k": "Improve chunk relevance by refining embeddings or query expansion.",
        "recall_at_k": "Increase retrieval coverage by raising `k` or improving chunking strategy.",
        "reciprocal_rank": "Tune ranking and reranking so relevant chunks appear earlier.",
        "faithfulness": "Strengthen grounding by providing better context and stricter generation prompts.",
        "relevance": "Adjust prompts and retrieval filters to better align answers with the question.",
        "hallucination": "Reduce unsupported claims by enforcing citation-heavy responses and guardrails.",
        "confidence": "Raise answer confidence with higher quality context and clearer answer formatting.",
        "score": "Investigate weak retrieval and judge dimensions to improve overall run quality.",
    }

    def generate_run_report(self, run: EvaluationRun) -> str:
        metrics = run.metrics or {}
        retrieval_metrics = self._metrics_dict(metrics.get("retrieval_metrics"))
        judge_scores = self._metrics_dict(metrics.get("judge_scores"))

        report_lines = [
            "# Run Report",
            "",
            "## Run Details",
            f"- **Run ID:** `{run.id}`",
            f"- **Dataset ID:** `{run.dataset_id}`",
            f"- **Pipeline Config ID:** `{run.pipeline_config_id}`",
            f"- **Status:** `{run.status}`",
            f"- **Started At:** {self._format_datetime(run.started_at)}",
            f"- **Completed At:** {self._format_datetime(run.completed_at)}",
            "",
            "## Retrieval Metrics",
            self._metrics_table(retrieval_metrics),
            "",
            "## Judge Metrics",
            self._metrics_table(judge_scores),
            "",
            "## Overall",
            self._metrics_table({"score": metrics.get("score")}),
            "",
            "## Recommendations",
            *self._recommendation_lines({**retrieval_metrics, **judge_scores, "score": metrics.get("score")}),
        ]

        return "\n".join(report_lines).strip() + "\n"

    def generate_dataset_report(self, dataset: Dataset, runs: list[EvaluationRun]) -> str:
        completed_runs = [run for run in runs if run.status == "completed"]

        aggregate_metrics = self._aggregate_metrics(completed_runs)

        report_lines = [
            "# Dataset Report",
            "",
            "## Dataset Details",
            f"- **Dataset ID:** `{dataset.id}`",
            f"- **Dataset Name:** {dataset.name}",
            f"- **Description:** {dataset.description or 'N/A'}",
            f"- **Created At:** {self._format_datetime(dataset.created_at)}",
            f"- **Total Runs:** {len(runs)}",
            f"- **Completed Runs:** {len(completed_runs)}",
            "",
            "## Run Summary",
            self._run_summary_table(runs),
            "",
            "## Aggregate Metrics (Completed Runs)",
            self._metrics_table(aggregate_metrics),
            "",
            "## Recommendations",
            *self._dataset_recommendation_lines(runs, aggregate_metrics),
        ]

        return "\n".join(report_lines).strip() + "\n"

    @staticmethod
    def _metrics_dict(raw_metrics: Any) -> dict[str, float]:
        if not isinstance(raw_metrics, dict):
            return {}

        parsed: dict[str, float] = {}
        for key, value in raw_metrics.items():
            if isinstance(value, int | float):
                parsed[key] = float(value)
        return parsed

    def _aggregate_metrics(self, runs: list[EvaluationRun]) -> dict[str, float]:
        sums: defaultdict[str, float] = defaultdict(float)
        counts: defaultdict[str, int] = defaultdict(int)

        for run in runs:
            metrics = run.metrics or {}
            grouped_metrics = {
                **self._metrics_dict(metrics.get("retrieval_metrics")),
                **self._metrics_dict(metrics.get("judge_scores")),
            }
            if isinstance(metrics.get("score"), int | float):
                grouped_metrics["score"] = float(metrics["score"])

            for metric_name, metric_value in grouped_metrics.items():
                sums[metric_name] += metric_value
                counts[metric_name] += 1

        return {name: sums[name] / counts[name] for name in sorted(sums.keys()) if counts[name] > 0}

    @staticmethod
    def _metrics_table(metrics: dict[str, float | None]) -> str:
        if not metrics:
            return "No metrics available."

        lines = ["| Metric | Value |", "| --- | ---: |"]
        for metric_name in sorted(metrics.keys()):
            lines.append(f"| {metric_name} | {ReportGenerator._format_metric_value(metrics[metric_name])} |")
        return "\n".join(lines)

    def _run_summary_table(self, runs: list[EvaluationRun]) -> str:
        if not runs:
            return "No runs available."

        lines = [
            "| Run ID | Status | Score | Completed At |",
            "| --- | --- | ---: | --- |",
        ]

        for run in runs:
            metrics = run.metrics or {}
            score = metrics.get("score") if isinstance(metrics, dict) else None
            lines.append(
                "| {run_id} | {status} | {score} | {completed_at} |".format(
                    run_id=run.id,
                    status=run.status,
                    score=self._format_metric_value(score),
                    completed_at=self._format_datetime(run.completed_at),
                )
            )

        return "\n".join(lines)

    def _recommendation_lines(self, metrics: dict[str, float | None]) -> list[str]:
        recommendations = self._collect_recommendations(metrics)
        if not recommendations:
            return ["- No immediate issues detected. Continue monitoring trends across runs."]
        return recommendations

    def _dataset_recommendation_lines(self, runs: list[EvaluationRun], metrics: dict[str, float]) -> list[str]:
        recommendations = []
        incomplete_runs = [run for run in runs if run.status != "completed"]

        if incomplete_runs:
            recommendations.append(
                f"- Resolve {len(incomplete_runs)} incomplete run(s) to improve report coverage and metric confidence."
            )

        recommendations.extend(self._collect_recommendations(metrics))

        if not recommendations:
            return ["- Dataset performance looks stable; keep tracking score drift over time."]

        return recommendations

    def _collect_recommendations(self, metrics: dict[str, float | None]) -> list[str]:
        recommendations = []
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, int | float):
                continue

            if metric_value < 0.7 and metric_name in self._METRIC_RECOMMENDATIONS:
                recommendations.append(
                    f"- **{metric_name}** is {metric_value:.3f}. {self._METRIC_RECOMMENDATIONS[metric_name]}"
                )

        return recommendations

    @staticmethod
    def _format_metric_value(value: float | None) -> str:
        if not isinstance(value, int | float):
            return "N/A"
        return f"{float(value):.3f}"

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if value is None:
            return "N/A"
        return value.isoformat()
