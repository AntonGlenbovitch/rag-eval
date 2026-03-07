import unittest
import uuid
from datetime import datetime, timezone

from src.models.evaluation import Dataset, EvaluationRun
from src.services.report_generator import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = ReportGenerator()

    def test_generate_run_report_includes_metrics_tables_and_recommendations(self) -> None:
        run = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            pipeline_config_id=uuid.uuid4(),
            status="completed",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            metrics={
                "retrieval_metrics": {
                    "precision_at_k": 0.4,
                    "recall_at_k": 0.9,
                    "reciprocal_rank": 1.0,
                },
                "judge_scores": {
                    "faithfulness": 0.65,
                    "relevance": 0.92,
                },
                "score": 0.71,
            },
        )

        report = self.generator.generate_run_report(run)

        self.assertIn("# Run Report", report)
        self.assertIn("## Retrieval Metrics", report)
        self.assertIn("| precision_at_k | 0.400 |", report)
        self.assertIn("## Judge Metrics", report)
        self.assertIn("| faithfulness | 0.650 |", report)
        self.assertIn("## Recommendations", report)
        self.assertIn("**precision_at_k** is 0.400", report)
        self.assertIn("**faithfulness** is 0.650", report)

    def test_generate_dataset_report_aggregates_metrics_and_flags_incomplete_runs(self) -> None:
        dataset = Dataset(
            id=uuid.uuid4(),
            name="Product QA",
            description="Questions about a product manual.",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        completed_run = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset.id,
            pipeline_config_id=uuid.uuid4(),
            status="completed",
            completed_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            metrics={
                "retrieval_metrics": {"precision_at_k": 0.8},
                "judge_scores": {"faithfulness": 0.9},
                "score": 0.85,
            },
        )
        pending_run = EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=dataset.id,
            pipeline_config_id=uuid.uuid4(),
            status="pending",
            metrics={
                "retrieval_metrics": {"precision_at_k": 0.2},
                "judge_scores": {"faithfulness": 0.1},
                "score": 0.15,
            },
        )

        report = self.generator.generate_dataset_report(dataset, [completed_run, pending_run])

        self.assertIn("# Dataset Report", report)
        self.assertIn("| precision_at_k | 0.800 |", report)
        self.assertIn("| faithfulness | 0.900 |", report)
        self.assertIn("| score | 0.850 |", report)
        self.assertIn("Resolve 1 incomplete run(s)", report)


if __name__ == "__main__":
    unittest.main()
