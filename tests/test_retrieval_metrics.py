import unittest

from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_at_k_handles_duplicates_and_k_larger_than_results(self) -> None:
        retrieved_ids = ["a", "a", "c"]
        relevant_ids = ["a", "a", "b"]

        self.assertEqual(recall_at_k(retrieved_ids, relevant_ids, k=10), 0.5)

    def test_precision_at_k_handles_duplicates_and_k_larger_than_results(self) -> None:
        retrieved_ids = ["a", "a", "b", "x"]
        relevant_ids = ["a", "b"]

        self.assertEqual(precision_at_k(retrieved_ids, relevant_ids, k=10), 2 / 3)

    def test_reciprocal_rank_uses_first_relevant_result(self) -> None:
        retrieved_ids = ["x", "x", "a", "b"]
        relevant_ids = ["a", "b"]

        self.assertEqual(reciprocal_rank(retrieved_ids, relevant_ids), 0.5)

    def test_empty_inputs_return_zero_metrics(self) -> None:
        self.assertEqual(recall_at_k([], [], 5), 0.0)
        self.assertEqual(precision_at_k([], [], 5), 0.0)
        self.assertEqual(reciprocal_rank([], []), 0.0)
        self.assertEqual(mean_reciprocal_rank([], []), 0.0)

    def test_mean_reciprocal_rank(self) -> None:
        retrieved_lists = [["x", "a"], ["b", "a"], ["z"]]
        relevant_lists = [["a"], ["a"], ["q"]]

        self.assertAlmostEqual(mean_reciprocal_rank(retrieved_lists, relevant_lists), (0.5 + 0.5 + 0.0) / 3)

    def test_compute_retrieval_metrics(self) -> None:
        metrics = compute_retrieval_metrics(
            retrieved_ids=["x", "a", "b"],
            relevant_ids=["a", "c"],
            k=2,
        )

        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)


if __name__ == "__main__":
    unittest.main()
