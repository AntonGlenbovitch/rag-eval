import unittest

from src.main import app


class ApiRoutesTests(unittest.TestCase):
    def test_expected_api_routes_are_registered(self):
        route_paths = {route.path for route in app.routes}

        expected_paths = {
            "/health",
            "/datasets",
            "/datasets/{dataset_id}/qa-pairs",
            "/datasets/{dataset_id}/chunks",
            "/embeddings",
            "/evaluation/runs",
            "/evaluation/runs/{run_id}",
            "/evaluation/runs/{run_id}/execute",
            "/analysis",
            "/optimization/candidates",
            "/optimization/experiments/schedule",
            "/reports/pipelines",
            "/clusters",
            "/api/v1/benchmark/{dataset_id}",
            "/api/v1/models",
            "/api/v1/models/ranking/{dataset_id}",
            "/api/v1/route",
            "/api/v1/routing/stats/{dataset_id}",
        }

        self.assertTrue(expected_paths.issubset(route_paths))


if __name__ == "__main__":
    unittest.main()
