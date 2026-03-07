import unittest

from src.services.claude_judge_service import ClaudeJudgeService


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [FakeTextBlock(text)]


class FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._response_text)


class FakeAnthropicClient:
    def __init__(self, response_text: str) -> None:
        self.messages = FakeMessages(response_text)


class ClaudeJudgeServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_returns_metrics_dict(self) -> None:
        client = FakeAnthropicClient(
            '{"faithfulness": 0.9, "relevance": 0.8, "hallucination": 0.95, "confidence": 0.7}'
        )
        service = ClaudeJudgeService(anthropic_client=client, model="claude-test")

        result = await service.evaluate(
            question="What is RAG?",
            answer="RAG combines retrieval and generation.",
            context_chunks=["RAG uses retrieved context to answer questions."],
        )

        self.assertEqual(
            result,
            {
                "faithfulness": 0.9,
                "relevance": 0.8,
                "hallucination": 0.95,
                "confidence": 0.7,
            },
        )
        self.assertEqual(len(client.messages.calls), 1)

    async def test_evaluate_parses_json_code_block(self) -> None:
        client = FakeAnthropicClient(
            """```json
{
  \"faithfulness\": 0.75,
  \"relevance\": 0.65,
  \"hallucination\": 0.8,
  \"confidence\": 0.6
}
```"""
        )
        service = ClaudeJudgeService(anthropic_client=client, model="claude-test")

        result = await service.evaluate(
            question="Q",
            answer="A",
            context_chunks=[],
        )

        self.assertEqual(result["faithfulness"], 0.75)
        self.assertEqual(result["relevance"], 0.65)
        self.assertEqual(result["hallucination"], 0.8)
        self.assertEqual(result["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
