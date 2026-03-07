import unittest
from types import SimpleNamespace

from src.providers.llm import AnthropicProvider, LocalModelProvider, OpenAIProvider


class FakeOpenAICompletions:
    def __init__(self, response):
        self._response = response

    async def create(self, **kwargs):
        return self._response


class FakeOpenAIClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions(response))


class FakeAnthropicMessages:
    def __init__(self, response):
        self._response = response

    async def create(self, **kwargs):
        return self._response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeAnthropicMessages(response)


class LLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_provider_normalizes_output(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello from OpenAI"))],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        )
        provider = OpenAIProvider(client=FakeOpenAIClient(response), model="gpt-test")

        generated = await provider.generate_response("Say hello")

        self.assertEqual(generated.provider, "openai")
        self.assertEqual(generated.model, "gpt-test")
        self.assertEqual(generated.text, "Hello from OpenAI")
        self.assertEqual(generated.usage["total_tokens"], 9)

    async def test_anthropic_provider_normalizes_output(self) -> None:
        response = SimpleNamespace(
            content=[SimpleNamespace(text="Hello"), SimpleNamespace(text="Anthropic")],
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )
        provider = AnthropicProvider(client=FakeAnthropicClient(response), model="claude-test")

        generated = await provider.generate_response("Say hello")

        self.assertEqual(generated.provider, "anthropic")
        self.assertEqual(generated.text, "Hello\nAnthropic")
        self.assertEqual(generated.usage["prompt_tokens"], 7)
        self.assertEqual(generated.usage["completion_tokens"], 3)
        self.assertEqual(generated.usage["total_tokens"], 10)

    async def test_local_model_provider_normalizes_output(self) -> None:
        provider = LocalModelProvider(model="llama3")

        def fake_post_generate(payload):
            return {
                "response": "Hello from Ollama",
                "prompt_eval_count": 12,
                "eval_count": 8,
            }

        provider._post_generate = fake_post_generate  # type: ignore[method-assign]

        generated = await provider.generate_response("Say hello")

        self.assertEqual(generated.provider, "ollama")
        self.assertEqual(generated.model, "llama3")
        self.assertEqual(generated.text, "Hello from Ollama")
        self.assertEqual(generated.usage["total_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
