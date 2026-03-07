import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


@dataclass(slots=True)
class ClaudeJudgeEvaluation:
    faithfulness: float
    relevance: float
    hallucination: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "hallucination": self.hallucination,
            "confidence": self.confidence,
        }


class ClaudeJudgeService:
    def __init__(
        self,
        anthropic_client: "AsyncAnthropic",
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._anthropic_client = anthropic_client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()

        if "```" in text:
            parts = text.split("```")
            for block in parts:
                candidate = block.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if not candidate:
                    continue
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        return json.loads(text)

    @staticmethod
    def _coerce_scores(payload: dict[str, Any]) -> ClaudeJudgeEvaluation:
        return ClaudeJudgeEvaluation(
            faithfulness=float(payload["faithfulness"]),
            relevance=float(payload["relevance"]),
            hallucination=float(payload["hallucination"]),
            confidence=float(payload["confidence"]),
        )

    async def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
    ) -> dict[str, float]:
        context_text = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
        user_prompt = (
            "Evaluate the answer quality for the question and provided context.\n"
            "Score each metric from 0.0 to 1.0 where higher is better.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Context:\n{context_text if context_text else '(no context provided)'}\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "faithfulness": number,\n'
            '  "relevance": number,\n'
            '  "hallucination": number,\n'
            '  "confidence": number\n'
            "}\n"
        )

        response = await self._anthropic_client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=(
                "You are an impartial RAG evaluator. "
                "Respond with strict JSON only and no surrounding prose."
            ),
            messages=[{"role": "user", "content": user_prompt}],
        )

        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        raw_text = "\n".join(text_blocks).strip()
        payload = self._extract_json_payload(raw_text)
        return self._coerce_scores(payload).to_dict()
