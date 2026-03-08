from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QueryFeatures:
    query_length: int
    query_type: str
    requires_reasoning: bool
    requires_long_context: bool
    difficulty_estimate: str


class QueryAnalyzer:
    REASONING_KEYWORDS = ("why", "how", "explain")

    def analyze_query(self, query: str) -> QueryFeatures:
        normalized = query.strip().lower()
        query_length = len(query)

        requires_reasoning = any(keyword in normalized for keyword in self.REASONING_KEYWORDS)
        requires_long_context = query_length > 120

        is_short_question = query_length < 80 and normalized.endswith("?") and not requires_reasoning
        query_type = "factual" if is_short_question else "analytical"

        if requires_reasoning and requires_long_context:
            difficulty = "high"
        elif requires_reasoning or requires_long_context:
            difficulty = "medium"
        else:
            difficulty = "low"

        return QueryFeatures(
            query_length=query_length,
            query_type=query_type,
            requires_reasoning=requires_reasoning,
            requires_long_context=requires_long_context,
            difficulty_estimate=difficulty,
        )
