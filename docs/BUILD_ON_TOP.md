# Build On Top

## Adaptive RAG Routing

Phase 16 introduces adaptive pipeline routing driven by historical evaluation outcomes.

- **Query analysis:** `QueryAnalyzer` classifies incoming questions into factual/analytical types, estimates difficulty, and flags reasoning/long-context requirements.
- **Pipeline scoring:** `RoutingPolicyService` computes scores from stored metrics with weighted components:
  - `0.40 * faithfulness`
  - `0.25 * relevance`
  - `0.15 * recall_at_k`
  - `0.10 * mrr`
  - `0.10 * cost_efficiency`
- **Model selection:** top pipelines are selected by combining evaluation signals, model rankings, reasoning boosts, and context-window preferences.
- **Routing logs:** every routing decision is written to `routing_decisions` for traceability and downstream analytics.
