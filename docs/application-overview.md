# Application Overview

## 1. Application Purpose

This repository implements a backend-oriented **RAG evaluation and optimization scaffold** called `ragprobe`. It provides APIs and background jobs for:

- creating datasets, QA pairs, and vectorized knowledge chunks,
- running evaluation jobs that combine retrieval metrics with LLM-judge scoring,
- benchmarking and ranking models,
- routing queries to pipelines/models based on historical performance,
- generating optimization artifacts (candidate configs, experiment scheduling, clustering of failed runs).

From the code and docs, the intended usage context appears to be teams iterating on Retrieval-Augmented Generation pipelines and comparing model/pipeline behavior over time.

## 2. High-Level Architecture Overview

The system is a **service-style FastAPI backend** with:

- a REST API layer (`src/api/routes.py`),
- a service layer containing evaluation, retrieval, ranking, routing, and optimization logic,
- PostgreSQL + pgvector for persistence and similarity search,
- Redis for embedding cache and Celery broker/backend,
- Celery worker(s) for asynchronous evaluation execution,
- provider adapters for OpenAI, Anthropic, and local/Ollama-compatible generation.

Architecturally, this is a modular monolith with clear domain services rather than microservices.

## 3. Repository Structure

Key areas:

- `src/main.py`: FastAPI app bootstrap and router mounting.
- `src/api/routes.py`: all HTTP endpoints and request/response models.
- `src/core/`: config, DB session factory, Celery app initialization.
- `src/models/`: SQLAlchemy ORM models for datasets, runs, models, rankings, routing decisions.
- `src/services/`: business logic (embedding, retrieval, evaluation, optimization, ranking, routing, reports).
- `src/providers/llm/`: provider abstraction + concrete LLM integrations.
- `src/tasks/`: Celery tasks that execute evaluation runs asynchronously.
- `src/evaluation/retrieval_metrics.py`: retrieval metric calculations.
- `alembic/` + `alembic/versions/`: schema migration history.
- `tests/`: unit tests for API registration and major services.
- `docker-compose.yml`, `Dockerfile`: local runtime topology.

## 4. Entry Points and Startup Flow

Primary runtime entry points:

1. **API server**: `uvicorn src.main:app` (Docker `api` service).
2. **Celery worker**: `celery -A src.tasks.worker worker --loglevel=info` (Docker `celery` service).

Startup sequence (API):

1. `Settings` loaded from environment / `.env` via `pydantic-settings`.
2. SQLAlchemy async engine/session factory initialized.
3. FastAPI app created and router included.
4. Requests resolve `get_db()` dependency for per-request `AsyncSession`.

Startup sequence (worker):

1. Celery app created with Redis broker/result backend.
2. `src.tasks.jobs` registered as task module.
3. `run_evaluation` task creates event loop and invokes async evaluation workflow.

## 5. Core Components and Responsibilities

### API Layer (`src/api/routes.py`)
- Defines all endpoints and payload models.
- Performs lightweight validation and error responses (e.g., 404 when dataset/run not found).
- Instantiates and orchestrates domain services.

### Persistence Layer (`src/models/*`, `src/core/database.py`, Alembic)
- ORM entities for datasets, chunks, pipeline configs, runs, experiments, models, rankings, routing logs.
- PostgreSQL + pgvector schema and migration lifecycle.

### Evaluation Pipeline (`EvaluationService`)
- Embeds query text.
- Retrieves relevant chunks by vector similarity.
- Computes retrieval metrics.
- Uses Anthropic judge service for quality metrics.
- Computes overall score and persists run/experiment results.
- Optionally routes to selected model/pipeline before generation.

### Retrieval & Embedding
- `EmbeddingService`: OpenAI embedding generation with Redis cache.
- `RetrievalService`: SQL vector search against `knowledge_chunks` with threshold and top-k limits.

### Optimization & Analysis
- `OptimizationService`: summarizes run performance, clusters failed queries, generates parameter candidates, schedules experiments, compares/selects best pipelines.

### Benchmarking and Ranking
- `ModelBenchmarkService`: executes evaluations across model IDs for a dataset.
- `ModelRankingService`: computes weighted model scores and stores ranking table.
- `ModelSelectorService`: fetches top-ranked model for a dataset.

### Adaptive Routing
- `QueryAnalyzer`: heuristic feature extraction for query type/difficulty.
- `RoutingPolicyService`: combines historical metrics, ranking bonus, cost efficiency, and query features to choose pipeline/model.
- `RoutingDecision` model captures per-query routing audit log.

### LLM Provider Abstraction
- `LLMProvider` base interface with concrete OpenAI, Anthropic, and local/Ollama providers.
- `ProviderFactory` resolves provider from model record or provider string.

### Background Execution
- Celery task `tasks.run_evaluation` transitions run state and executes `EvaluationService` asynchronously.

## 6. End-to-End Processing Flow

Typical evaluation execution flow:

1. Client creates dataset/QA/chunks via API.
2. Client creates an `evaluation_run` for a pipeline config.
3. Client triggers execution endpoint, which enqueues Celery task.
4. Worker loads run + first QA pair from dataset.
5. Evaluation service runs retrieval + judge scoring pipeline.
6. Results are persisted in `evaluation_runs.metrics` and `pipeline_experiments`.
7. Analysis/ranking/routing endpoints consume persisted results.

## 7. Data Flow

Primary data movement:

- **Input data**: dataset metadata, QA pairs, and precomputed chunk embeddings are written to PostgreSQL.
- **Query-time embedding**: questions are converted to vectors via OpenAI embedding API, cached in Redis by hash key.
- **Retrieval**: query vector is used in pgvector similarity SQL against `knowledge_chunks`.
- **Evaluation transform**:
  - retrieved IDs compared with expected IDs for retrieval metrics,
  - answer/context scored by Claude judge,
  - aggregate score computed as average of retrieval + judge metric values.
- **Persistence**: run status/metrics and experiment record committed to DB.
- **Derived artifacts**: model rankings and routing decisions stored in dedicated tables.

## 8. External Dependencies and Integrations

Observed integrations in code:

- **PostgreSQL + pgvector**: main store and vector search.
- **Redis**: embedding cache + Celery broker/result backend.
- **OpenAI API**:
  - embeddings (`EmbeddingService`),
  - optional chat generation through provider abstraction.
- **Anthropic API**:
  - Claude judge scoring,
  - optional failure-cluster labeling.
- **Local/Ollama-compatible HTTP endpoint** (`/api/generate`) for local model generation.

No message queue beyond Celery/Redis, no object storage integration, and no frontend/UI in this repository.

## 9. Configuration and Environment

Configuration is centralized in `src/core/config.py` and supports `.env` loading.

Main settings groups:

- app: name/env/host/port,
- DB: database URL + postgres parts,
- Redis: host/port/url,
- Celery broker/backend URLs,
- provider credentials/models (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, embedding/judge model names).

Docker Compose wires API, worker, Postgres, and Redis; DB init script enables pgvector extension.

## 10. Error Handling, Logging, and Observability

Observed approach:

- API-level validation failures return HTTP 404/400 where implemented.
- Worker catches evaluation exceptions, marks run `failed`, and persists error message in `metrics.error`.
- `EvaluationService` swallows provider-generation exceptions in optional routed-answer generation and falls back to provided answer.
- No explicit structured logging, tracing, metrics export, or centralized observability stack is present.

## 11. Testing Strategy

The repository includes unit tests focused on:

- route registration,
- retrieval metrics utilities,
- embedding/retrieval/evaluation services,
- optimization, ranking, benchmark, routing logic,
- Celery evaluation task behavior,
- report generation,
- provider normalization behavior.

Most tests are in-memory/fake-based unit tests; there is no evident end-to-end integration test with real Postgres/Redis/provider APIs.

## 12. Design Patterns and Implementation Notes

Patterns actually used:

- **Service layer**: domain logic encapsulated in dedicated service classes.
- **Adapter/provider pattern**: provider-specific implementations normalized through `LLMProvider` interface.
- **Factory pattern**: `ProviderFactory` for provider selection from runtime identifiers.
- **Task orchestration**: Celery task wraps async workflow for background execution.
- **Heuristic policy routing**: rule/weight-based pipeline choice from persisted signals.

Dependency injection is lightweight/manual (constructor injection or FastAPI dependencies), not container-based.

## 13. Limitations, Assumptions, and Gaps

Repository-grounded limitations:

- README is minimal; several flows are only discoverable via source/tests.
- No authentication/authorization layer present.
- No explicit schema for API versioning consistency; router is included both unprefixed and with `/api/v1`, while several route paths already embed `/api/v1` (leading to duplicated-path variants).
- Benchmark flow appears to reuse `EvaluationRun.pipeline_config_id` with model IDs, which is likely an intentional shortcut or a model/pipeline ID coupling assumption.
- Run execution currently uses the first QA pair per dataset (`limit(1)`) rather than full dataset sweeps.
- Observability and production-hardening (logging, retries, rate-limit handling, circuit breakers) are minimal.
- Some architectural conclusions (e.g., intended persona/workflow depth) are inferred from code/tests and not extensively documented in repository docs.
