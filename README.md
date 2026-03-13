# ragprobe

`ragprobe` is a backend service for evaluating Retrieval-Augmented Generation (RAG) pipelines. It provides APIs and background jobs to store evaluation datasets, run retrieval+LLM-judge evaluations, benchmark models, and route incoming queries based on historical performance.

# Overview

This project exists to make RAG iteration more systematic.

Instead of evaluating prompts/pipelines ad hoc, `ragprobe` gives teams a repeatable workflow:
- store datasets (questions, expected answers, and chunk embeddings),
- execute evaluation runs asynchronously,
- compute retrieval and judge-based quality metrics,
- compare/rank models and select routing policies from observed results.

Primary audience: backend/ML engineers building and tuning RAG systems.

## Documentation Map

- `docs/README.md` — docs index.
- `docs/run-environment.md` — required services, env vars, and deployment/runtime topology.
- `docs/step-by-step-execution.md` — exact startup + execution sequence from API request to Celery completion.
- `docs/runtime-flowchart.md` — Mermaid flowchart for request and background-job processing.

# Features

Implemented features in the current repository:
- FastAPI endpoints for datasets, QA pairs, knowledge chunks, embeddings, evaluation runs, analysis, optimization, benchmarking, ranking, and routing.
- PostgreSQL + pgvector persistence for datasets, runs, experiments, model rankings, and routing decisions.
- Redis-backed embedding cache and Celery broker/result backend.
- Asynchronous evaluation execution via Celery (`tasks.run_evaluation`).
- Retrieval metrics computation (`precision_at_k`, `recall_at_k`, `f1_at_k`, `reciprocal_rank`).
- LLM judge integration through Anthropic for answer/context scoring.
- Provider abstraction layer for OpenAI, Anthropic, and local/Ollama-compatible generation providers.

# How It Works

High-level flow:

1. **API server startup** (`src.main:app`) loads settings, DB session factory, and routes.
2. **Data ingestion** happens through API endpoints:
   - create datasets,
   - add QA pairs,
   - add knowledge chunks with embeddings.
3. **Evaluation run creation** stores a pipeline config and pending run record.
4. **Run execution** (`POST /evaluation/runs/{run_id}/execute`) queues a Celery task.
5. **Worker execution** (`tasks.run_evaluation`) loads the run and currently evaluates the **first QA pair** in the dataset.
6. **Evaluation pipeline**:
   - embed query (OpenAI embedding API, cached in Redis),
   - retrieve chunks via pgvector similarity search,
   - compute retrieval metrics,
   - score answer/context with Anthropic judge,
   - persist metrics + experiment record.
7. **Post-run APIs** expose analysis, optimization helpers, benchmarks, rankings, and routing decisions.

# Repository Structure

- `src/main.py` — FastAPI application entry point.
- `src/api/routes.py` — API routes and request/response models.
- `src/core/` — settings, DB setup, Celery app setup.
- `src/models/` — SQLAlchemy models for datasets, runs, experiments, models, rankings, routing decisions.
- `src/services/` — evaluation, retrieval, embeddings, ranking, routing, optimization, reporting logic.
- `src/providers/llm/` — provider interface + OpenAI/Anthropic/local implementations.
- `src/tasks/` — Celery task definitions and worker entry points.
- `src/evaluation/` — retrieval metric utilities.
- `alembic/` — DB migrations.
- `tests/` — unit tests.
- `docker-compose.yml` / `Dockerfile` — local containerized setup.

# Prerequisites

From repository configuration:
- Python **3.11** (Dockerfile and `pyproject.toml` target this version).
- PostgreSQL with `pgvector` extension enabled.
- Redis.
- OpenAI API key (for embeddings endpoint/evaluation embedding stage).
- Anthropic API key (for judge scoring stage).

Optional:
- Docker + Docker Compose (easiest local setup).

# Installation

## Option A: Docker Compose (recommended)

1. Clone and enter the repository.
2. Create a `.env` file in project root (see [Configuration](#configuration)).

Example `.env` for local Docker usage:

```dotenv
APP_NAME=ragprobe
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

POSTGRES_DB=ragprobe
POSTGRES_USER=ragprobe
POSTGRES_PASSWORD=ragprobe
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://ragprobe:ragprobe@postgres:5432/ragprobe

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

3. Start all services:

```bash
docker compose up --build
```

This starts:
- API server on `http://localhost:8000`
- Celery worker
- Postgres (pgvector image)
- Redis

4. Run migrations in the API container:

```bash
docker compose exec api alembic upgrade head
```

5. Verify health:

```bash
curl http://localhost:8000/health
```

## Option B: Local Python environment

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide infrastructure services (Postgres + Redis) and `.env` settings. For local host setup, a working baseline is:

```dotenv
DATABASE_URL=postgresql+asyncpg://ragprobe:ragprobe@localhost:5432/ragprobe
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
POSTGRES_HOST=localhost
REDIS_HOST=localhost
```
4. Run database migrations:

```bash
alembic upgrade head
```

5. Start the API:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

6. Start Celery worker (separate shell):

```bash
celery -A src.tasks.worker worker --loglevel=info
```

# Configuration

Configuration is defined in `src/core/config.py` and read from environment variables / `.env`.

## Application
- `APP_NAME` (default: `ragprobe`)
- `APP_ENV` (default: `development`)
- `APP_HOST` (default: `0.0.0.0`)
- `APP_PORT` (default: `8000`)

## Database
- `DATABASE_URL` (default: `postgresql+asyncpg://ragprobe:ragprobe@db:5432/ragprobe`)
- `POSTGRES_DB` (default: `ragprobe`)
- `POSTGRES_USER` (default: `ragprobe`)
- `POSTGRES_PASSWORD` (default: `ragprobe`)
- `POSTGRES_HOST` (default: `db`)
- `POSTGRES_PORT` (default: `5432`)

## Redis / Celery
- `REDIS_URL` (default: `redis://redis:6379/0`)
- `REDIS_HOST` (default: `redis`)
- `REDIS_PORT` (default: `6379`)
- `CELERY_BROKER_URL` (default: `redis://redis:6379/0`)
- `CELERY_RESULT_BACKEND` (default: `redis://redis:6379/1`)

## LLM Providers
- `OPENAI_API_KEY` (default: empty; required for real embedding calls)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `ANTHROPIC_API_KEY` (default: empty; required for real judge scoring)
- `ANTHROPIC_MODEL` (default: `claude-3-5-sonnet-latest`)

Notes:
- No `.env.example` is currently included; you must create `.env` manually.
- Defaults for hostnames (`db`, `redis`) match Docker Compose networking. For non-Docker local runs, use reachable hostnames (for example `localhost`).

# Running the Application

Detailed runtime internals are documented in:
- `docs/run-environment.md`
- `docs/step-by-step-execution.md`
- `docs/runtime-flowchart.md`

## API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Worker

```bash
celery -A src.tasks.worker worker --loglevel=info
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

# Usage

Open API docs after startup:
- `http://localhost:8000/docs`

Typical workflow:

1. **Create dataset**
```bash
curl -X POST http://localhost:8000/datasets \
  -H "Content-Type: application/json" \
  -d '{"name":"demo-dataset","description":"sample"}'
```

2. **Add QA pair**
```bash
curl -X POST http://localhost:8000/datasets/<dataset_id>/qa-pairs \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","answer":"Retrieval-augmented generation."}'
```

3. **Add chunk with embedding**
```bash
curl -X POST http://localhost:8000/datasets/<dataset_id>/chunks \
  -H "Content-Type: application/json" \
  -d '{"content":"RAG combines retrieval and generation.","embedding":[0.01,0.02,0.03]}'
```

4. **Create evaluation run**
```bash
curl -X POST http://localhost:8000/evaluation/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"<dataset_id>","provider":"openai","pipeline_name":"baseline","config":{"model_id":"<model_uuid>"}}'
```

5. **Execute run asynchronously**
```bash
curl -X POST http://localhost:8000/evaluation/runs/<run_id>/execute
```

6. **Fetch run status/results**
```bash
curl http://localhost:8000/evaluation/runs/<run_id>
```

Important:
- The embedding vector column is fixed at dimension 1536; chunk embeddings must match this.
- Several routes are available with and without `/api/v1` prefix due to router registration strategy.

# Testing

Run unit tests:

```bash
pytest -q
```

Current test suite focuses on service-level and route-registration behavior. There is no full end-to-end integration suite using real Postgres/Redis/provider APIs in this repository.

# Architecture Summary

`ragprobe` is a modular FastAPI backend with a service-layer architecture:
- API layer orchestrates requests and response schemas.
- Service layer executes embeddings, retrieval, judging, scoring, ranking, and routing.
- PostgreSQL/pgvector stores all domain data and supports vector similarity search.
- Redis serves both embedding cache and Celery transport/backends.
- Celery workers execute evaluation runs asynchronously and persist results.

# Limitations / Current Gaps

Current repository limitations and assumptions:
- No authentication/authorization is implemented.
- No `.env.example` is provided.
- Evaluation worker currently evaluates only the first QA pair in a dataset (`limit(1)`), not all QA pairs.
- Benchmark/routing flows assume model metadata exists and is compatible with pipeline configs (for example `config.model_id`).
- API route versioning is inconsistent: router is mounted at `/` and `/api/v1`, while some route paths already include `/api/v1`.
- Observability is minimal (no built-in metrics/tracing stack).

# Contributing

Contributions are welcome. A practical baseline workflow:
1. Create a branch.
2. Make changes with tests.
3. Run `pytest -q`.
4. Open a pull request with a clear description of behavior changes and validation steps.

# License

This project is licensed under the MIT License. See `LICENSE`.
