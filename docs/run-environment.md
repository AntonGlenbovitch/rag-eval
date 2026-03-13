# Run Environment

This document explains what infrastructure `ragprobe` needs, which processes run, and how environment variables wire everything together.

## 1) Runtime Components

The application runs as a multi-process backend:

- **FastAPI API server** (`uvicorn src.main:app`) for HTTP endpoints.
- **Celery worker** (`celery -A src.tasks.worker worker`) for asynchronous evaluation runs.
- **PostgreSQL + pgvector** for relational data and vector similarity search.
- **Redis** for Celery broker/result backend and embedding cache.

## 2) Network / Ports

Default local ports from `docker-compose.yml`:

- API: `8000`
- Postgres: `5432`
- Redis: `6379`

## 3) Required Environment Variables

From `src/core/config.py`, these variables define runtime behavior:

### Application
- `APP_NAME` (default `ragprobe`)
- `APP_ENV` (default `development`)
- `APP_HOST` (default `0.0.0.0`)
- `APP_PORT` (default `8000`)

### Database
- `DATABASE_URL` (SQLAlchemy async URL)
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`

### Redis / Celery
- `REDIS_URL`
- `REDIS_HOST`
- `REDIS_PORT`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`

### Providers
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`

## 4) Environment Profiles

### A. Docker Compose profile
Use service hostnames:

- Postgres hostname: `postgres`
- Redis hostname: `redis`

Example:

```dotenv
DATABASE_URL=postgresql+asyncpg://ragprobe:ragprobe@postgres:5432/ragprobe
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

### B. Native local profile
Use localhost hostnames:

```dotenv
DATABASE_URL=postgresql+asyncpg://ragprobe:ragprobe@localhost:5432/ragprobe
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

## 5) Process Startup Order

Recommended order:

1. Start Postgres + Redis.
2. Run migrations (`alembic upgrade head`).
3. Start API.
4. Start Celery worker.
5. Validate health endpoint (`GET /health`).

## 6) Operational Notes

- If API works but evaluation runs do not progress, check worker logs first.
- If retrieval fails, verify pgvector-enabled Postgres and embedding dimension consistency (1536).
- If embedding/judge steps fail, verify OpenAI/Anthropic keys.
