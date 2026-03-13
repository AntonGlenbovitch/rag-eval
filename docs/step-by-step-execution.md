# Step-by-Step Execution

This document describes exactly how `ragprobe` executes from startup through one evaluation run.

## 1) Bootstrapping

1. **Configuration load**
   - `src/core/config.py` loads settings from environment variables / `.env`.
2. **API app creation**
   - `src/main.py` creates FastAPI app and includes routes (root + `/api/v1`).
3. **Worker app creation**
   - `src/core/celery_app.py` configures Celery with Redis broker/result backend.

## 2) Bring the System Up

## Docker Compose

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
```

## Native local

```bash
alembic upgrade head
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
celery -A src.tasks.worker worker --loglevel=info
```

## 3) Data Preparation Flow

1. Create dataset: `POST /datasets`
2. Add QA pair(s): `POST /datasets/{dataset_id}/qa-pairs`
3. Add knowledge chunks with embeddings: `POST /datasets/{dataset_id}/chunks`

At this point the dataset is ready for execution.

## 4) Evaluation Run Creation

1. Create run metadata: `POST /evaluation/runs`
2. API persists:
   - pipeline config
   - evaluation run record (`pending`)

## 5) Evaluation Run Execution (Async)

1. Trigger execution: `POST /evaluation/runs/{run_id}/execute`
2. API enqueues Celery task: `tasks.run_evaluation`
3. Worker receives task and performs:
   - load run
   - mark status `running`
   - load first QA pair for dataset
   - build `EvaluationService`
   - execute evaluation pipeline

## 6) EvaluationService Internal Steps

1. Generate query embedding (`EmbeddingService`).
2. Retrieve top-k chunks from pgvector (`RetrievalService`).
3. Compute retrieval metrics (precision/recall/F1/MRR helpers).
4. Score answer/context quality (`ClaudeJudgeService`).
5. Compute aggregate score.
6. Persist completed run metrics and experiment record.

If an error occurs, worker marks the run as `failed` and stores error in `metrics.error`.

## 7) Result Inspection

- `GET /evaluation/runs/{run_id}` for run status/metrics.
- Optional: call analysis/ranking/routing endpoints for downstream insights.

## 8) Sanity Checks

- Health: `GET /health`
- API docs: `GET /docs`
- Worker logs: `docker compose logs -f celery` (or local Celery console)
