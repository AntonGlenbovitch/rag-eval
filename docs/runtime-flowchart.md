# Runtime Flowchart

This document provides a focused runtime flowchart for request handling and background evaluation execution.

```mermaid
flowchart TD
    Start[Client Request] --> APIServer[FastAPI: src.main app]

    APIServer --> Health{Health endpoint?}
    Health -- Yes --> DBPing[SELECT 1]
    DBPing --> HealthResp[Return status ok]

    Health -- No --> EvalCreate{Create evaluation run?}
    EvalCreate -- Yes --> PersistRun[Persist pipeline config + pending run]
    PersistRun --> CreatedResp[Return run id]

    EvalCreate -- No --> ExecuteRun{Execute run endpoint?}
    ExecuteRun -- Yes --> Enqueue[Queue Celery task tasks.run_evaluation]
    Enqueue --> Accepted[Return queued response]

    ExecuteRun -- No --> OtherRoutes[Other dataset/analysis/ranking/routing endpoints]
    OtherRoutes --> SyncResp[Return response]

    Enqueue --> Worker[Celery worker: src.tasks.jobs]
    Worker --> LoadRun[Load EvaluationRun by id]
    LoadRun --> MarkRunning[Set status=running]
    MarkRunning --> LoadQA[Load first QA pair for dataset]
    LoadQA --> BuildSvc[Build EvaluationService + dependencies]
    BuildSvc --> Embed[EmbeddingService embed_text]
    Embed --> Retrieve[RetrievalService vector search]
    Retrieve --> Metrics[Compute retrieval metrics]
    Metrics --> Judge[ClaudeJudgeService evaluate]
    Judge --> Score[Aggregate score]
    Score --> PersistDone[Persist run completed + experiment]
    PersistDone --> DoneResp[Task returns result payload]

    BuildSvc --> Error{Any exception?}
    Error -- Yes --> PersistFail[Set status=failed + metrics.error]
    PersistFail --> FailResp[Task raises error]
```

## Notes

- The API and worker are decoupled through Redis-backed Celery queues.
- Evaluation execution is asynchronous and status-driven (`pending` → `running` → `completed`/`failed`).
- Query embedding, retrieval, and judge scoring are the core stages that feed the final run score.
