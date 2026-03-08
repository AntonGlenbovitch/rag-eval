import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.models.evaluation import Dataset, EvaluationRun, KnowledgeChunk, PipelineConfig, PipelineExperiment, QAPair
from src.models.model import Model
from src.models.model_ranking import ModelRanking
from src.schemas.health import HealthResponse
from src.services.claude_judge_service import ClaudeJudgeService
from src.services.embedding_service import EmbeddingService
from src.services.evaluation_service import EvaluationService
from src.services.model_benchmark_service import ModelBenchmarkService
from src.services.model_ranking_service import ModelRankingService
from src.services.retrieval_service import RetrievalService
from src.services.optimization_service import OptimizationService
from src.tasks.jobs import run_evaluation

router = APIRouter()


class DatasetCreateRequest(BaseModel):
    name: str
    description: str | None = None
    metadata: dict[str, Any] | None = None


class DatasetResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    metadata: dict[str, Any] | None


class QAPairCreateRequest(BaseModel):
    question: str
    answer: str
    metadata: dict[str, Any] | None = None


class KnowledgeChunkCreateRequest(BaseModel):
    content: str
    embedding: list[float] = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class EmbeddingsRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbeddingsResponse(BaseModel):
    embeddings: list[list[float]]


class EvaluationRunCreateRequest(BaseModel):
    dataset_id: uuid.UUID
    provider: str
    pipeline_name: str
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    pipeline_config_id: uuid.UUID
    status: str
    metrics: dict[str, Any] | None


class OptimizationCandidatesRequest(BaseModel):
    base_config: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)


class ScheduleExperimentsRequest(BaseModel):
    dataset_id: uuid.UUID
    provider: str
    pipeline_candidates: list[dict[str, Any]] = Field(min_length=1)


class ComparePipelinesRequest(BaseModel):
    metric_weights: dict[str, float] | None = None


class FailureClusterResponse(BaseModel):
    label: str
    size: int
    queries: list[str]
    run_ids: list[uuid.UUID]


class AnalysisResponse(BaseModel):
    total_runs: int
    completed_runs: int
    failed_runs: int
    average_score: float
    best_run_id: uuid.UUID | None
    best_score: float | None
    failure_clusters: list[FailureClusterResponse]


class PipelineComparisonResponse(BaseModel):
    pipeline_config_id: uuid.UUID
    evaluation_run_id: uuid.UUID | None
    score: float
    metrics: dict[str, float]


class BenchmarkRequest(BaseModel):
    model_ids: list[uuid.UUID] | None = None


class ModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    context_window: int
    cost_per_1k_tokens: float


class ModelRankingResponse(BaseModel):
    model_id: uuid.UUID
    model_name: str
    provider: str
    weighted_score: float
    rank: int


# This lightweight dependency makes EmbeddingService overridable in tests

def get_embedding_service() -> EmbeddingService:
    from openai import OpenAI
    from redis import Redis

    from src.core.config import settings

    return EmbeddingService(
        openai_client=OpenAI(api_key=settings.openai_api_key),
        redis_client=Redis.from_url(settings.redis_url),
        model=settings.openai_embedding_model,
    )


def get_evaluation_service(db: AsyncSession) -> EvaluationService:
    from anthropic import AsyncAnthropic
    from openai import OpenAI
    from redis import Redis

    from src.core.config import settings

    embedding_service = EmbeddingService(
        openai_client=OpenAI(api_key=settings.openai_api_key),
        redis_client=Redis.from_url(settings.redis_url),
        model=settings.openai_embedding_model,
    )
    retrieval_service = RetrievalService(db_session=db)
    judge_service = ClaudeJudgeService(
        anthropic_client=AsyncAnthropic(api_key=settings.anthropic_api_key),
        model=settings.anthropic_model,
    )
    return EvaluationService(
        db_session=db,
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        judge_service=judge_service,
    )


@router.get("/health", response_model=HealthResponse)
async def healthcheck(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))
    return HealthResponse(status="ok")


@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(payload: DatasetCreateRequest, db: AsyncSession = Depends(get_db)):
    dataset = Dataset(name=payload.name, description=payload.description, extra_metadata=payload.metadata)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        metadata=dataset.extra_metadata,
    )


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            metadata=dataset.extra_metadata,
        )
        for dataset in datasets
    ]


@router.post("/datasets/{dataset_id}/qa-pairs")
async def create_qa_pair(dataset_id: uuid.UUID, payload: QAPairCreateRequest, db: AsyncSession = Depends(get_db)):
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    qa_pair = QAPair(
        dataset_id=dataset_id,
        question=payload.question,
        answer=payload.answer,
        extra_metadata=payload.metadata,
    )
    db.add(qa_pair)
    await db.commit()
    return {"id": str(qa_pair.id)}


@router.post("/datasets/{dataset_id}/chunks")
async def create_knowledge_chunk(
    dataset_id: uuid.UUID,
    payload: KnowledgeChunkCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    chunk = KnowledgeChunk(
        dataset_id=dataset_id,
        content=payload.content,
        embedding=payload.embedding,
        extra_metadata=payload.metadata,
    )
    db.add(chunk)
    await db.commit()
    return {"id": str(chunk.id)}


@router.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(payload: EmbeddingsRequest, embedding_service: EmbeddingService = Depends(get_embedding_service)):
    return EmbeddingsResponse(embeddings=embedding_service.embed_batch(payload.texts))


@router.post("/evaluation/runs", response_model=EvaluationRunResponse)
async def create_evaluation_run(payload: EvaluationRunCreateRequest, db: AsyncSession = Depends(get_db)):
    dataset = await db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = PipelineConfig(name=payload.pipeline_name, provider=payload.provider, config=payload.config)
    db.add(config)
    await db.flush()

    run = EvaluationRun(dataset_id=payload.dataset_id, pipeline_config_id=config.id, status="pending")
    db.add(run)
    await db.commit()
    await db.refresh(run)

    return EvaluationRunResponse(
        id=run.id,
        dataset_id=run.dataset_id,
        pipeline_config_id=run.pipeline_config_id,
        status=run.status,
        metrics=run.metrics,
    )


@router.get("/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    return EvaluationRunResponse(
        id=run.id,
        dataset_id=run.dataset_id,
        pipeline_config_id=run.pipeline_config_id,
        status=run.status,
        metrics=run.metrics,
    )


@router.post("/evaluation/runs/{run_id}/execute")
async def execute_evaluation_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    run_evaluation.delay(str(run_id))
    return {"status": "queued", "run_id": str(run_id)}


@router.get("/analysis", response_model=AnalysisResponse)
async def analyze_runs(
    dataset_id: uuid.UUID | None = Query(default=None),
    pipeline_config_id: uuid.UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    service = OptimizationService(db_session=db)
    analysis = await service.analyze_evaluation_runs(dataset_id=dataset_id, pipeline_config_id=pipeline_config_id)
    return AnalysisResponse(
        total_runs=analysis.total_runs,
        completed_runs=analysis.completed_runs,
        failed_runs=analysis.failed_runs,
        average_score=analysis.average_score,
        best_run_id=analysis.best_run_id,
        best_score=analysis.best_score,
        failure_clusters=[
            FailureClusterResponse(
                label=cluster.label,
                size=cluster.size,
                queries=cluster.queries,
                run_ids=cluster.run_ids,
            )
            for cluster in analysis.failure_clusters
        ],
    )


@router.post("/optimization/candidates")
async def generate_candidates(payload: OptimizationCandidatesRequest):
    candidates = OptimizationService.generate_pipeline_candidates(
        base_config=payload.base_config,
        search_space=payload.search_space,
    )
    return {"candidates": candidates}


@router.post("/optimization/experiments/schedule", response_model=list[EvaluationRunResponse])
async def schedule_experiments(payload: ScheduleExperimentsRequest, db: AsyncSession = Depends(get_db)):
    service = OptimizationService(
        db_session=db,
        enqueue_evaluation_run=lambda run_id: run_evaluation.delay(run_id),
    )
    runs = await service.schedule_evaluation_experiments(
        dataset_id=payload.dataset_id,
        provider=payload.provider,
        pipeline_candidates=payload.pipeline_candidates,
    )

    return [
        EvaluationRunResponse(
            id=run.id,
            dataset_id=run.dataset_id,
            pipeline_config_id=run.pipeline_config_id,
            status=run.status,
            metrics=run.metrics,
        )
        for run in runs
    ]


@router.post("/reports/pipelines", response_model=PipelineComparisonResponse | None)
async def pipeline_report(payload: ComparePipelinesRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PipelineExperiment))
    experiments = list(result.scalars().all())

    service = OptimizationService(db_session=db)
    best = service.select_best_pipeline(experiments=experiments, metric_weights=payload.metric_weights)
    if best is None:
        return None

    return PipelineComparisonResponse(
        pipeline_config_id=best.pipeline_config_id,
        evaluation_run_id=best.evaluation_run_id,
        score=best.score,
        metrics=best.metrics,
    )


@router.get("/clusters", response_model=list[FailureClusterResponse])
async def failure_clusters(
    dataset_id: uuid.UUID | None = Query(default=None),
    pipeline_config_id: uuid.UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    service = OptimizationService(db_session=db)
    analysis = await service.analyze_evaluation_runs(dataset_id=dataset_id, pipeline_config_id=pipeline_config_id)
    return [
        FailureClusterResponse(
            label=cluster.label,
            size=cluster.size,
            queries=cluster.queries,
            run_ids=cluster.run_ids,
        )
        for cluster in analysis.failure_clusters
    ]


@router.post("/api/v1/benchmark/{dataset_id}")
async def run_model_benchmark(
    dataset_id: uuid.UUID,
    payload: BenchmarkRequest,
    db: AsyncSession = Depends(get_db),
):
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    selected_models = payload.model_ids
    if selected_models is None:
        model_result = await db.execute(select(Model.id))
        selected_models = [model_id for model_id in model_result.scalars().all()]

    if not selected_models:
        raise HTTPException(status_code=400, detail="No models available for benchmarking")

    benchmark_service = ModelBenchmarkService(db_session=db, evaluation_service=get_evaluation_service(db))
    benchmark_result = await benchmark_service.benchmark_dataset(dataset_id=dataset_id, model_ids=selected_models)

    rankings = await ModelRankingService(db_session=db).rank_models(dataset_id)
    return {
        **benchmark_result,
        "rankings": [
            {
                "model_id": str(ranking.model_id),
                "model_name": ranking.model_name,
                "provider": ranking.provider,
                "weighted_score": ranking.weighted_score,
                "rank": ranking.rank,
            }
            for ranking in rankings
        ],
    }


@router.get("/api/v1/benchmark/{dataset_id}")
async def get_model_benchmark(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ranking_rows = await db.execute(
        select(ModelRanking, Model)
        .join(Model, Model.id == ModelRanking.model_id)
        .where(ModelRanking.dataset_id == dataset_id)
        .order_by(ModelRanking.rank.asc())
    )
    rows = ranking_rows.all()

    rankings = [
        {
            "model_id": str(ranking.model_id),
            "model_name": model.name,
            "provider": model.provider,
            "weighted_score": ranking.weighted_score,
            "rank": ranking.rank,
        }
        for ranking, model in rows
    ]
    average_score = sum(item["weighted_score"] for item in rankings) / len(rankings) if rankings else 0.0

    return {
        "dataset_id": str(dataset_id),
        "average_score": average_score,
        "rankings": rankings,
    }


@router.get("/api/v1/models", response_model=list[ModelResponse])
async def list_available_models(
    provider: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    query = select(Model)
    if provider:
        query = query.where(Model.provider == provider)

    result = await db.execute(query.order_by(Model.created_at.desc()))
    models = result.scalars().all()
    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            provider=model.provider,
            context_window=model.context_window,
            cost_per_1k_tokens=model.cost_per_1k_tokens,
        )
        for model in models
    ]


@router.get("/api/v1/models/ranking/{dataset_id}", response_model=list[ModelRankingResponse])
async def get_model_ranking(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    rankings = await ModelRankingService(db_session=db).rank_models(dataset_id)
    return [
        ModelRankingResponse(
            model_id=ranking.model_id,
            model_name=ranking.model_name,
            provider=ranking.provider,
            weighted_score=ranking.weighted_score,
            rank=ranking.rank,
        )
        for ranking in rankings
    ]
