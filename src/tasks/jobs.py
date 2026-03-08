import uuid
from datetime import datetime, timezone
import asyncio
from typing import Any, Callable

from celery import Task  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.celery_app import celery_app
from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.models.evaluation import EvaluationRun, QAPair
from src.services.claude_judge_service import ClaudeJudgeService
from src.services.embedding_service import EmbeddingService
from src.services.evaluation_service import EvaluationService
from src.services.retrieval_service import RetrievalService


def _extract_relevant_chunk_ids(qa_pair: QAPair) -> list[uuid.UUID]:
    metadata = qa_pair.extra_metadata or {}
    raw_ids = metadata.get("relevant_chunk_ids", [])

    relevant_chunk_ids: list[uuid.UUID] = []
    for raw_id in raw_ids:
        try:
            relevant_chunk_ids.append(uuid.UUID(str(raw_id)))
        except (TypeError, ValueError):
            continue

    return relevant_chunk_ids


def _build_evaluation_service(db_session: AsyncSession) -> EvaluationService:
    from anthropic import AsyncAnthropic
    from openai import OpenAI
    from redis import Redis

    redis_client = Redis.from_url(settings.redis_url)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    embedding_service = EmbeddingService(
        openai_client=openai_client,
        redis_client=redis_client,
        model=settings.openai_embedding_model,
    )
    retrieval_service = RetrievalService(db_session=db_session)
    judge_service = ClaudeJudgeService(
        anthropic_client=anthropic_client,
        model=settings.anthropic_model,
    )

    return EvaluationService(
        db_session=db_session,
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        judge_service=judge_service,
    )


async def _run_evaluation_async(
    *,
    evaluation_run_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    service_factory: Callable[[AsyncSession], EvaluationService] = _build_evaluation_service,
) -> dict[str, Any]:
    async with session_factory() as db_session:
        run = await db_session.get(EvaluationRun, evaluation_run_id)
        if run is None:
            raise ValueError(f"EvaluationRun not found: {evaluation_run_id}")

        run.status = "running"
        await db_session.commit()

        qa_result = await db_session.execute(
            select(QAPair).where(QAPair.dataset_id == run.dataset_id).order_by(QAPair.created_at.asc()).limit(1)
        )
        qa_pair = qa_result.scalars().first()
        if qa_pair is None:
            raise ValueError(f"No QA pair found for dataset: {run.dataset_id}")

        service = service_factory(db_session)

        try:
            result = await service.evaluate(
                evaluation_run_id=run.id,
                dataset_id=run.dataset_id,
                pipeline_config_id=run.pipeline_config_id,
                question=qa_pair.question,
                answer=qa_pair.answer,
                relevant_chunk_ids=_extract_relevant_chunk_ids(qa_pair),
            )
            return result.to_dict()
        except Exception as exc:
            run.status = "failed"
            run.metrics = {"error": str(exc)}
            run.completed_at = datetime.now(timezone.utc)
            await db_session.commit()
            raise


@celery_app.task(name="tasks.ping")
def ping() -> str:
    return "pong"


@celery_app.task(name="tasks.run_evaluation", bind=True)
def run_evaluation(self: Task, evaluation_run_id: str) -> dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run_evaluation_async(evaluation_run_id=uuid.UUID(evaluation_run_id)))
    finally:
        loop.close()
        asyncio.set_event_loop(None)
