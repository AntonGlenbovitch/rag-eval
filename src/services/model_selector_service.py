from __future__ import annotations

import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.model_ranking import ModelRanking


class ModelSelectorService:
    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def select_best_model(self, dataset_id: uuid.UUID) -> uuid.UUID | None:
        result = await self._db_session.execute(
            select(ModelRanking.model_id)
            .where(ModelRanking.dataset_id == dataset_id)
            .order_by(desc(ModelRanking.weighted_score), ModelRanking.rank)
            .limit(1)
        )
        return result.scalar_one_or_none()
