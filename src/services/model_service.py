import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.model import Model


class ModelService:
    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def register_model(
        self,
        *,
        name: str,
        provider: str,
        context_window: int,
        cost_per_1k_tokens: float,
    ) -> Model:
        model = Model(
            name=name,
            provider=provider,
            context_window=context_window,
            cost_per_1k_tokens=cost_per_1k_tokens,
        )
        self._db_session.add(model)
        await self._db_session.commit()
        await self._db_session.refresh(model)
        return model

    async def get_model(self, model_id: uuid.UUID) -> Model | None:
        return await self._db_session.get(Model, model_id)

    async def list_models(self, *, provider: str | None = None) -> list[Model]:
        query = select(Model)
        if provider is not None:
            query = query.where(Model.provider == provider)

        query = query.order_by(Model.created_at.desc())
        result = await self._db_session.execute(query)
        return list(result.scalars().all())

    async def delete_model(self, model_id: uuid.UUID) -> bool:
        model = await self.get_model(model_id)
        if model is None:
            return False

        await self._db_session.delete(model)
        await self._db_session.commit()
        return True
