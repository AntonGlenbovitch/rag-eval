from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def healthcheck(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))
    return HealthResponse(status="ok")
