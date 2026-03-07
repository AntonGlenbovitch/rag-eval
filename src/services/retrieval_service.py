import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True)
class RetrievalResult:
    id: uuid.UUID
    dataset_id: uuid.UUID
    content: str
    similarity: float
    metadata: dict | None


class RetrievalService:
    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    @staticmethod
    def _to_vector_literal(query_embedding: list[float]) -> str:
        return "[" + ",".join(str(value) for value in query_embedding) + "]"

    async def search(
        self,
        dataset_id: uuid.UUID,
        query_embedding: list[float],
        k: int,
        similarity_threshold: float,
    ) -> list[RetrievalResult]:
        if not query_embedding or k <= 0:
            return []

        query_vector = self._to_vector_literal(query_embedding)

        result = await self._db_session.execute(
            text(
                """
                SELECT
                    id,
                    dataset_id,
                    content,
                    metadata,
                    1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity
                FROM knowledge_chunks
                WHERE dataset_id = :dataset_id
                  AND 1 - (embedding <=> CAST(:query_vector AS vector)) >= :similarity_threshold
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :k
                """
            ),
            {
                "dataset_id": dataset_id,
                "query_vector": query_vector,
                "k": k,
                "similarity_threshold": similarity_threshold,
            },
        )

        return [
            RetrievalResult(
                id=row["id"],
                dataset_id=row["dataset_id"],
                content=row["content"],
                similarity=float(row["similarity"]),
                metadata=row["metadata"],
            )
            for row in result.mappings().all()
        ]
