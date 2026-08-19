from fastapi import APIRouter

from app.core import database, redis_client

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health():
    mongo_ok = await database.ping()
    redis_ok = await redis_client.ping()
    return {
        "status": "ok" if mongo_ok and redis_ok else "degraded",
        "mongo": mongo_ok,
        "redis": redis_ok,
    }
