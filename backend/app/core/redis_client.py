"""
Redis используется в двух ролях:
1. Брокер результатов/очередей для Taskiq (агентные пайплайны).
2. Pub/Sub канал, через который тяжёлые агенты публикуют статус прогресса,
   а WebSocket-роут в api/routes/ws.py транслирует его на фронтенд.
"""
import redis.asyncio as redis

from app.core.config import get_settings

settings = get_settings()

_redis: redis.Redis | None = None

PROJECT_STATUS_CHANNEL_PREFIX = "project-status:"


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(
            settings.redis_uri,
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
        )
    return _redis



def project_channel(project_id: str) -> str:
    return f"{PROJECT_STATUS_CHANNEL_PREFIX}{project_id}"


async def publish_status(project_id: str, payload: dict) -> None:
    import json

    await get_redis().publish(project_channel(project_id), json.dumps(payload))


async def ping() -> bool:
    try:
        return await get_redis().ping()
    except Exception:
        return False
