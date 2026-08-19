"""
WebSocket-канал статуса пайплайна проекта.
Фронтенд подключается на /ws/projects/{project_id} и получает push-события
("analyzing_floorplan" -> ... -> "ready") вместо поллинга GET /projects/{id}.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.redis_client import get_redis, project_channel

router = APIRouter(tags=["ws"])


@router.websocket("/ws/projects/{project_id}")
async def project_status_ws(websocket: WebSocket, project_id: str):
    await websocket.accept()
    pubsub = get_redis().pubsub()
    await pubsub.subscribe(project_channel(project_id))
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(project_channel(project_id))
