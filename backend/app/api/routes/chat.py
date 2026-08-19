"""AI-чат: пользователь отправляет реплику, Conversation Agent патчит сцену
активного варианта дизайна (Project.active_variant_id)."""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.database import chat_messages_collection, projects_collection
from app.models.project import ChatMessage, ChatRole
from app.tasks.pipeline_tasks import run_conversation_turn

router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


class ChatIn(BaseModel):
    message: str


@router.post("")
async def send_message(project_id: str, body: ChatIn):
    project_doc = await projects_collection().find_one({"id": project_id})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    variant_id = project_doc.get("active_variant_id", "variant_a")

    user_msg = ChatMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}", project_id=project_id, role=ChatRole.user, content=body.message
    )
    await chat_messages_collection().insert_one(user_msg.model_dump())

    task = await run_conversation_turn.kiq(project_id=project_id, variant_id=variant_id, user_message=body.message)
    result = await task.wait_result(timeout=30)
    payload = result.return_value or {}

    if payload.get("error"):
        content = payload["error"]
    elif payload.get("operations"):
        content = "Изменения применены."
    else:
        content = "Не удалось распознать изменение — попробуйте переформулировать."

    assistant_msg = ChatMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        role=ChatRole.assistant,
        content=content,
        created_at=datetime.utcnow(),
        applied_patch=payload,
    )
    await chat_messages_collection().insert_one(assistant_msg.model_dump())

    return assistant_msg.model_dump()


@router.get("")
async def get_history(project_id: str):
    cursor = chat_messages_collection().find({"project_id": project_id}, {"_id": 0}).sort("created_at", 1)
    return [doc async for doc in cursor]
