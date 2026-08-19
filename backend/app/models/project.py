"""
Модели проекта, статуса пайплайна и анкеты пользователя (пожелания к дизайну).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    uploaded = "uploaded"
    analyzing_floorplan = "analyzing_floorplan"
    detecting_rooms = "detecting_rooms"
    awaiting_architect_decision = "awaiting_architect_decision"
    designing_interior = "designing_interior"
    planning_furniture = "planning_furniture"
    designing_lighting = "designing_lighting"
    adding_decor = "adding_decor"
    generating_scene = "generating_scene"
    ready = "ready"
    failed = "failed"


class UserPreferences(BaseModel):
    style: str | None = Field(default=None, description='например "Modern Minimalism"')
    budget_usd: float | None = None
    adults: int = 2
    children: int = 0
    pets: list[str] = Field(default_factory=list)
    favorite_colors: list[str] = Field(default_factory=list)
    needs_office: bool = False
    likes_hosting_guests: bool = False


class Project(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    original_plan_key: str | None = Field(default=None, description="Ключ файла в S3/MinIO")
    stage: PipelineStage = PipelineStage.uploaded
    error: str | None = None
    preferences: UserPreferences | None = None
    active_variant_id: str = "variant_a"


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    id: str
    project_id: str
    role: ChatRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    applied_patch: dict | None = Field(
        default=None, description="Если сообщение вызвало изменение сцены — что именно изменилось"
    )


class AgentRun(BaseModel):
    id: str
    project_id: str
    agent_name: str
    status: str = Field(description="pending | running | done | failed")
    input_summary: dict | None = None
    output_summary: dict | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
