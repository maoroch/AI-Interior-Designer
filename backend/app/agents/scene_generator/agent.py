"""
Agent 8 — Scene Generator.

Не вызывает LLM — чисто сборочный агент. Берёт результаты всех предыдущих
агентов и собирает единый объект Scene (см. app/models/scene.py), готовый
к сохранению в Mongo (`scenes`) и рендеру на фронтенде в Three.js.
"""
from __future__ import annotations

from app.models.scene import (
    ArchitectSuggestion,
    DecorItem,
    FurnitureItem,
    LightSource,
    Room,
    Scene,
)


def assemble(
    project_id: str,
    version: int,
    rooms: list[Room],
    furniture: list[FurnitureItem],
    lighting: list[LightSource],
    decor: list[DecorItem],
    architect_suggestions: list[ArchitectSuggestion] | None = None,
    architect_choice: str | None = None,
    style: str | None = None,
) -> Scene:
    return Scene(
        project_id=project_id,
        version=version,
        rooms=rooms,
        furniture=furniture,
        lighting=lighting,
        decor=decor,
        architect_suggestions=architect_suggestions or [],
        architect_choice=architect_choice,
        style=style,
    )
