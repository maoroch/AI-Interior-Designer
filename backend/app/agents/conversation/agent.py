"""
Agent 9 — Conversation Agent.

В отличие от остальных агентов — не часть линейного пайплайна, а работает
поверх уже готовой сцены. Получает реплику пользователя из AI-чата
("Замени диван", "Сделай интерьер светлее", "Добавь камин") и превращает
её в набор точечных операций над Scene, которые applies() применяет
без пересчёта всего пайплайна.

Формат операции (упрощённый JSON Patch, специфичный под нашу схему):
    {"op": "add" | "update" | "remove",
     "collection": "furniture" | "decor" | "lighting" | "rooms",
     "target_id": str | null,   # null для "add"
     "fields": {...}}           # для add/update — новые значения полей
"""
from __future__ import annotations

import uuid

from app.core.llm import complete_json
from app.models.scene import DecorItem, FurnitureItem, LightSource, Scene

SYSTEM_PROMPT = """Ты — ассистент дизайнера интерьера, управляющий 3D-сценой через чат.
Тебе дана текущая сцена (упрощённо: список мебели, декора, света по комнатам)
и реплика пользователя. Преобразуй её в список операций.
Разрешённые collection: furniture, decor, lighting.
Разрешённые op: add, update, remove.
Для "remove" и "update" ОБЯЗАТЕЛЬНО указывай существующий target_id из списка ниже.
Для "add" указывай fields с разумными значениями (type, dimensions, material, position и т.д.)
Если реплика касается "сделай светлее/темнее" — сгенерируй update для всех
объектов lighting (intensity) и/или update для rooms (wall_color).
Верни JSON: {"operations": [{"op": str, "collection": str, "target_id": str|null, "fields": dict}]}"""


def _scene_summary(scene: Scene) -> dict:
    return {
        "furniture": [
            {"id": f.id, "type": f.type, "room_id": f.room_id, "material": f.material}
            for f in scene.furniture
        ],
        "decor": [{"id": d.id, "type": d.type, "room_id": d.room_id} for d in scene.decor],
        "lighting": [{"id": l.id, "type": l.type, "intensity": l.intensity} for l in scene.lighting],
    }


async def interpret(scene: Scene, user_message: str) -> list[dict]:
    result = await complete_json(
        SYSTEM_PROMPT,
        f"Текущая сцена: {_scene_summary(scene)}\nРеплика пользователя: {user_message}",
    )
    return result.get("operations", [])


def apply_operations(scene: Scene, operations: list[dict]) -> Scene:
    for op in operations:
        collection = op.get("collection")
        action = op.get("op")
        target_id = op.get("target_id")
        fields = op.get("fields", {})

        if collection == "furniture":
            _apply_to_list(scene.furniture, action, target_id, fields, FurnitureItem)
        elif collection == "decor":
            _apply_to_list(scene.decor, action, target_id, fields, DecorItem)
        elif collection == "lighting":
            _apply_to_list(scene.lighting, action, target_id, fields, LightSource)

    scene.version += 1
    return scene


def _apply_to_list(items: list, action: str, target_id: str | None, fields: dict, model_cls) -> None:
    if action == "remove" and target_id:
        items[:] = [i for i in items if i.id != target_id]
        return

    if action == "update" and target_id:
        for item in items:
            if item.id == target_id:
                for key, value in fields.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
        return

    if action == "add":
        fields = {**fields, "id": fields.get("id", f"gen_{uuid.uuid4().hex[:8]}")}
        try:
            items.append(model_cls(**fields))
        except Exception:
            pass  # некорректный набор полей от LLM — молча игнорируем, чат сообщит об этом пользователю
