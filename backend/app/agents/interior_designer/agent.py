"""
Agent 4 — Interior Designer.

Выбирает стиль, палитру и материалы для каждой комнаты на основе анкеты
пользователя (UserPreferences). Не расставляет мебель (это Furniture Planner) —
только задаёт "язык" дизайна, которым дальше пользуются остальные агенты.
"""
from __future__ import annotations

from app.core.llm import complete_json
from app.models.project import UserPreferences
from app.models.scene import Room

SYSTEM_PROMPT = """Ты — дизайнер интерьера. На основе пожеланий клиента (стиль, бюджет,
состав семьи, домашние животные, любимые цвета) подбери для каждой комнаты
материал пола и цвет стен, сочетающиеся друг с другом по всей квартире.
Учитывай практичность (например, для кухни/санузла — влагостойкие материалы,
если есть кот — не самые маркие ткани и покрытия).
Верни JSON: {"rooms": {"<room_id>": {"floor_material": str, "wall_color": str}}}."""


async def run(rooms: list[Room], preferences: UserPreferences | None) -> list[Room]:
    if not rooms:
        return rooms

    prefs_summary = preferences.model_dump() if preferences else {}
    rooms_summary = [{"id": r.id, "type": r.type.value} for r in rooms]

    result = await complete_json(
        SYSTEM_PROMPT, f"Пожелания: {prefs_summary}\nКомнаты: {rooms_summary}"
    )
    room_styles = result.get("rooms", {})

    for room in rooms:
        style = room_styles.get(room.id, {})
        room.floor_material = style.get("floor_material", room.floor_material)
        room.wall_color = style.get("wall_color", room.wall_color)

    return rooms
