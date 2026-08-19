"""
Agent 2 — Room Detector.

Классифицирует каждую комнату (RoomType) на основе геометрии: площади,
пропорций, количества дверей/окон, соседства с другими комнатами.
Использует LLM (Groq), т.к. эвристики "самая маленькая комната с одной дверью
= санузел" работают лишь частично — LLM лучше учитывает совокупность признаков.
"""
from __future__ import annotations

from app.core.llm import complete_json
from app.models.scene import Room, RoomType

SYSTEM_PROMPT = """Ты — архитектор-аналитик. Тебе дан список комнат квартиры/дома
с их геометрией (площадь в кв.м, количество дверей, количество окон).
Определи наиболее вероятный тип каждой комнаты.
Верни JSON вида {"room_types": {"<room_id>": "<type>"}},
где <type> — одно из: kitchen, living_room, bedroom, bathroom, hallway,
office, dining_room, kids_room, unknown."""


def _room_area(room: Room) -> float:
    points = room.polygon
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


async def run(rooms: list[Room]) -> list[Room]:
    if not rooms:
        return rooms

    summary = [
        {
            "id": room.id,
            "area_sqm": round(_room_area(room), 1),
            "doors": sum(1 for w in room.walls for o in w.openings if o.type == "door"),
            "windows": sum(1 for w in room.walls for o in w.openings if o.type == "window"),
        }
        for room in rooms
    ]

    result = await complete_json(SYSTEM_PROMPT, f"Комнаты: {summary}")
    room_types = result.get("room_types", {})

    for room in rooms:
        type_value = room_types.get(room.id)
        if type_value in RoomType.__members__.values():
            room.type = RoomType(type_value)

    return rooms
