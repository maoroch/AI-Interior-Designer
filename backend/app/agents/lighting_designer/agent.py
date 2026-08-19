"""
Agent 6 — Lighting Designer.

MVP-версия: одна потолочная точка света по центру каждой комнаты + точечные
светильники у функциональных зон (например, над обеденным столом), плюс
"естественный" свет учитывается через цветовую температуру у комнат с окнами.
Полноценный расчёт освещённости (люксы, IES-профили) — вне рамок MVP.
"""
from __future__ import annotations

import uuid

from app.models.scene import FurnitureItem, LightSource, Room


def _room_center(room: Room) -> tuple[float, float]:
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _has_window(room: Room) -> bool:
    return any(o.type == "window" for w in room.walls for o in w.openings)


async def run(rooms: list[Room], furniture: list[FurnitureItem]) -> list[LightSource]:
    lights: list[LightSource] = []

    for room in rooms:
        cx, cy = _room_center(room)
        warmth = 4000 if _has_window(room) else 3000
        lights.append(
            LightSource(
                id=f"light_{uuid.uuid4().hex[:8]}",
                type="ceiling",
                position=(cx, room.height - 0.1, cy),
                color_temperature_k=warmth,
                intensity=0.9,
            )
        )

    # Точечный свет над обеденным столом, если такой есть
    for item in furniture:
        if item.type == "dining_table":
            x, _, z = item.position
            lights.append(
                LightSource(
                    id=f"light_{uuid.uuid4().hex[:8]}",
                    type="pendant",
                    position=(x, 2.2, z),
                    color_temperature_k=2700,
                    intensity=0.7,
                )
            )

    return lights
