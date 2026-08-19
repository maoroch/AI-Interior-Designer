"""
Agent 7 — Decorator.

Добавляет финальные штрихи: растения, ковры, картины — на основе стиля
и пожеланий (например, наличие кота — учитывать в выборе растений/тканей
в дальнейшем, MVP пока просто добавляет базовый набор декора на комнату).
"""
from __future__ import annotations

import uuid

from app.models.scene import DecorItem, FurnitureItem, Room


def _room_center(room: Room) -> tuple[float, float]:
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


async def run(rooms: list[Room], furniture: list[FurnitureItem]) -> list[DecorItem]:
    decor: list[DecorItem] = []

    for room in rooms:
        cx, cy = _room_center(room)
        has_sofa_or_bed = any(
            f.room_id == room.id and f.type in ("sofa", "bed") for f in furniture
        )
        if has_sofa_or_bed:
            decor.append(
                DecorItem(
                    id=f"decor_{uuid.uuid4().hex[:8]}",
                    type="rug",
                    room_id=room.id,
                    position=(cx, 0.01, cy),
                )
            )
            decor.append(
                DecorItem(
                    id=f"decor_{uuid.uuid4().hex[:8]}",
                    type="plant",
                    room_id=room.id,
                    position=(cx + 0.5, 0.0, cy + 0.5),
                )
            )

    return decor
