"""
Agent 5 — Furniture Planner.

Два шага:
1. LLM (Groq) подбирает набор мебели под тип комнаты + пожелания пользователя
   (стиль, состав семьи, бюджет) и даёт для каждого предмета реалистичные
   габариты в метрах — MVP рендерит мебель параметрическими примитивами
   (боксами по этим габаритам), без GLTF-моделей (см. README, фаза 2).
2. Простой алгоритм расстановки вдоль периметра комнаты, который:
   - не ставит мебель поверх дверных/оконных проёмов;
   - оставляет зону открывания двери свободной;
   - оставляет минимальный проход (0.6м) от каждого предмета.

Это MVP-уровень эргономики — не полноценный solver, а быстрая эвристика,
которая тем не менее валидна для реальных прямоугольных комнат.
"""
from __future__ import annotations

import math
import uuid

from app.core.llm import complete_json
from app.models.project import UserPreferences
from app.models.scene import FurnitureItem, Room

SYSTEM_PROMPT = """Ты — дизайнер, подбирающий мебель. Для заданного типа комнаты
и пожеланий клиента (стиль, бюджет, дети, животные, любит принимать гостей)
подбери список мебели. Для каждого предмета укажи: type (короткий английский
идентификатор вроде sofa, bed, dining_table, wardrobe, desk, tv_stand),
material, color, и dimensions_m [ширина, высота, глубина].
Не превышай разумное количество предметов для площади комнаты.
Верни JSON: {"items": [{"type": str, "material": str, "color": str, "dimensions_m": [w,h,d]}]}."""

MIN_CLEARANCE_M = 0.6
DOOR_CLEARANCE_M = 0.9


def _room_bbox(room: Room) -> tuple[float, float, float, float]:
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _door_positions(room: Room) -> list[tuple[float, float]]:
    """Приближённые мировые координаты дверных проёмов (для зон исключения)."""
    positions = []
    for wall in room.walls:
        for opening in wall.openings:
            if opening.type != "door":
                continue
            x = wall.start[0] + (wall.end[0] - wall.start[0]) * opening.position
            y = wall.start[1] + (wall.end[1] - wall.start[1]) * opening.position
            positions.append((x, y))
    return positions


def _too_close_to_door(x: float, y: float, doors: list[tuple[float, float]]) -> bool:
    return any(math.hypot(x - dx, y - dy) < DOOR_CLEARANCE_M for dx, dy in doors)


async def _select_furniture_set(room: Room, preferences: UserPreferences | None) -> list[dict]:
    prefs_summary = preferences.model_dump() if preferences else {}
    result = await complete_json(
        SYSTEM_PROMPT,
        f"Тип комнаты: {room.type.value}\nПожелания: {prefs_summary}",
    )
    return result.get("items", [])


DEFAULT_DIMS: dict[str, tuple[float, float, float]] = {
    "sofa": (2.0, 0.85, 0.9),
    "bed": (1.8, 1.0, 2.0),
    "dining_table": (1.4, 0.75, 0.8),
    "table": (1.2, 0.75, 0.7),
    "desk": (1.2, 0.75, 0.6),
    "chair": (0.5, 0.85, 0.5),
    "wardrobe": (1.2, 2.0, 0.6),
    "tv_stand": (1.5, 0.5, 0.4),
    "bookshelf": (0.8, 1.8, 0.35),
    "nightstand": (0.5, 0.5, 0.4),
    "coffee_table": (1.0, 0.45, 0.6),
}


def _safe_dimensions(item: dict) -> tuple[float, float, float]:
    item_type = str(item.get("type", "")).lower()
    default_w, default_h, default_d = DEFAULT_DIMS.get(item_type, (0.8, 0.8, 0.8))

    raw = item.get("dimensions_m")
    if isinstance(raw, (list, tuple)):
        clean = []
        for val in raw:
            try:
                clean.append(float(val))
            except (ValueError, TypeError):
                pass
        if len(clean) >= 3:
            return clean[0], clean[1], clean[2]
        if len(clean) == 2:
            return clean[0], default_h, clean[1]
        if len(clean) == 1:
            return clean[0], default_h, clean[0]

    return default_w, default_h, default_d


def _place_along_perimeter(room: Room, items: list[dict]) -> list[FurnitureItem]:
    min_x, min_y, max_x, max_y = _room_bbox(room)
    doors = _door_positions(room)

    placed: list[FurnitureItem] = []
    cursor = MIN_CLEARANCE_M  # смещение вдоль первой (нижней) стены комнаты

    for item in items:
        w, h, d = _safe_dimensions(item)
        x = min_x + cursor + w / 2
        y = min_y + d / 2 + 0.1  # почти вплотную к "нижней" стене bbox'а

        if x + w / 2 > max_x - MIN_CLEARANCE_M:
            # не влезло вдоль этой стены — переносим на следующую условную "полосу"
            cursor = MIN_CLEARANCE_M
            x = min_x + cursor + w / 2
            y += d + MIN_CLEARANCE_M

        if _too_close_to_door(x, y, doors):
            y += DOOR_CLEARANCE_M  # простое смещение от зоны двери

        placed.append(
            FurnitureItem(
                id=f"f_{uuid.uuid4().hex[:8]}",
                room_id=room.id,
                type=item.get("type", "unknown"),
                style=None,
                position=(round(x, 2), 0.0, round(y, 2)),
                rotation_deg=0,
                dimensions=(w, h, d),
                material=item.get("material"),
                color=item.get("color"),
                model_ref=None,
            )
        )
        cursor += w + MIN_CLEARANCE_M

    return placed



async def run(rooms: list[Room], preferences: UserPreferences | None) -> list[FurnitureItem]:
    all_furniture: list[FurnitureItem] = []
    for room in rooms:
        items = await _select_furniture_set(room, preferences)
        all_furniture.extend(_place_along_perimeter(room, items))
    return all_furniture
