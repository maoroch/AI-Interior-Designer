"""
Agent 5 — Furniture Planner (Decoupled Semantic CAD Architecture).

1. Локальный Semantic Bridge формирует понятный человеку бриф помещения в метрах.
2. LLM (Groq) генерирует профессиональный план расстановки по правилу 5 слоёв интерьера:
   - Слой 1: Основной каркас (диван, кровать, шкаф, ТВ-тумба, обеденный/рабочий стол)
   - Слой 2: Компаньоны (журнальный столик, кресло, прикроватные тумбы)
   - Слой 3: Текстиль пола (ковёр rug для объединения зоны)
   - Слой 4: Вертикальные акценты и декор (высокие растения plant, торшеры floor_lamp, стеллажи bookshelf)
3. Детерминированный CAD-компилятор (compiler.py) рассчитывает точные 3D-координаты для Three.js.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.furniture_planner.compiler import (
    DEFAULT_DIMS,
    compile_semantic_layout_to_3d,
)
from app.agents.furniture_planner.math_engine import (
    ChromaticBalanceCalculator,
    GoldenRatioScaler,
    OccupancyBudgetOptimizer,
)
from app.core.llm import complete_json
from app.cv.semantic_bridge import generate_semantic_room_brief
from app.cv.wall_graph import RoomFace, WallEdge
from app.knowledge.fast_path import FastPathLayoutMatcher
from app.knowledge.vector_engine import VectorRuleRAG
from app.models.project import UserPreferences
from app.models.scene import FurnitureItem, Room

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — ведущий архитектор-дизайнер интерьеров премиум-класса и эксперт по эргономике.
Твоя задача — наполнить комнату полноценным, уютным, функциональным набором мебели, чтобы пространство не выглядело пустым.

Используй архитектурное правило 5 слоёв интерьера:
1. Основные фокусные якоря:
   - Гостиная/Студия: диван (sofa), ТВ-тумба (tv_stand), обеденная группа (dining_table) или рабочий стол (desk).
   - Спальня: двуспальная кровать (bed), платяной шкаф (wardrobe).
2. Эргономичные компаньоны:
   - Журнальный столик (coffee_table), кресло для отдыха (armchair), прикроватные тумбы (nightstand).
3. Напольный текстиль:
   - Большой ковёр (rug) в зоне отдыха или спальни (объединяет диван и столик, убирает пустоту пола).
4. Вертикали, освещение и декор:
   - Высокие комнатные растения в кашпо (plant), напольный торшер (floor_lamp), открытый стеллаж (bookshelf).

Правила размещения:
- Для комнат >= 18 кв.м создавай от 7 до 12 предметов с чётким зонированием (Lounge + Work/Dining).
- Для спален и комнат 10-18 кв.м создавай 5-8 предметов.
- Для прихожих создавай шкаф (wardrobe), банкетку/обувницу (bench) и зеркало (mirror).
- Крупные предметы привязывай к глухим стенам (solid) или под окна (under_window).
- Высокие шкафы не ставь перед окнами.
- Дверные проёмы и проходы должны оставаться свободными.

Формат для каждого предмета:
- type: sofa, bed, dining_table, desk, chair, armchair, wardrobe, tv_stand, coffee_table, bookshelf, nightstand, plant, floor_lamp, rug, bench, mirror
- anchor_wall: "north", "south", "east", "west", "center"
- placement: "center", "left", "right"
- distance_from_wall_cm: отступ от стены в см (0-20)
- dimensions_cm: {"width": int, "height": int, "depth": int}
- material: fabric, leather, oak, walnut, boucle, metal
- color: HEX-код цвета

Верни строго валидный JSON:
{"items": [{"type": "sofa", "anchor_wall": "south", "placement": "center", "distance_from_wall_cm": 10, "dimensions_cm": {"width": 210, "height": 85, "depth": 90}, "material": "fabric", "color": "#4A5568"}]}
"""

MIN_CLEARANCE_M = 0.6
DOOR_CLEARANCE_M = 0.9


def _room_bbox(room: Room) -> tuple[float, float, float, float]:
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _door_positions(room: Room) -> list[tuple[float, float]]:
    positions = []
    for wall in room.walls:
        for opening in wall.openings:
            op_type = opening.type.value if hasattr(opening.type, "value") else str(opening.type)
            if op_type != "door":
                continue
            x = wall.start[0] + (wall.end[0] - wall.start[0]) * opening.position
            y = wall.start[1] + (wall.end[1] - wall.start[1]) * opening.position
            positions.append((x, y))
    return positions


def _place_along_perimeter(room: Room, items: list[dict[str, Any]]) -> list[FurnitureItem]:
    return compile_semantic_layout_to_3d(room, items)


def _build_room_face_from_room(room: Room) -> RoomFace:
    wall_edges = []
    for i, w in enumerate(room.walls):
        wall_edges.append(
            WallEdge(
                id=f"wall_{i+1}",
                start=w.start,
                end=w.end,
                thickness=w.thickness if hasattr(w, "thickness") else 0.2,
                openings=[],
                is_exterior=False,
            )
        )
    return RoomFace(
        id=room.id,
        polygon=room.polygon,
        walls=wall_edges,
        area_sqm=getattr(room, "area_sqm", 0.0),
    )


def _get_rich_fallback_items(room: Room, preferences: UserPreferences | None) -> list[dict[str, Any]]:
    """Профессиональный гармоничный набор мебели на случай отсутствия ответа LLM."""
    r_type = room.type.value if hasattr(room.type, "value") else str(room.type)
    area = getattr(room, "area_sqm", 20.0)

    # 1. Большая гостиная или открытая студия (от 15 кв.м)
    if area >= 15.0 and r_type in ["living_room", "studio", "open_space", "lounge"]:
        return [
            # Зона отдыха (Lounge)
            {"type": "rug", "anchor_wall": "center", "placement": "center", "dimensions_cm": {"width": 240, "height": 2, "depth": 180}, "material": "wool", "color": "#E2DCD5"},
            {"type": "sofa", "anchor_wall": "south", "placement": "center", "distance_from_wall_cm": 10, "dimensions_cm": {"width": 210, "height": 85, "depth": 90}, "material": "fabric", "color": "#3B4252"},
            {"type": "coffee_table", "anchor_wall": "center", "placement": "center", "dimensions_cm": {"width": 100, "height": 45, "depth": 60}, "material": "oak", "color": "#D4A373"},
            {"type": "tv_stand", "anchor_wall": "north", "placement": "center", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 160, "height": 50, "depth": 40}, "material": "walnut", "color": "#4C3828"},
            {"type": "armchair", "anchor_wall": "east", "placement": "center", "distance_from_wall_cm": 15, "dimensions_cm": {"width": 85, "height": 80, "depth": 85}, "material": "boucle", "color": "#ECEFF4"},
            # Рабочее место / библиотека
            {"type": "desk", "anchor_wall": "west", "placement": "left", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 120, "height": 75, "depth": 60}, "material": "oak", "color": "#5E81AC"},
            {"type": "bookshelf", "anchor_wall": "west", "placement": "right", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 90, "height": 180, "depth": 35}, "material": "metal", "color": "#2E3440"},
            # Вертикальные акценты
            {"type": "floor_lamp", "anchor_wall": "south", "placement": "right", "distance_from_wall_cm": 15, "dimensions_cm": {"width": 40, "height": 160, "depth": 40}, "material": "brass", "color": "#D08770"},
            {"type": "plant", "anchor_wall": "north", "placement": "left", "distance_from_wall_cm": 15, "dimensions_cm": {"width": 45, "height": 130, "depth": 45}, "material": "ceramic", "color": "#A3BE8C"},
        ]
    # 2. Спальня (от 10 кв.м)
    elif r_type == "bedroom" and area >= 10.0:
        return [
            {"type": "rug", "anchor_wall": "south", "placement": "center", "dimensions_cm": {"width": 220, "height": 2, "depth": 180}, "material": "wool", "color": "#E5E9F0"},
            {"type": "bed", "anchor_wall": "south", "placement": "center", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 180, "height": 100, "depth": 200}, "material": "fabric", "color": "#434C5E"},
            {"type": "nightstand", "anchor_wall": "south", "placement": "left", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 45, "height": 50, "depth": 40}, "material": "oak", "color": "#D4A373"},
            {"type": "nightstand", "anchor_wall": "south", "placement": "right", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 45, "height": 50, "depth": 40}, "material": "oak", "color": "#D4A373"},
            {"type": "wardrobe", "anchor_wall": "west", "placement": "center", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 140, "height": 210, "depth": 60}, "material": "walnut", "color": "#2E3440"},
            {"type": "plant", "anchor_wall": "east", "placement": "center", "distance_from_wall_cm": 15, "dimensions_cm": {"width": 40, "height": 120, "depth": 40}, "material": "ceramic", "color": "#8FBCBB"},
        ]
    # 3. Прихожая / Коридор / Санузел (5-14 кв.м)
    elif area >= 5.0:
        return [
            {"type": "wardrobe", "anchor_wall": "west", "placement": "center", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 120, "height": 210, "depth": 60}, "material": "oak", "color": "#4C566A"},
            {"type": "bench", "anchor_wall": "north", "placement": "center", "distance_from_wall_cm": 5, "dimensions_cm": {"width": 90, "height": 45, "depth": 40}, "material": "leather", "color": "#D08770"},
            {"type": "mirror", "anchor_wall": "south", "placement": "center", "distance_from_wall_cm": 0, "dimensions_cm": {"width": 60, "height": 160, "depth": 10}, "material": "metal", "color": "#ECEFF4"},
        ]
    # 4. Компактное помещение (< 5 кв.м)
    else:
        return [
            {"type": "mirror", "anchor_wall": "north", "placement": "center", "distance_from_wall_cm": 0, "dimensions_cm": {"width": 60, "height": 80, "depth": 5}, "material": "glass", "color": "#ECEFF4"},
            {"type": "plant", "anchor_wall": "south", "placement": "right", "distance_from_wall_cm": 10, "dimensions_cm": {"width": 30, "height": 60, "depth": 30}, "material": "ceramic", "color": "#A3BE8C"},
        ]


async def _select_furniture_set(room: Room, preferences: UserPreferences | None) -> list[dict[str, Any]]:
    prefs_summary = preferences.model_dump() if preferences else {}
    style = prefs_summary.get("style", "Modern Minimalism")
    r_type = room.type.value if hasattr(room.type, "value") else str(room.type)

    # 1. Проверка Zero-LLM Fast-Path кэша (Мгновенное сопоставление с эталонами CubiCasa5K/LCSF)
    matched_items, similarity, archetype_id = FastPathLayoutMatcher.match_golden_template(
        room=room,
        style=style,
        similarity_threshold=0.88,
    )
    if matched_items:
        logger.info(
            "🚀 Zero-LLM Fast-Path selected for room '%s' (archetype: %s, sim: %.3f)",
            room.id,
            archetype_id,
            similarity,
        )
        return matched_items

    # 2. Нестандартная геометрия -> Точечный Vector RAG по стандартам Poché/Neufert
    room_face = _build_room_face_from_room(room)
    brief = generate_semantic_room_brief(room_face, room_type=r_type)
    rag_rules = VectorRuleRAG.format_rules_for_prompt(room_type=r_type, style=style)

    system_prompt = f"{SYSTEM_PROMPT}\n\n{rag_rules}" if rag_rules else SYSTEM_PROMPT

    user_prompt = (
        f"{brief.text_summary}\n\n"
        f"Пожелания клиента:\n"
        f"- Стиль: {style}\n"
        f"- Состав семьи / дети / животные: {prefs_summary.get('family_members', 1)}, pets={prefs_summary.get('pets', False)}\n"
        f"- Бюджет: {prefs_summary.get('budget', 'Medium')}"
    )

    try:
        result = await complete_json(system_prompt, user_prompt)
        items = result.get("items", [])
        if isinstance(items, list) and len(items) >= 3:
            return items
    except Exception as e:
        logger.warning(f"Semantic furniture selection failed, using rich fallback: {e}")

    return _get_rich_fallback_items(room, preferences)


async def run(rooms: list[Room], preferences: UserPreferences | None) -> list[FurnitureItem]:
    all_furniture: list[FurnitureItem] = []
    for room in rooms:
        items = await _select_furniture_set(room, preferences)
        placed = compile_semantic_layout_to_3d(room, items)
        all_furniture.extend(placed)
    return all_furniture
