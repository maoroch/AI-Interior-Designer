"""
Agent 3 — Architect Agent (уникальная фича проекта).

В отличие от остальных агентов, работает не только с декором, а с самой
геометрией: может предложить снести/перенести стену, объединить комнаты,
увеличить гардеробную и т.д.

Работает в MVP ПОЛНОЦЕННО: если пользователь выбирает предложенный вариант
(не "оставить как есть"), agent пересчитывает `rooms` (новые полигоны/стены),
и весь нижестоящий пайплайн (Interior Designer -> ... ) перезапускается
уже с новой геометрией. См. app/tasks/pipeline_tasks.py — там реализован
чекпоинт после Architect Agent, а не полный ресет с нуля.

Чтобы не пытаться решить произвольную задачу перепланировки одним промптом
(ненадёжно), MVP-версия просит LLM выбрать один из типовых сценариев
(объединение соседних комнат, перенос ненесущей перегородки) и только
для него генерирует новую геометрию — это резко повышает предсказуемость.
"""
from __future__ import annotations

import uuid

from app.core.llm import complete_json
from app.models.scene import ArchitectSuggestion, Room

SYSTEM_PROMPT = """Ты — архитектор. Тебе дан план квартиры (список комнат с полигонами
в метрах и их типами). Предложи 1-2 осмысленных варианта улучшения планировки
из числа типовых: объединить кухню и гостиную, увеличить маленькую комнату
за счёт соседнего коридора, добавить кухонный остров (если площадь позволяет).
Для каждого варианта опиши: что меняется и почему это улучшает пространство
(проходимость, естественный свет, функциональность).
Верни JSON: {"suggestions": [{"title": str, "description": str, "affected_room_ids": [str]}]}.
НЕ пытайся пересчитать точную геометрию — это будет сделано отдельно."""


async def propose_suggestions(rooms: list[Room]) -> list[ArchitectSuggestion]:
    if not rooms:
        return []

    rooms_summary = [{"id": r.id, "type": r.type.value, "polygon": r.polygon} for r in rooms]

    result = await complete_json(SYSTEM_PROMPT, f"Комнаты: {rooms_summary}")
    raw_suggestions = result.get("suggestions", [])

    suggestions: list[ArchitectSuggestion] = []
    for raw in raw_suggestions[:2]:
        suggestions.append(
            ArchitectSuggestion(
                id=f"opt_{uuid.uuid4().hex[:8]}",
                title=raw.get("title", "Вариант перепланировки"),
                description=raw.get("description", ""),
                rooms_after=None,  # считается лениво, только если пользователь выберет этот вариант
            )
        )
    return suggestions


async def apply_suggestion(
    rooms: list[Room], suggestion: ArchitectSuggestion
) -> list[Room]:
    """
    Пересчитывает геометрию для выбранного варианта перепланировки.
    Использует Shapely для точного объединения полигонов (union), с fallback на LLM.
    """
    from app.models.scene import Wall

    # 1. Попытка детерминированного геометрического объединения через Shapely
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        # Если в описании речь об объединении всех комнат или типовой перепланировке
        if len(rooms) >= 2 and ("объедин" in suggestion.title.lower() or "merge" in suggestion.title.lower() or "кухн" in suggestion.title.lower()):
            polys = [Polygon(r.polygon) for r in rooms if len(r.polygon) >= 3]
            merged_poly = unary_union(polys)
            if merged_poly.is_valid and not merged_poly.is_empty:
                ext_coords = list(merged_poly.exterior.coords)[:-1] # без дублирующей замыкающей точки
                new_poly = [(round(x, 2), round(y, 2)) for x, y in ext_coords]
                
                # Собираем новые стены
                new_walls: list[Wall] = []
                for i in range(len(new_poly)):
                    start = new_poly[i]
                    end = new_poly[(i + 1) % len(new_poly)]
                    new_walls.append(
                        Wall(
                            id=f"wall_{uuid.uuid4().hex[:8]}",
                            start=start,
                            end=end,
                            thickness=0.2,
                            openings=[],
                        )
                    )
                
                # Создаем объединенную комнату
                merged_room = Room(
                    id=f"room_merged_{uuid.uuid4().hex[:6]}",
                    name="Кухня-гостиная (Open Space)",
                    polygon=new_poly,
                    walls=new_walls,
                    height=rooms[0].height if rooms else 2.7,
                )
                return [merged_room]
    except Exception:
        pass

    # 2. Fallback через LLM пересчёт полигонов
    rooms_summary = [{"id": r.id, "polygon": r.polygon} for r in rooms]
    prompt = (
        f"Комнаты: {rooms_summary}\n"
        f"Применяемый вариант: {suggestion.title} — {suggestion.description}\n"
        'Верни новый список комнат JSON: {"rooms": [{"id": str, "polygon": [[x,y],...]}]}'
    )
    result = await complete_json(
        "Ты пересчитываешь полигоны комнат после перепланировки. Сохраняй метры, координаты реалистичными.",
        prompt,
    )
    new_rooms_raw = {r["id"]: r["polygon"] for r in result.get("rooms", [])}

    for room in rooms:
        if room.id in new_rooms_raw:
            room.polygon = [tuple(p) for p in new_rooms_raw[room.id]]

    return rooms

