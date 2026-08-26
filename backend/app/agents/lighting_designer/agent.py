"""
Agent 6 — Lighting Designer (Photometric Lighting Engine).

Использует фотометрический расчёт освещённости (Photometric Lighting & Lux Targets):
- Фотометрический расчёт требуемого светового потока (люмены) и температуры Кельвина.
- Многоточечное потолочное освещение (равномерное распределение люксов по площади).
- Локальная подсветка функциональных зон (подвесы над столом, торшеры, бра).
"""
from __future__ import annotations

import uuid

from app.agents.furniture_planner.math_engine import PhotometricLightingCalculator
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
        area = getattr(room, "area_sqm", 20.0)
        r_type = room.type.value if hasattr(room.type, "value") else str(room.type)

        # Фотометрический расчёт СНиП / Lux
        photo_req = PhotometricLightingCalculator.calculate_lighting_requirements(area, r_type)
        kelvin = photo_req["color_temperature_k"]

        # Если в комнате есть естественный свет из окна — чуть повышаем цветовую температуру
        if _has_window(room):
            kelvin = min(4000, kelvin + 300)

        # Основной потолочный светильник
        lights.append(
            LightSource(
                id=f"light_{uuid.uuid4().hex[:8]}",
                type="ceiling",
                position=(cx, room.height - 0.1, cy),
                color_temperature_k=kelvin,
                intensity=round(min(1.0, photo_req["target_lux"] / 200.0), 2),
            )
        )

        # Для больших комнат (>25 кв.м) добавляем вспомогательный акцентный потолочный спот
        if area >= 25.0:
            lights.append(
                LightSource(
                    id=f"light_{uuid.uuid4().hex[:8]}",
                    type="ceiling",
                    position=(cx + 1.2, room.height - 0.1, cy - 1.0),
                    color_temperature_k=kelvin,
                    intensity=0.75,
                )
            )

    # Точечный акцентный свет над обеденным столом и рабочим местом
    for item in furniture:
        if item.type in ["dining_table", "table"]:
            x, _, z = item.position
            lights.append(
                LightSource(
                    id=f"light_{uuid.uuid4().hex[:8]}",
                    type="pendant",
                    position=(x, 2.2, z),
                    color_temperature_k=2700,
                    intensity=0.85,
                )
            )
        elif item.type == "floor_lamp":
            x, _, z = item.position
            lights.append(
                LightSource(
                    id=f"light_{uuid.uuid4().hex[:8]}",
                    type="spot",
                    position=(x, 1.4, z),
                    color_temperature_k=2700,
                    intensity=0.60,
                )
            )

    return lights
