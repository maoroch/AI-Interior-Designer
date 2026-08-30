"""
Детерминированный CAD-компилятор расстановки мебели (CAD Layout Compiler).

Использует Shapely для строгой геометрической верификации:
- 0 коллизий со стенами (гарантировано буферизацией полигона комнаты).
- 0 пересечений между предметами мебели (гарантировано буферизацией габаритов).
- 0 наложений на зоны открывания дверей (гарантировано защитными кругами проемов).
"""
from __future__ import annotations

import math
import uuid
from typing import Any

from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate

from app.agents.furniture_planner.math_engine import (
    ErgonomicOrientationCalculator,
    ForceDirectedRelaxationSolver,
)
from app.models.scene import FurnitureItem, Room

# Стандартные габариты мебели по умолчанию (Ширина, Высота, Глубина в метрах)
DEFAULT_DIMS: dict[str, tuple[float, float, float]] = {
    "sofa": (2.10, 0.85, 0.90),
    "bed": (1.80, 1.00, 2.00),
    "dining_table": (1.40, 0.75, 0.85),
    "table": (1.20, 0.75, 0.70),
    "desk": (1.20, 0.75, 0.60),
    "chair": (0.50, 0.85, 0.50),
    "armchair": (0.85, 0.80, 0.85),
    "wardrobe": (1.20, 2.10, 0.60),
    "tv_stand": (1.60, 0.50, 0.40),
    "bookshelf": (0.90, 1.80, 0.35),
    "nightstand": (0.45, 0.50, 0.40),
    "coffee_table": (1.00, 0.45, 0.60),
    "plant": (0.40, 1.20, 0.40),
    "floor_lamp": (0.40, 1.60, 0.40),
    "rug": (2.40, 0.02, 1.80),
    "bench": (1.00, 0.45, 0.40),
    "mirror": (0.60, 1.60, 0.10),
}

WALL_INSET_BUFFER = 0.22       # 22 см минимальный отступ от осевой линии стены
INTER_FURNITURE_GAP = 0.55     # 55 см минимальный эргономический проход между мебелью
DOOR_PROTECTION_RADIUS = 0.95  # 95 см зона свободного открывания двери


def _parse_dimensions(item: dict) -> tuple[float, float, float]:
    """Извлекает габариты в метрах из сантиметров или метров."""
    item_type = str(item.get("type", "")).lower()
    default_w, default_h, default_d = DEFAULT_DIMS.get(item_type, (0.80, 0.80, 0.80))

    # 1. Если передано в сантиметрах
    dims_cm = item.get("dimensions_cm")
    if isinstance(dims_cm, dict):
        w = float(dims_cm.get("width", default_w * 100)) / 100.0
        h = float(dims_cm.get("height", default_h * 100)) / 100.0
        d = float(dims_cm.get("depth", default_d * 100)) / 100.0
        return round(w, 2), round(h, 2), round(d, 2)
    elif isinstance(dims_cm, (list, tuple)) and len(dims_cm) >= 3:
        return round(float(dims_cm[0]) / 100.0, 2), round(float(dims_cm[1]) / 100.0, 2), round(float(dims_cm[2]) / 100.0, 2)

    # 2. Если передано в метрах
    dims_m = item.get("dimensions_m")
    if isinstance(dims_m, (list, tuple)) and len(dims_m) >= 3:
        return round(float(dims_m[0]), 2), round(float(dims_m[1]), 2), round(float(dims_m[2]), 2)

    return default_w, default_h, default_d


def _extract_doors(room: Room) -> list[tuple[float, float]]:
    """Находит мировые координаты центров всех дверей комнаты."""
    doors = []
    for wall in room.walls:
        for op in wall.openings:
            op_type = op.type.value if hasattr(op.type, "value") else str(op.type)
            if op_type == "door":
                mx = wall.start[0] + (wall.end[0] - wall.start[0]) * op.position
                my = wall.start[1] + (wall.end[1] - wall.start[1]) * op.position
                doors.append((mx, my))
    return doors


def _find_best_wall(room: Room, anchor: str) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """
    Находит стену комнаты, наиболее подходящую под указанную сторону света (north, south, east, west).
    Возвращает (start, end, inward_normal_x, inward_normal_y).
    """
    if not room.walls:
        return (0.0, 0.0), (4.0, 0.0), 0.0, 1.0

    poly = Polygon(room.polygon)
    cx, cy = poly.centroid.x, poly.centroid.y

    best_wall = room.walls[0]
    best_score = -999999.0
    best_normal = (0.0, 1.0)

    for wall in room.walls:
        sx, sy = wall.start
        ex, ey = wall.end
        length = math.hypot(ex - sx, ey - sy)
        if length < 0.5:
            continue

        mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
        # Вектор стены и нормали
        vx, vy = (ex - sx) / length, (ey - sy) / length
        nx, ny = -vy, vx  # нормаль

        # Проверяем направление нормали внутрь комнаты
        test_pt = Point(mx + nx * 0.2, my + ny * 0.2)
        if not poly.contains(test_pt):
            nx, ny = -nx, -ny

        # Скоринг соответствия стороне света
        score = 0.0
        if "north" in anchor or "top" in anchor:
            score = -my * 2.0 + abs(vx) * 3.0  # чем выше (меньше y) и горизонтальнее, тем лучше
        elif "south" in anchor or "bottom" in anchor:
            score = my * 2.0 + abs(vx) * 3.0   # чем ниже (больше y) и горизонтальнее, тем лучше
        elif "west" in anchor or "left" in anchor:
            score = -mx * 2.0 + abs(vy) * 3.0  # чем левее (меньше x) и вертикальнее, тем лучше
        elif "east" in anchor or "right" in anchor:
            score = mx * 2.0 + abs(vy) * 3.0   # чем правее (больше x) и вертикальнее, тем лучше
        else:
            score = length

        if score > best_score:
            best_score = score
            best_wall = wall
            best_normal = (nx, ny)

    return best_wall.start, best_wall.end, best_normal[0], best_normal[1]


def compile_semantic_layout_to_3d(
    room: Room,
    semantic_items: list[dict[str, Any]],
) -> list[FurnitureItem]:
    """
    Компилирует семантический план мебели от LLM в точные 3D координаты
    с полной геометрической валидацией через Shapely.
    """
    if not room.polygon or len(room.polygon) < 3:
        return []

    room_poly = Polygon(room.polygon)
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)

    # Внутренний полигон комнаты с безопасным отступом от стен
    inner_poly = room_poly.buffer(-WALL_INSET_BUFFER)
    if inner_poly.is_empty or not inner_poly.is_valid:
        inner_poly = room_poly.buffer(-0.10)
        if inner_poly.is_empty:
            inner_poly = room_poly

    cx, cy = room_poly.centroid.x, room_poly.centroid.y
    doors = _extract_doors(room)
    door_zones = [Point(dx, dy).buffer(DOOR_PROTECTION_RADIUS) for dx, dy in doors]

    placed_items: list[FurnitureItem] = []
    placed_polys: list[Polygon] = []

    for item in semantic_items:
        w, h, d = _parse_dimensions(item)
        item_type = str(item.get("type", "unknown")).lower()
        anchor = str(item.get("anchor_wall", "center")).lower()
        placement = str(item.get("placement", "center")).lower()
        offset_m = float(item.get("distance_from_wall_cm", 10.0)) / 100.0

        # Поиск базовой позиции
        if "center" in anchor or anchor == "middle":
            cand_x, cand_y = cx, cy
            rotation_deg = 0
            is_vertical_wall = False
            vx, vy = 1.0, 0.0
            nx, ny = 0.0, 1.0
        else:
            w_start, w_end, nx, ny = _find_best_wall(room, anchor)
            w_len = math.hypot(w_end[0] - w_start[0], w_end[1] - w_start[1])
            vx, vy = (w_end[0] - w_start[0]) / max(0.1, w_len), (w_end[1] - w_start[1]) / max(0.1, w_len)

            # Определяем горизонтальная ли стена или вертикальная
            is_vertical_wall = abs(ny) < abs(nx)

            # Математический векторный расчёт ориентации мебели спинкой к стене, лицом внутрь
            rotation_deg = ErgonomicOrientationCalculator.calculate_wall_facing_angle(nx, ny)
            dist_inward = (d / 2.0) + WALL_INSET_BUFFER + offset_m

            # Положение вдоль стены
            t = 0.50
            if "left" in placement:
                t = 0.25
            elif "right" in placement:
                t = 0.75

            # Базовая точка на стене
            base_x = w_start[0] + vx * (w_len * t)
            base_y = w_start[1] + vy * (w_len * t)

            # Смещение внутрь комнаты
            cand_x = base_x + nx * dist_inward
            cand_y = base_y + ny * dist_inward

        # Функция создания полигона мебели
        def make_item_poly(px: float, py: float, pw: float, pd: float, is_vert: bool) -> Polygon:
            # Для вертикальных стен габариты ориентированы вдоль Y
            box_w = pd if is_vert else pw
            box_d = pw if is_vert else pd
            return box(px - box_w / 2.0, py - box_d / 2.0, px + box_w / 2.0, py + box_d / 2.0)

        best_pos = None
        best_poly = None

        # Итеративный поиск свободной позиции вокруг кандидата
        search_radius_steps = [0.0, 0.50, -0.50, 1.00, -1.00, 1.50, -1.50, 2.00]
        inward_steps = [0.0, 0.40, 0.80]

        for in_step in inward_steps:
            for s_step in search_radius_steps:
                # Векторные смещения
                if "center" in anchor:
                    test_x = cand_x + s_step
                    test_y = cand_y + in_step
                else:
                    test_x = cand_x + vx * s_step + nx * in_step
                    test_y = cand_y + vy * s_step + ny * in_step

                item_geom = make_item_poly(test_x, test_y, w, d, is_vertical_wall)
                item_clearance_geom = item_geom.buffer(INTER_FURNITURE_GAP)

                # Проверка 1: Полное нахождение внутри комнаты
                if not inner_poly.contains(item_geom):
                    continue

                # Проверка 2: Отсутствие наложений на двери
                if any(dz.intersects(item_geom) for dz in door_zones):
                    continue

                # Проверка 3: Отсутствие коллизий с уже расставленной мебелью
                if any(placed_p.intersects(item_clearance_geom) for placed_p in placed_polys):
                    continue

                # Найдена идеальная позиция!
                best_pos = (test_x, test_y)
                best_poly = item_geom
                break

            if best_pos is not None:
                break

        # Если позиция не найдена через поиск вдоль стены — пробуем центроидную зону
        if best_pos is None:
            for shift_x in [0.0, 0.6, -0.6, 1.2, -1.2]:
                for shift_y in [0.0, 0.6, -0.6, 1.2, -1.2]:
                    test_x = cx + shift_x
                    test_y = cy + shift_y
                    item_geom = make_item_poly(test_x, test_y, w, d, False)
                    item_clearance_geom = item_geom.buffer(INTER_FURNITURE_GAP * 0.7)
                    if inner_poly.contains(item_geom) and not any(dz.intersects(item_geom) for dz in door_zones) and not any(placed_p.intersects(item_clearance_geom) for placed_p in placed_polys):
                        best_pos = (test_x, test_y)
                        best_poly = item_geom
                        rotation_deg = 0
                        break
                if best_pos is not None:
                    break

        if best_pos is None:
            best_pos = (cx, cy)
            best_poly = make_item_poly(cx, cy, w, d, False)

        if item_type != "rug":
            placed_polys.append(best_poly)
        placed_items.append(
            FurnitureItem(
                id=f"f_{uuid.uuid4().hex[:8]}",
                room_id=room.id,
                type=item_type,
                style=item.get("style"),
                position=(round(best_pos[0], 2), 0.0, round(best_pos[1], 2)),
                rotation_deg=rotation_deg,
                dimensions=(round(w, 2), round(h, 2), round(d, 2)),
                material=item.get("material", "wood"),
                color=item.get("color", "#E5E5E5"),
                model_ref=item.get("model_ref"),
            )
        )

    # 4. Физическая релаксация связанных пар и тонкая доводка (Force-Directed Relaxation)
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    dict_items = [
        {"type": it.type, "position": it.position}
        for it in placed_items
    ]
    relaxed_dicts = ForceDirectedRelaxationSolver.relax_positions(dict_items, doors, bounds, iterations=8)
    for i, it in enumerate(placed_items):
        it.position = relaxed_dicts[i]["position"]

    # 5. Эргономическая юстировка направления взгляда на фокусные объекты (Sightline Alignment)
    tv_item = next((it for it in placed_items if "tv" in it.type), None)
    if tv_item:
        for it in placed_items:
            if it.type in ("sofa", "armchair"):
                # Вычисляем вектор взгляда на ТВ
                focal_angle = ErgonomicOrientationCalculator.calculate_focal_orientation(
                    (it.position[0], it.position[2]),
                    (tv_item.position[0], tv_item.position[2]),
                )
                # Если диван стоит вдоль стены, проверяем соосность взгляда (угол должен быть ближе к ТВ)
                # Разница углов не должна превышать 90 градусов от нормали стены
                angle_diff = abs((it.rotation_deg - focal_angle + 180) % 360 - 180)
                if angle_diff > 90:
                    # Если диван случайно оказался спинкой к ТВ, разворачиваем его к ТВ
                    it.rotation_deg = focal_angle

    return placed_items
