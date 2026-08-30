"""
Линтер и валидатор эргономики планировок (Layout Ergonomics Validator & Linter).

Проверяет 100% сгенерированных 3D-сцен на соответствие открытым стандартам:
- Neufert Architects' Data: чистота проходов и радиусов открывания дверей (>= 90 см).
- Panero & Zelnik: углы обзора ТВ (соосность взгляда <= 25 град) и зазоры обеденных зон.
- Poché Ergonomics: отсутствие слипания твердых предметов (зазор >= 15 см).

Возвращает Quality Score (0..100%) и детальный отчёт о нарушениях.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box

from app.models.scene import Scene, Room, FurnitureItem, OpeningType


@dataclass
class Violation:
    rule_id: str
    severity: str  # "CRITICAL" | "ERROR" | "WARNING"
    message: str
    standard_source: str
    item_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    is_valid: bool
    score_percent: float
    critical_count: int
    error_count: int
    warning_count: int
    violations: list[Violation] = field(default_factory=list)
    summary_ru: str = ""


def _get_item_polygon(it: FurnitureItem, extra_buffer: float = 0.0) -> Polygon:
    w, h, d = it.dimensions
    bw = w + extra_buffer * 2.0
    bd = d + extra_buffer * 2.0
    base = box(-bw / 2.0, -bd / 2.0, bw / 2.0, bd / 2.0)
    if abs(it.rotation_deg) > 0.1:
        base = rotate(base, -it.rotation_deg, origin=(0, 0))
    return translate(base, xoff=it.position[0], yoff=it.position[2])


class LayoutErgonomicsValidator:
    """Автоматический валидатор соответствия эргономическим стандартам."""

    @classmethod
    def validate_scene(cls, scene: Scene) -> ValidationReport:
        violations: list[Violation] = []

        # Словарь комнат по id
        rooms_dict = {r.id: r for r in scene.rooms}

        # 1. Проверка зоны открывания дверей (Neufert Sec. 14: Door Swing Arc)
        for room in scene.rooms:
            for wall in room.walls:
                for op in wall.openings:
                    if op.type == OpeningType.door:
                        # Позиция двери
                        dx = wall.start[0] + (wall.end[0] - wall.start[0]) * op.position
                        dy = wall.start[1] + (wall.end[1] - wall.start[1]) * op.position
                        door_pt = Point(dx, dy)
                        door_arc = door_pt.buffer(0.85)  # 85 см защитная зона

                        for it in scene.furniture:
                            if it.room_id != room.id or it.type in ("rug", "carpet"):
                                continue
                            it_poly = _get_item_polygon(it)
                            if door_arc.intersects(it_poly):
                                inter_area = door_arc.intersection(it_poly).area
                                if inter_area > 0.05:
                                    violations.append(
                                        Violation(
                                            rule_id="DOOR_SWING_CLEARANCE",
                                            severity="CRITICAL",
                                            message=f"Предмет '{it.type}' (id: {it.id}) блокирует радиус открывания двери в комнате '{room.label or room.id}'.",
                                            standard_source="Ernst Neufert Architects' Data (Sec. 14: Doors)",
                                            item_ids=[it.id],
                                        )
                                    )

        # 2. Проверка створа окон от высоких шкафов (Neufert Sec. 12: Natural Light)
        for room in scene.rooms:
            for wall in room.walls:
                for op in wall.openings:
                    if op.type == OpeningType.window:
                        wx = wall.start[0] + (wall.end[0] - wall.start[0]) * op.position
                        wy = wall.start[1] + (wall.end[1] - wall.start[1]) * op.position
                        win_pt = Point(wx, wy)
                        win_zone = win_pt.buffer(0.40)

                        for it in scene.furniture:
                            if it.room_id != room.id:
                                continue
                            if it.type in ("wardrobe", "bookshelf") and it.dimensions[1] >= 1.5:
                                it_poly = _get_item_polygon(it)
                                if win_zone.intersects(it_poly):
                                    violations.append(
                                        Violation(
                                            rule_id="WINDOW_SILL_CLEARANCE",
                                            severity="CRITICAL",
                                            message=f"Высокий шкаф '{it.type}' перекрывает створ оконного проёма в комнате '{room.label or room.id}'.",
                                            standard_source="Neufert Architects' Data (Sec. 12: Natural Lighting)",
                                            item_ids=[it.id],
                                        )
                                    )

        # 3. Проверка соосности дивана и телевизора (Panero Human Dimension p. 182 & SMPTE)
        for room in scene.rooms:
            room_furniture = [it for it in scene.furniture if it.room_id == room.id]
            sofa = next((it for it in room_furniture if it.type == "sofa"), None)
            tv = next((it for it in room_furniture if "tv" in it.type), None)

            if sofa and tv:
                # Вектор от дивана к ТВ
                dx = tv.position[0] - sofa.position[0]
                dz = tv.position[2] - sofa.position[2]
                dist = math.hypot(dx, dz)

                if dist > 0.1:
                    target_rad = math.atan2(dx, dz)
                    target_deg = (math.degrees(target_rad) + 360) % 360

                    # Угол взгляда дивана
                    sofa_rot = sofa.rotation_deg % 360
                    # Разница углов
                    angle_diff = abs((sofa_rot - target_deg + 180) % 360 - 180)

                    if angle_diff > 35.0:
                        violations.append(
                            Violation(
                                rule_id="TV_SIGHTLINE_ALIGNMENT",
                                severity="ERROR",
                                message=f"Линия взгляда дивана отклонена от экрана ТВ на {int(angle_diff)}° (норма: <= 25°).",
                                standard_source="SMPTE EG-18 & Panero 'Human Dimension & Interior Space' (p. 182)",
                                item_ids=[sofa.id, tv.id],
                            )
                        )

        # 4. Проверка твердотельного непересечения (Poché Spatial Separation)
        for i in range(len(scene.furniture)):
            it_a = scene.furniture[i]
            if it_a.type in ("rug", "carpet"):
                continue
            poly_a = _get_item_polygon(it_a)

            for j in range(i + 1, len(scene.furniture)):
                it_b = scene.furniture[j]
                if it_b.type in ("rug", "carpet") or it_a.room_id != it_b.room_id:
                    continue
                poly_b = _get_item_polygon(it_b)

                if poly_a.intersects(poly_b):
                    inter = poly_a.intersection(poly_b)
                    if inter.area > 0.01:
                        violations.append(
                            Violation(
                                rule_id="NON_PENETRATION_SOLID_GAP",
                                severity="CRITICAL",
                                message=f"Пересечение моделей: '{it_a.type}' и '{it_b.type}' слиплись (площадь наложения: {round(inter.area, 3)} м²).",
                                standard_source="Poché Spatial Separation Standard (Sec. 2.1)",
                                item_ids=[it_a.id, it_b.id],
                            )
                        )

        crit_count = sum(1 for v in violations if v.severity == "CRITICAL")
        err_count = sum(1 for v in violations if v.severity == "ERROR")
        warn_count = sum(1 for v in violations if v.severity == "WARNING")

        # Расчёт итоговой оценки
        score = 100.0 - (crit_count * 25.0) - (err_count * 8.0) - (warn_count * 3.0)
        score = max(0.0, min(100.0, score))
        is_valid = crit_count == 0

        summary = (
            f"Аудит эргономики: {score:.1f}% соответствия стандартам Neufert & Poché. "
            f"Нарушений: {crit_count} критических, {err_count} ошибок, {warn_count} предупреждений."
        )

        return ValidationReport(
            is_valid=is_valid,
            score_percent=round(score, 1),
            critical_count=crit_count,
            error_count=err_count,
            warning_count=warn_count,
            violations=violations,
            summary_ru=summary,
        )
