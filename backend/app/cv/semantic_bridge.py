"""
Модуль семантического моста (Semantic Bridge).

Преобразует векторный граф помещения (RoomFace, WallEdge) в компактное,
понятное человеку и LLM семантическое описание в метрах и сантиметрах:
- Габариты: ширина, глубина, площадь в м².
- Стены: ориентация (север, юг, восток, запад), длина, назначение (окно, дверь, глухая стена).
- Полная изоляция точных float-координат от LLM-промптов.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.cv.wall_graph import RoomFace, WallEdge


@dataclass
class SemanticWallDesc:
    wall_id: str
    orientation: str  # "north" | "south" | "east" | "west" | "custom"
    length_m: float
    is_exterior: bool
    features: list[str] = field(default_factory=list)
    has_window: bool = False
    has_door: bool = False
    is_solid: bool = True
    is_open_passage: bool = False


@dataclass
class SemanticRoomBrief:
    room_id: str
    room_type: str
    width_m: float
    depth_m: float
    area_sqm: float
    is_l_shaped: bool
    walls: list[SemanticWallDesc]
    text_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "dimensions": {
                "width_m": self.width_m,
                "depth_m": self.depth_m,
                "area_sqm": self.area_sqm,
                "is_l_shaped": self.is_l_shaped,
            },
            "walls": [
                {
                    "id": w.wall_id,
                    "orientation": w.orientation,
                    "length_m": w.length_m,
                    "features": w.features,
                    "is_solid": w.is_solid,
                    "has_window": w.has_window,
                    "has_door": w.has_door,
                    "is_open_passage": w.is_open_passage,
                }
                for w in self.walls
            ],
            "text_summary": self.text_summary,
        }


def _determine_orientation(
    start: tuple[float, float],
    end: tuple[float, float],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> str:
    """Определяет сторону света стены относительно центра и габаритов комнаты."""
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0

    # Горизонтальная стена (dx > dy)
    if dx >= dy * 1.5:
        dist_top = abs(mid_y - min_y)
        dist_bot = abs(mid_y - max_y)
        return "north" if dist_top <= dist_bot else "south"

    # Вертикальная стена (dy > dx)
    if dy >= dx * 1.5:
        dist_left = abs(mid_x - min_x)
        dist_right = abs(mid_x - max_x)
        return "west" if dist_left <= dist_right else "east"

    # Диагональная или смешанная
    if mid_y < (min_y + max_y) / 2.0:
        return "north" if mid_x < (min_x + max_x) / 2.0 else "east"
    else:
        return "west" if mid_x < (min_x + max_x) / 2.0 else "south"


def generate_semantic_room_brief(
    room: RoomFace,
    room_type: str = "living_room",
) -> SemanticRoomBrief:
    """
    Формирует структурированный семантический бриф для LLM.
    """
    pts = room.polygon
    if not pts:
        return SemanticRoomBrief(
            room_id=room.id,
            room_type=room_type,
            width_m=4.0,
            depth_m=4.0,
            area_sqm=16.0,
            is_l_shaped=False,
            walls=[],
            text_summary="Standard rectangular room 4.0m x 4.0m",
        )

    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)

    width_m = round(max(1.0, max_x - min_x), 2)
    depth_m = round(max(1.0, max_y - min_y), 2)
    area_sqm = round(room.area_sqm if room.area_sqm > 0 else width_m * depth_m, 1)
    is_l_shaped = len(pts) > 4

    walls_desc: list[SemanticWallDesc] = []
    summary_lines = [
        f"Room: {room_type.replace('_', ' ').title()} (ID: {room.id})",
        f"Dimensions: {width_m:.2f}m wide x {depth_m:.2f}m deep (Total Area: {area_sqm:.1f} m²)",
        "Walls & Openings Layout:",
    ]

    orientation_counts: dict[str, int] = {}

    for wall in room.walls:
        base_orient = _determine_orientation(wall.start, wall.end, min_x, min_y, max_x, max_y)
        orientation_counts[base_orient] = orientation_counts.get(base_orient, 0) + 1
        orient_label = base_orient if orientation_counts[base_orient] == 1 else f"{base_orient}_{orientation_counts[base_orient]}"

        length_m = round(math.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]), 2)
        has_win = any(o.type == "window" for o in wall.openings)
        has_dr = any(o.type == "door" for o in wall.openings)
        is_passage = wall.is_virtual or any(o.type == "passage" for o in wall.openings)

        features: list[str] = []
        if wall.is_exterior:
            features.append("exterior_facade")
        if is_passage:
            features.append("open_passage")
        elif not wall.openings:
            features.append("solid_wall")

        for o in wall.openings:
            if o.type == "window":
                features.append(f"window_{o.width_m}m")
            elif o.type == "door":
                features.append(f"door_{o.width_m}m")

        is_solid = not has_win and not has_dr and not is_passage

        desc = SemanticWallDesc(
            wall_id=wall.id,
            orientation=orient_label,
            length_m=length_m,
            is_exterior=wall.is_exterior,
            features=features,
            has_window=has_win,
            has_door=has_dr,
            is_solid=is_solid,
            is_open_passage=is_passage,
        )
        walls_desc.append(desc)

        # Человекочитаемая строка
        feat_str = ", ".join(features) if features else "solid wall"
        summary_lines.append(f"  - Wall {orient_label.upper()} ({length_m:.2f}m): {feat_str}")

    return SemanticRoomBrief(
        room_id=room.id,
        room_type=room_type,
        width_m=width_m,
        depth_m=depth_m,
        area_sqm=area_sqm,
        is_l_shaped=is_l_shaped,
        walls=walls_desc,
        text_summary="\n".join(summary_lines),
    )
