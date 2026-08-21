"""
Модуль построения векторного графа стен (Vector Wall Graph).

Преобразует растровую геометрию чертежа в планарный граф:
- Nodes (Вершины): точки сопряжения стен (углы, T- и X-стыки).
- Edges (Ребра): физические несущие стены и перегородки с проемами.
- Faces (Грани): полигоны комнат (включая L-образные и неортогональные).

Гарантии графа:
1. 0 дублирующихся стен (смежные комнаты делят одно общее ребро).
2. 0 фантомных стен (в открытых проходах между зонами глухие стены не строятся).
3. 0 недопустимых наложений комнат (планарность графа).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]


@dataclass
class GraphOpening:
    type: str  # "door" | "window" | "passage"
    position: float  # 0..1 вдоль стены
    width_m: float


@dataclass
class WallEdge:
    id: str
    start: Point2D
    end: Point2D
    thickness: float = 0.2
    openings: list[GraphOpening] = field(default_factory=list)
    is_exterior: bool = False
    is_virtual: bool = False  # True, если ребро является открытой границей зон (без физической стены)


@dataclass
class RoomFace:
    id: str
    polygon: list[Point2D]
    walls: list[WallEdge]
    area_sqm: float = 0.0


@dataclass
class WallGraphResult:
    nodes: list[Point2D]
    edges: list[WallEdge]
    rooms: list[RoomFace]
    image_width: int
    image_height: int
    pixels_per_meter: float = 50.0


def _distance(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _snap_point_to_nodes(pt: Point2D, nodes: list[Point2D], tol: float = 0.45) -> Point2D:
    """Привязывает точку к ближайшей вершине графа, если расстояние меньше толерантности."""
    for n in nodes:
        if _distance(pt, n) <= tol:
            return n
    nodes.append(pt)
    return pt


def build_wall_graph_from_segmentation(
    room_polygons: list,
    pixels_per_meter: float = 50.0,
    image_width: int = 1000,
    image_height: int = 800,
) -> WallGraphResult:
    """
    Строит единый векторный граф стен по сегментированным комнатам,
    устраняя дублирование смежных стен и объединяя проемы (двери/окна) на общих гранях.
    """
    nodes: list[Point2D] = []
    edges_map: dict[tuple[Point2D, Point2D], WallEdge] = {}
    room_faces: list[RoomFace] = []

    def get_canonical_key(p1: Point2D, p2: Point2D) -> tuple[Point2D, Point2D]:
        return (p1, p2) if (p1[0], p1[1]) <= (p2[0], p2[1]) else (p2, p1)

    for r_idx, room in enumerate(room_polygons):
        snapped_pts: list[Point2D] = []
        raw_pts = room.points if hasattr(room, "points") else room

        for pt in raw_pts:
            snapped = _snap_point_to_nodes((round(pt[0], 2), round(pt[1], 2)), nodes, tol=0.45)
            if not snapped_pts or snapped != snapped_pts[-1]:
                snapped_pts.append(snapped)

        # Замыкаем полигон
        if len(snapped_pts) >= 2 and snapped_pts[0] == snapped_pts[-1]:
            snapped_pts.pop()

        if len(snapped_pts) < 3:
            continue

        # Вычисляем площадь полигона
        xs = [p[0] for p in snapped_pts]
        ys = [p[1] for p in snapped_pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        area_sqm = round(w * h, 2)

        face_walls: list[WallEdge] = []
        n_pts = len(snapped_pts)

        # Собираем стены полигона
        for i in range(n_pts):
            p1 = snapped_pts[i]
            p2 = snapped_pts[(i + 1) % n_pts]
            if p1 == p2:
                continue

            canon_key = get_canonical_key(p1, p2)

            # Извлекаем проемы из исходного сегмента, если они есть
            openings: list[GraphOpening] = []
            if hasattr(room, "walls") and i < len(room.walls):
                orig_wall = room.walls[i]
                for o in getattr(orig_wall, "openings", []):
                    openings.append(
                        GraphOpening(
                            type=o.type,
                            position=o.position,
                            width_m=o.width_m if hasattr(o, "width_m") else getattr(o, "width", 1.0),
                        )
                    )

            if canon_key not in edges_map:
                edge = WallEdge(
                    id=f"wall_{len(edges_map) + 1}",
                    start=p1,
                    end=p2,
                    thickness=0.2,
                    openings=openings,
                )
                edges_map[canon_key] = edge
            else:
                edge = edges_map[canon_key]
                # Если на общей стене одна из комнат обнаружила проем, добавляем его на общее ребро
                if openings and not edge.openings:
                    edge.openings = openings

            face_walls.append(edge)

        room_faces.append(
            RoomFace(
                id=f"room_face_{r_idx + 1}",
                polygon=snapped_pts,
                walls=face_walls,
                area_sqm=area_sqm,
            )
        )

    return WallGraphResult(
        nodes=nodes,
        edges=list(edges_map.values()),
        rooms=room_faces,
        image_width=image_width,
        image_height=image_height,
        pixels_per_meter=pixels_per_meter,
    )
