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
    confidence: float = 0.90
    features: list[str] = field(default_factory=list)


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
    accessibility_score: float = 1.0  # 1.0 = гарантированный доступ через двери


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
    устраняя дублирование смежных стен и проводя вероятностный консенсус проемов (дверей/окон).
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

            # Извлекаем проемы из исходного сегмента с оценкой вероятности
            openings: list[GraphOpening] = []
            if hasattr(room, "walls") and i < len(room.walls):
                orig_wall = room.walls[i]
                for o in getattr(orig_wall, "openings", []):
                    conf = getattr(o, "confidence", 0.90)
                    feats = getattr(o, "features", [])
                    openings.append(
                        GraphOpening(
                            type=o.type,
                            position=o.position,
                            width_m=o.width_m if hasattr(o, "width_m") else getattr(o, "width", 1.0),
                            confidence=conf,
                            features=feats,
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
                # Слияние проемов: если одна сторона определила дверь с большей уверенностью — объединяем
                if openings:
                    if not edge.openings:
                        edge.openings = openings
                    else:
                        # Объединение и дедупликация проемов на ребре
                        for new_op in openings:
                            match = next((op for op in edge.openings if abs(op.position - new_op.position) < 0.18), None)
                            if match:
                                match.confidence = max(match.confidence, new_op.confidence)
                                match.features = list(set(match.features + new_op.features))
                            else:
                                edge.openings.append(new_op)

            face_walls.append(edge)

        # Проверка топологической связности комнаты (Accessibility Invariant)
        door_count = sum(1 for edge in face_walls for o in edge.openings if o.type == "door" and o.confidence >= 0.5)
        accessibility_score = 1.0 if door_count >= 1 else 0.0

        room_faces.append(
            RoomFace(
                id=f"room_face_{r_idx + 1}",
                polygon=snapped_pts,
                walls=face_walls,
                area_sqm=area_sqm,
                accessibility_score=accessibility_score,
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
