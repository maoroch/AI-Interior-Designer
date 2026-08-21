"""
Тесты модуля векторного графа стен (backend/app/cv/wall_graph.py).
"""
import pytest
from app.cv.wall_graph import (
    build_wall_graph_from_segmentation,
    RoomFace,
    WallEdge,
    GraphOpening,
)
from app.cv.segmentation import RoomPolygon, WallSegment, DetectedOpening, segment_floor_plan


def test_shared_partition_wall_is_deduplicated():
    """Проверяет, что общая стена между двумя смежными комнатами создает ровно одно ребро графа."""
    # Комната 1: (0,0) -> (4,0) -> (4,4) -> (0,4)
    # Комната 2: (4,0) -> (8,0) -> (8,4) -> (4,4)
    # Общая стена: (4,0) -> (4,4)
    room1 = RoomPolygon(
        points=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        walls=[
            WallSegment(start=(0.0, 0.0), end=(4.0, 0.0)),
            WallSegment(start=(4.0, 0.0), end=(4.0, 4.0), openings=[DetectedOpening(type="door", position=0.5, width_m=0.9)]),
            WallSegment(start=(4.0, 4.0), end=(0.0, 4.0)),
            WallSegment(start=(0.0, 4.0), end=(0.0, 0.0)),
        ],
    )
    room2 = RoomPolygon(
        points=[(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)],
        walls=[
            WallSegment(start=(4.0, 0.0), end=(8.0, 0.0)),
            WallSegment(start=(8.0, 0.0), end=(8.0, 4.0)),
            WallSegment(start=(8.0, 4.0), end=(4.0, 4.0)),
            WallSegment(start=(4.0, 4.0), end=(4.0, 0.0)),
        ],
    )

    graph = build_wall_graph_from_segmentation([room1, room2])
    assert len(graph.rooms) == 2
    # Всего 8 сторон у двух комнат минус 1 общая = 7 уникальных ребер
    assert len(graph.edges) == 7

    # Проверяем, что обе комнаты ссылаются на один и тот же объект общей стены
    shared_edge_room1 = [w for w in graph.rooms[0].walls if (w.start == (4.0, 0.0) and w.end == (4.0, 4.0)) or (w.start == (4.0, 4.0) and w.end == (4.0, 0.0))][0]
    shared_edge_room2 = [w for w in graph.rooms[1].walls if (w.start == (4.0, 0.0) and w.end == (4.0, 4.0)) or (w.start == (4.0, 4.0) and w.end == (4.0, 0.0))][0]
    assert shared_edge_room1.id == shared_edge_room2.id
    assert len(shared_edge_room1.openings) == 1
    assert shared_edge_room1.openings[0].type == "door"


def test_wall_graph_snapping_near_nodes():
    """Проверяет привязку координат с небольшим расхождением (< 25см) к единой вершине."""
    room1 = RoomPolygon(
        points=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        walls=[],
    )
    room2 = RoomPolygon(
        points=[(4.05, 0.0), (8.0, 0.0), (8.0, 4.0), (4.02, 4.0)], # Смещение 2-5 см
        walls=[],
    )

    graph = build_wall_graph_from_segmentation([room1, room2])
    assert len(graph.rooms) == 2
    assert len(graph.edges) == 7


def test_wall_graph_on_sample_floorplan():
    """Проверяет построение графа на реальном образце чертежа sample_floorplan.png."""
    with open("sample_plans/plan1_studio.png", "rb") as f:
        img_bytes = f.read()

    seg_res = segment_floor_plan(img_bytes)
    graph = build_wall_graph_from_segmentation(seg_res.room_polygons)
    assert len(graph.rooms) == len(seg_res.room_polygons)
    assert len(graph.edges) >= 4
    for r in graph.rooms:
        assert len(r.polygon) >= 3
        assert r.area_sqm > 0
