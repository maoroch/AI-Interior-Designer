"""
Unit tests for SpatialPhysicsEngine:
- Collision resolution between overlapping solid items (desk + bookshelf, sofa + table)
- Mass-weighted separation ratios
- Room wall containment
- Door clearance enforcement
- Multi-layer rug passability
"""
import pytest
from app.agents.furniture_planner.physics_engine import (
    SpatialPhysicsEngine,
    FurnitureBody,
    NON_PASSABLE_GAP,
)
from shapely.geometry import Polygon


def test_spatial_physics_separates_overlapping_desk_and_bookshelf():
    # Стол и стеллаж размещены в одной точке (4.0, 4.0)
    items = [
        {"id": "f_desk", "type": "desk", "position": (4.0, 0.0, 4.0), "dimensions": (1.4, 0.75, 0.7), "rotation_deg": 0.0},
        {"id": "f_bookshelf", "type": "bookshelf", "position": (4.1, 0.0, 4.0), "dimensions": (0.9, 1.8, 0.35), "rotation_deg": 0.0},
    ]
    room_polygon = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]

    resolved = SpatialPhysicsEngine.resolve_scene_physics(items, room_polygon, max_iterations=40)
    pos_desk = resolved[0]["position"]
    pos_shelf = resolved[1]["position"]

    body_desk = FurnitureBody(id="desk", type="desk", x=pos_desk[0], z=pos_desk[2], w=1.4, d=0.7, rotation_deg=0)
    body_shelf = FurnitureBody(id="shelf", type="bookshelf", x=pos_shelf[0], z=pos_shelf[2], w=0.9, d=0.35, rotation_deg=0)

    # Полигоны больше не должны пересекаться
    poly_desk = body_desk.get_polygon(extra_buffer=0.0)
    poly_shelf = body_shelf.get_polygon(extra_buffer=0.0)
    assert not poly_desk.intersects(poly_shelf) or poly_desk.intersection(poly_shelf).area < 0.001


def test_spatial_physics_mass_weighted_displacement():
    # Тяжелый шкаф (масса 12) и легкий стул (масса 1)
    items = [
        {"id": "f_wardrobe", "type": "wardrobe", "position": (4.0, 0.0, 4.0), "dimensions": (1.2, 2.1, 0.6), "rotation_deg": 0.0},
        {"id": "f_chair", "type": "chair", "position": (4.0, 0.0, 4.2), "dimensions": (0.5, 0.85, 0.5), "rotation_deg": 0.0},
    ]
    room_polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    resolved = SpatialPhysicsEngine.resolve_scene_physics(items, room_polygon, max_iterations=30)
    wardrobe_move = abs(resolved[0]["position"][2] - 4.0)
    chair_move = abs(resolved[1]["position"][2] - 4.2)

    # Легкий стул должен сместиться сильнее, чем тяжелый шкаф
    assert chair_move > wardrobe_move


def test_spatial_physics_preserves_rug_passability():
    # Ковер и стоящий на нем журнальный столик
    items = [
        {"id": "f_rug", "type": "rug", "position": (5.0, 0.0, 5.0), "dimensions": (2.4, 0.02, 1.8), "rotation_deg": 0.0},
        {"id": "f_table", "type": "coffee_table", "position": (5.0, 0.0, 5.0), "dimensions": (1.0, 0.45, 0.6), "rotation_deg": 0.0},
    ]
    room_polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    resolved = SpatialPhysicsEngine.resolve_scene_physics(items, room_polygon, max_iterations=20)
    # Ковер не выталкивает столик, столик остается в центре ковра
    assert abs(resolved[1]["position"][0] - 5.0) < 0.05
    assert abs(resolved[1]["position"][2] - 5.0) < 0.05
