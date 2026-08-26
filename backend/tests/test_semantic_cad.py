"""
Тесты семантического моста и CAD-компилятора расстановки мебели (Decoupled Semantic CAD).
"""
import pytest
from app.agents.furniture_planner.compiler import compile_semantic_layout_to_3d
from app.cv.semantic_bridge import generate_semantic_room_brief
from app.cv.wall_graph import GraphOpening, RoomFace, WallEdge
from app.models.scene import Opening, OpeningType, Room, RoomType, Wall


def test_semantic_bridge_room_brief_extraction():
    """Проверяет генерацию семантического брифа в метрах без float-координат."""
    wall_north = WallEdge(
        id="w_north",
        start=(0.0, 0.0),
        end=(5.0, 0.0),
        openings=[GraphOpening(type="window", position=0.5, width_m=1.8)],
        is_exterior=True,
    )
    wall_east = WallEdge(id="w_east", start=(5.0, 0.0), end=(5.0, 4.0), openings=[])
    wall_south = WallEdge(id="w_south", start=(5.0, 4.0), end=(0.0, 4.0), openings=[])
    wall_west = WallEdge(
        id="w_west",
        start=(0.0, 4.0),
        end=(0.0, 0.0),
        openings=[GraphOpening(type="door", position=0.5, width_m=0.9)],
    )

    room = RoomFace(
        id="room_1",
        polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
        walls=[wall_north, wall_east, wall_south, wall_west],
        area_sqm=20.0,
    )

    brief = generate_semantic_room_brief(room, room_type="living_room")
    assert brief.width_m == 5.0
    assert brief.depth_m == 4.0
    assert brief.area_sqm == 20.0
    assert len(brief.walls) == 4

    # Проверяем наличие окна на северной стене и двери на западной
    north_wall = next(w for w in brief.walls if "north" in w.orientation)
    assert north_wall.has_window is True
    assert any("window" in f for f in north_wall.features)

    west_wall = next(w for w in brief.walls if "west" in w.orientation)
    assert west_wall.has_door is True


def test_cad_compiler_places_against_anchor_walls():
    """Проверяет точное размещение мебели по привязкам к стенам (anchor_wall)."""
    room = Room(
        id="room_1",
        name="Living Room",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)],
        walls=[
            Wall(id="w1", start=(0.0, 0.0), end=(6.0, 0.0)),
            Wall(id="w2", start=(6.0, 0.0), end=(6.0, 4.0)),
            Wall(id="w3", start=(6.0, 4.0), end=(0.0, 4.0)),
            Wall(id="w4", start=(0.0, 4.0), end=(0.0, 0.0)),
        ],
        area_sqm=24.0,
    )

    semantic_items = [
        {
            "type": "sofa",
            "anchor_wall": "south",
            "placement": "center",
            "distance_from_wall_cm": 10,
            "dimensions_cm": {"width": 200, "height": 85, "depth": 90},
            "material": "fabric",
            "color": "#333333",
        },
        {
            "type": "tv_stand",
            "anchor_wall": "north",
            "placement": "center",
            "distance_from_wall_cm": 5,
            "dimensions_cm": {"width": 180, "height": 45, "depth": 40},
            "material": "wood",
            "color": "#8B5A2B",
        },
    ]

    placed = compile_semantic_layout_to_3d(room, semantic_items)
    assert len(placed) == 2

    sofa = next(p for p in placed if p.type == "sofa")
    # Диван привязан к южной стене (y = 4.0) -> y должен быть около 4.0 - 0.45 - 0.10 = 3.45
    assert 3.2 <= sofa.position[2] <= 3.8
    assert abs(sofa.position[0] - 3.0) < 0.5  # по центру x = 3.0

    tv = next(p for p in placed if p.type == "tv_stand")
    # ТВ привязан к северной стене (y = 0.0) -> y около 0.0 + 0.20 + 0.05 = 0.25
    assert 0.15 <= tv.position[2] <= 0.60


def test_cad_compiler_respects_door_clearance():
    """Проверяет, что мебель автоматически отодвигается от дверного проёма."""
    room = Room(
        id="room_1",
        name="Bedroom",
        type=RoomType.bedroom,
        polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        walls=[
            Wall(
                id="w1",
                start=(0.0, 0.0),
                end=(4.0, 0.0),
                openings=[Opening(type=OpeningType.door, position=0.5, width=0.9)],  # Дверь в центре северной стены (2.0, 0.0)
            ),
            Wall(id="w2", start=(4.0, 0.0), end=(4.0, 4.0)),
            Wall(id="w3", start=(4.0, 4.0), end=(0.0, 4.0)),
            Wall(id="w4", start=(0.0, 4.0), end=(0.0, 0.0)),
        ],
        area_sqm=16.0,
    )

    # Пытаемся поставить шкаф прямо перед дверью
    semantic_items = [
        {
            "type": "wardrobe",
            "anchor_wall": "north",
            "placement": "center",
            "distance_from_wall_cm": 0,
            "dimensions_cm": {"width": 120, "height": 200, "depth": 60},
        }
    ]

    placed = compile_semantic_layout_to_3d(room, semantic_items)
    assert len(placed) == 1
    wardrobe = placed[0]

    # Расстояние от центра двери (2.0, 0.0) до шкафа должно быть >= 0.80м
    door_x, door_y = 2.0, 0.0
    dist = ((wardrobe.position[0] - door_x) ** 2 + (wardrobe.position[2] - door_y) ** 2) ** 0.5
    assert dist >= 0.80, f"Шкаф слишком близко к двери: {dist}м"

