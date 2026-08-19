"""
Тесты алгоритма расстановки мебели (app/agents/furniture_planner/agent.py).
Тестируем только чистую геометрическую часть — подбор мебели через LLM
(_select_furniture_set) не тестируется здесь, это требует мока Groq.
"""
from app.agents.furniture_planner.agent import (
    MIN_CLEARANCE_M,
    _door_positions,
    _place_along_perimeter,
    _room_bbox,
)
from app.models.scene import Opening, OpeningType, Room, RoomType, Wall


def _make_square_room(size: float = 5.0) -> Room:
    return Room(
        id="room_1",
        type=RoomType.living_room,
        polygon=[(0, 0), (size, 0), (size, size), (0, size)],
        walls=[
            Wall(id="w1", start=(0, 0), end=(size, 0)),
            Wall(
                id="w2",
                start=(size, 0),
                end=(size, size),
                openings=[Opening(type=OpeningType.door, position=0.5, width=0.9)],
            ),
            Wall(id="w3", start=(size, size), end=(0, size)),
            Wall(id="w4", start=(0, size), end=(0, 0)),
        ],
    )


def test_room_bbox_matches_polygon_extent():
    room = _make_square_room(5.0)
    min_x, min_y, max_x, max_y = _room_bbox(room)
    assert (min_x, min_y, max_x, max_y) == (0, 0, 5, 5)


def test_door_positions_extracted_from_openings():
    room = _make_square_room(5.0)
    doors = _door_positions(room)
    assert len(doors) == 1
    # Дверь на середине правой стены (5,0)->(5,5) => (5, 2.5)
    assert doors[0] == (5.0, 2.5)


def test_placed_furniture_stays_within_room_bounds():
    room = _make_square_room(5.0)
    items = [
        {"type": "sofa", "dimensions_m": [2.0, 0.85, 0.9], "material": "grey_fabric", "color": None},
        {"type": "tv_stand", "dimensions_m": [1.2, 0.5, 0.4], "material": "oak_wood", "color": None},
    ]
    placed = _place_along_perimeter(room, items)

    assert len(placed) == 2
    for item in placed:
        x, _, z = item.position
        w, _, d = item.dimensions
        assert 0 - 0.5 <= x - w / 2 and x + w / 2 <= 5 + 0.5
        assert 0 - 0.5 <= z - d / 2 and z + d / 2 <= 5 + 0.5


def test_placed_furniture_keeps_clearance_from_each_other():
    room = _make_square_room(6.0)
    items = [
        {"type": "sofa", "dimensions_m": [1.5, 0.85, 0.9]},
        {"type": "armchair", "dimensions_m": [0.9, 0.9, 0.9]},
    ]
    placed = _place_along_perimeter(room, items)
    assert len(placed) == 2
    x1 = placed[0].position[0]
    x2 = placed[1].position[0]
    w1 = placed[0].dimensions[0]
    w2 = placed[1].dimensions[0]
    gap = x2 - (x1 + w1 / 2) - w2 / 2
    assert gap >= MIN_CLEARANCE_M - 0.05  # небольшой допуск на округление
