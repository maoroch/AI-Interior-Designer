"""
Тесты CV-сегментации плана (app/cv/segmentation.py) — чистые, без сети/LLM/БД.
Строим синтетический план из линий OpenCV с известными разрывами (дверь/окно)
и проверяем, что геометрия и проёмы восстанавливаются с разумной точностью.
"""
import cv2
import numpy as np
import pytest

from app.cv.segmentation import segment_floor_plan


def _draw_rectangular_room_with_gaps(
    door_gap_px: tuple[int, int] | None = (100, 140),
    window_gap_px: tuple[int, int] | None = (100, 160),
) -> bytes:
    """Комната 4x3м (200x150px при 50px/м) с опциональным разрывом-дверью в нижней
    стене и разрывом-окном в правой стене."""
    img = np.full((250, 300), 255, dtype=np.uint8)
    thickness = 4

    cv2.line(img, (50, 50), (250, 50), 0, thickness)  # верх
    cv2.line(img, (50, 50), (50, 200), 0, thickness)  # лево

    if door_gap_px:
        cv2.line(img, (50, 200), (door_gap_px[0], 200), 0, thickness)
        cv2.line(img, (door_gap_px[1], 200), (250, 200), 0, thickness)
    else:
        cv2.line(img, (50, 200), (250, 200), 0, thickness)

    if window_gap_px:
        cv2.line(img, (250, 50), (250, window_gap_px[0]), 0, thickness)
        cv2.line(img, (250, window_gap_px[1]), (250, 200), 0, thickness)
    else:
        cv2.line(img, (250, 50), (250, 200), 0, thickness)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_detects_exactly_one_room():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    assert len(result.room_polygons) == 1


def test_room_bounding_box_matches_drawn_walls():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    room = result.room_polygons[0]
    xs = [p[0] for p in room.points]
    ys = [p[1] for p in room.points]
    # Стены нарисованы в px (50,50)-(250,200) => в метрах (1,1)-(5,4) при 50px/м.
    assert min(xs) == pytest.approx(1.0, abs=0.2)
    assert max(xs) == pytest.approx(5.0, abs=0.2)
    assert min(ys) == pytest.approx(1.0, abs=0.2)
    assert max(ys) == pytest.approx(4.0, abs=0.2)


def test_detects_door_gap_in_wall():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    room = result.room_polygons[0]
    all_openings = [o for wall in room.walls for o in wall.openings]
    assert len(all_openings) >= 1
    # Разрыв 100..140px = 0.8м, должен попасть в диапазон обнаружения.
    widths = [o.width_m for o in all_openings]
    assert any(0.5 <= w <= 1.2 for w in widths)


def test_no_openings_when_walls_are_solid():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps(door_gap_px=None, window_gap_px=None))
    room = result.room_polygons[0]
    all_openings = [o for wall in room.walls for o in wall.openings]
    assert all_openings == []


def test_empty_image_returns_no_rooms():
    img = np.full((200, 200), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    result = segment_floor_plan(buf.tobytes())
    assert result.room_polygons == []


def test_invalid_bytes_raise_value_error():
    with pytest.raises(ValueError):
        segment_floor_plan(b"not an image")
