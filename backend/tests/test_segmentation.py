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


def _draw_multi_room_apartment() -> bytes:
    """Генерирует многокомнатный чертёж с 4 комнатами (гостиная, кухня, спальня, санузел)."""
    img = np.full((500, 600), 255, dtype=np.uint8)
    thickness = 6

    # Внешний периметр: (50, 50) до (550, 450)
    cv2.rectangle(img, (50, 50), (550, 450), 0, thickness)

    # Внутренняя вертикальная перегородка x=300 от y=50 до y=300 (с дверным проёмом)
    cv2.line(img, (300, 50), (300, 150), 0, thickness)
    cv2.line(img, (300, 220), (300, 300), 0, thickness)

    # Внутренняя горизонтальная перегородка y=300 от x=50 до x=550 (с дверным проёмом)
    cv2.line(img, (50, 300), (200, 300), 0, thickness)
    cv2.line(img, (260, 300), (550, 300), 0, thickness)

    # Санузел в правом нижнем углу: перегородка x=420 от y=300 до y=450
    cv2.line(img, (420, 300), (420, 370), 0, thickness)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_detects_exactly_one_room():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    assert len(result.room_polygons) == 1


def test_multi_room_floorplan_detects_multiple_rooms():
    result = segment_floor_plan(_draw_multi_room_apartment())
    # Должно быть обнаружено от 3 до 4 раздельных комнат
    assert len(result.room_polygons) >= 3
    # Проверяем, что у каждой комнаты есть замкнутые стены
    for room in result.room_polygons:
        assert len(room.points) >= 4
        assert len(room.walls) >= 4


def test_room_bounding_box_matches_drawn_walls():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    room = result.room_polygons[0]
    xs = [p[0] for p in room.points]
    ys = [p[1] for p in room.points]
    # Стены нарисованы в px (50,50)-(250,200) => в метрах (50/ppm, 50/ppm)-(250/ppm, 200/ppm)
    ppm = result.pixels_per_meter
    assert min(xs) == pytest.approx(50.0 / ppm, abs=0.2)
    assert max(xs) == pytest.approx(250.0 / ppm, abs=0.2)
    assert min(ys) == pytest.approx(50.0 / ppm, abs=0.2)
    assert max(ys) == pytest.approx(200.0 / ppm, abs=0.2)


def test_detects_door_gap_in_wall():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps())
    room = result.room_polygons[0]
    all_openings = [o for wall in room.walls for o in wall.openings]
    assert len(all_openings) >= 1
    # Разрыв 100..140px = 0.8м, должен попасть в диапазон обнаружения.
    widths = [o.width_m for o in all_openings]
    assert any(0.5 <= w <= 1.5 for w in widths)


def test_no_openings_when_walls_are_solid():
    result = segment_floor_plan(_draw_rectangular_room_with_gaps(door_gap_px=None, window_gap_px=None))
    room = result.room_polygons[0]
    all_openings = [o for wall in room.walls for o in wall.openings]
    assert all_openings == []


def test_distinguishes_exterior_windows_and_interior_doors():
    """Проверяет, что система безошибочно определяет фасадные проёмы как window,
    а межкомнатные как door."""
    img = np.full((400, 500), 255, dtype=np.uint8)
    thickness = 6

    # Внешний периметр (50, 50) до (450, 350)
    # На верхней фасадной стене — окно разрывом (180..280)
    cv2.line(img, (50, 50), (180, 50), 0, thickness)
    cv2.line(img, (280, 50), (450, 50), 0, thickness)
    cv2.line(img, (450, 50), (450, 350), 0, thickness)
    cv2.line(img, (50, 350), (450, 350), 0, thickness)
    cv2.line(img, (50, 50), (50, 350), 0, thickness)

    # Внутренняя межкомнатная стена x=250 от y=50 до y=350 с дверным проёмом (y=150..230)
    cv2.line(img, (250, 50), (250, 150), 0, thickness)
    cv2.line(img, (250, 230), (250, 350), 0, thickness)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = segment_floor_plan(buf.tobytes())

    all_openings = [o for r in result.room_polygons for w in r.walls for o in w.openings]
    types = {o.type for o in all_openings}

    # В результате должны быть и окна, и двери
    assert "window" in types, "Фасадный проём должен быть распознан как окно (window)"
    assert "door" in types, "Межкомнатный проём должен быть распознан как дверь (door)"


def test_wide_interior_opening_is_classified_as_door():
    """Широкий межкомнатный проём (1.6м) всё равно должен быть дверью/проходом, а не окном."""
    img = np.full((400, 500), 255, dtype=np.uint8)
    thickness = 6

    # Сплошной внешний периметр
    cv2.rectangle(img, (50, 50), (450, 350), 0, thickness)

    # Внутренняя стена с широким проёмом 80px = 1.6м (y=140..220)
    cv2.line(img, (250, 50), (250, 140), 0, thickness)
    cv2.line(img, (250, 220), (250, 350), 0, thickness)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = segment_floor_plan(buf.tobytes())

    wall_x_m = 250.0 / result.pixels_per_meter
    interior_openings = [
        o for r in result.room_polygons for w in r.walls if abs(w.start[0] - wall_x_m) < 0.5 and abs(w.end[0] - wall_x_m) < 0.5 for o in w.openings
    ]
    assert len(interior_openings) >= 1
    for o in interior_openings:
        assert o.type == "door", f"Межкомнатный проём не должен быть окном: {o}"


def test_narrow_exterior_opening_is_classified_as_window():
    """Узкий проём на фасадной стене (0.8м) должен классифицироваться как окно."""
    img = np.full((300, 300), 255, dtype=np.uint8)
    thickness = 6

    # Окно шириной 40px = 0.8м на верхней фасадной стене
    cv2.line(img, (50, 50), (130, 50), 0, thickness)
    cv2.line(img, (170, 50), (250, 50), 0, thickness)
    cv2.line(img, (250, 50), (250, 250), 0, thickness)
    cv2.line(img, (50, 250), (250, 250), 0, thickness)
    cv2.line(img, (50, 50), (50, 250), 0, thickness)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = segment_floor_plan(buf.tobytes())

    exterior_openings = [
        o for r in result.room_polygons for w in r.walls if abs(w.start[1] - 1.0) < 0.3 and abs(w.end[1] - 1.0) < 0.3 for o in w.openings
    ]
    assert len(exterior_openings) >= 1
    for o in exterior_openings:
        assert o.type == "window", f"Фасадный проём должен быть окном: {o}"


def test_empty_image_returns_no_rooms():
    img = np.full((200, 200), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    result = segment_floor_plan(buf.tobytes())
    assert result.room_polygons == []
    assert result.quality_score is not None
    assert result.quality_score.overall_score == 0.0
    assert result.quality_score.is_valid is False


def test_invalid_bytes_raise_value_error():
    with pytest.raises(ValueError):
        segment_floor_plan(b"not an image")


def test_cv_quality_score_is_calculated_and_valid():
    """Проверяет, что качественный чертеж получает высокий скор (> 85%) и статус is_valid=True."""
    result = segment_floor_plan(_draw_multi_room_apartment())
    score = result.quality_score
    assert score is not None
    assert score.overall_score >= 0.80
    assert score.overlap_score == pytest.approx(1.0, abs=0.05)
    assert score.coverage_score >= 0.70
    assert score.is_valid is True


def test_cv_validation_flags_overlapping_rooms():
    """Проверяет, что система валидации строго ловит пересечения комнат (ROOM_OVERLAP)."""
    from app.cv.segmentation import evaluate_segmentation_quality, RoomPolygon, WallSegment

    # Создаём две пересекающиеся комнаты
    room1 = RoomPolygon(
        points=[(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)],
        walls=[WallSegment(start=(1.0, 1.0), end=(6.0, 1.0))]
    )
    room2 = RoomPolygon(
        points=[(3.0, 3.0), (8.0, 3.0), (8.0, 8.0), (3.0, 8.0)],  # Пересечение 3x3=9 кв.м
        walls=[WallSegment(start=(3.0, 3.0), end=(8.0, 3.0))]
    )

    score = evaluate_segmentation_quality([room1, room2], ox_m=1.0, oy_m=1.0, ow_m=7.0, oh_m=7.0)
    assert score.overlap_score < 0.80
    assert score.is_valid is False
    issue_codes = [iss.code for iss in score.issues]
    assert "ROOM_OVERLAP" in issue_codes


def test_cv_validation_flags_rooms_missing_doors():
    """Проверяет, что комната без входной двери отмечается предупреждением NO_DOOR."""
    from app.cv.segmentation import evaluate_segmentation_quality, RoomPolygon, WallSegment, DetectedOpening

    room_with_door = RoomPolygon(
        points=[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)],
        walls=[WallSegment(start=(1.0, 1.0), end=(5.0, 1.0), openings=[DetectedOpening(type="door", position=0.5, width_m=0.9)])]
    )
    room_without_door = RoomPolygon(
        points=[(5.0, 1.0), (9.0, 1.0), (9.0, 5.0), (5.0, 5.0)],
        walls=[WallSegment(start=(5.0, 1.0), end=(9.0, 1.0), openings=[])]  # Сплошные стены
    )

    score = evaluate_segmentation_quality([room_with_door, room_without_door], ox_m=1.0, oy_m=1.0, ow_m=8.0, oh_m=4.0)
    assert score.connectivity_score == pytest.approx(0.5, abs=0.01)
    issue_codes = [iss.code for iss in score.issues]
    assert "NO_DOOR" in issue_codes


def test_cv_validation_flags_extreme_aspect_ratios():
    """Проверяет детекцию нереалистично узких комнат (EXTREME_ASPECT)."""
    from app.cv.segmentation import evaluate_segmentation_quality, RoomPolygon, WallSegment, DetectedOpening

    # Узкая полоса 10м x 1м (aspect ratio 10.0)
    narrow_room = RoomPolygon(
        points=[(1.0, 1.0), (11.0, 1.0), (11.0, 2.0), (1.0, 2.0)],
        walls=[WallSegment(start=(1.0, 1.0), end=(11.0, 1.0), openings=[DetectedOpening(type="door", position=0.5, width_m=0.9)])]
    )

    score = evaluate_segmentation_quality([narrow_room], ox_m=1.0, oy_m=1.0, ow_m=10.0, oh_m=1.0)
    assert score.geometry_score < 0.90
    issue_codes = [iss.code for iss in score.issues]
    assert "EXTREME_ASPECT" in issue_codes or "TINY_DIMENSION" in issue_codes



