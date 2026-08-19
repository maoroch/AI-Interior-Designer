"""
CV-детекция геометрии плана.

СЕЙЧАС: упрощённые OpenCV-эвристики, чтобы пайплайн работал end-to-end на
MVP-этапе без обученной модели. Осознанные упрощения (важно понимать при
дальнейшей разработке):

1. Комната аппроксимируется ПРЯМОУГОЛЬНИКОМ по bounding-box пикселей стен,
   а не произвольным многоугольником. Причина: попытки честно восстановить
   многоугольник через findContours/approxPolyDP оказались очень
   чувствительны к разрывам стен (двери/окна) — контур "утекает" в эти
   разрывы и approxPolyDP либо даёт зубчатый многоугольник, либо, при
   агрессивном сглаживании, съезжает в сторону (проверено эмпирически на
   синтетических тестах). Bounding box по сырым пикселям стен устойчив к
   разрывам, т.к. они не меняют крайние координаты стен. Покрывает
   подавляющее большинство реальных квартир (прямоугольные комнаты);
   сложные формы — TODO для будущей сегментационной модели.
2. Многокомнатные планы пока не разделяются на отдельные комнаты —
   RETR_EXTERNAL берёт только внешний контур всей группы стен. Настоящее
   разделение на комнаты требует анализа внутренних контуров/графа смежности
   и тоже отнесено к будущей модели (см. ниже).
3. Детекция проёмов (двери/окна) — эвристика на разрывах линии стены:
   вдоль каждой стороны прямоугольника сканируется исходное изображение;
   участки, где линия стены "прерывается" (пиксели светлые, как фон),
   считаются проёмом. Тип проёма определяется ГРУБО по ширине разрыва —
   настоящее различение требует распознавания символа дуги открывания
   (дверь) или двойной линии (окно), что по силам только обученной модели.

ДАЛЬШЕ: заменить `segment_floor_plan` на инференс сегментационной модели
(U-Net/DeepLabV3+, обучена на датасете в духе CubiCasa5K) — сигнатура функции
не меняется, поэтому замена не затронет agents/floorplan_analyzer.
Перед использованием готовых весов — проверить лицензию (часть публичных
репозиториев с весами CubiCasa5K распространяется non-commercial-only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

Point2D = tuple[float, float]

WALL_INTENSITY_THRESHOLD = 200  # пиксели темнее — считаем "стена/линия"
MIN_OPENING_WIDTH_M = 0.5
MAX_OPENING_WIDTH_M = 3.0
DOOR_MAX_WIDTH_M = 1.3  # уже — дверь, шире — окно (грубая эвристика, см. докстринг)
MAX_OPENING_FRACTION_OF_WALL = 0.85  # если "проём" почти во всю стену — это не проём, а артефакт


@dataclass
class DetectedOpening:
    type: str  # "door" | "window"
    position: float  # 0..1 вдоль стены
    width_m: float


@dataclass
class WallSegment:
    start: Point2D
    end: Point2D
    openings: list[DetectedOpening] = field(default_factory=list)


@dataclass
class RoomPolygon:
    points: list[Point2D]
    walls: list[WallSegment]


@dataclass
class SegmentationResult:
    room_polygons: list[RoomPolygon]
    image_width: int
    image_height: int
    pixels_per_meter: float = 50.0  # заглушка масштаба; в реале — из LLM-разметки/аннотаций


def _polygon_to_wall_segments(points: list[Point2D]) -> list[tuple[Point2D, Point2D]]:
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _sample_is_wall(image: np.ndarray, x_px: float, y_px: float) -> bool:
    h, w = image.shape
    xi, yi = int(round(x_px)), int(round(y_px))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    # Небольшая окрестность вместо одного пикселя — устойчивее к сглаживанию/шуму скана.
    x0, x1 = max(0, xi - 1), min(w, xi + 2)
    y0, y1 = max(0, yi - 1), min(h, yi + 2)
    patch = image[y0:y1, x0:x1]
    return float(patch.mean()) < WALL_INTENSITY_THRESHOLD


def _detect_openings(
    image: np.ndarray, start_m: Point2D, end_m: Point2D, pixels_per_meter: float
) -> list[DetectedOpening]:
    start_px = (start_m[0] * pixels_per_meter, start_m[1] * pixels_per_meter)
    end_px = (end_m[0] * pixels_per_meter, end_m[1] * pixels_per_meter)
    length_m = math.hypot(end_m[0] - start_m[0], end_m[1] - start_m[1])
    if length_m < MIN_OPENING_WIDTH_M:
        return []

    n_samples = max(10, int(length_m * pixels_per_meter / 3))
    is_wall_samples = []
    for i in range(n_samples + 1):
        t = i / n_samples
        x_px = start_px[0] + (end_px[0] - start_px[0]) * t
        y_px = start_px[1] + (end_px[1] - start_px[1]) * t
        is_wall_samples.append(_sample_is_wall(image, x_px, y_px))

    # Группируем подряд идущие "не стена" (gap) участки в кандидаты на проём.
    openings: list[DetectedOpening] = []
    gap_start_idx: int | None = None
    for i, is_wall in enumerate(is_wall_samples + [True]):  # sentinel, чтобы закрыть последний gap
        if not is_wall and gap_start_idx is None:
            gap_start_idx = i
        elif is_wall and gap_start_idx is not None:
            gap_end_idx = i
            frac_start = gap_start_idx / n_samples
            frac_end = gap_end_idx / n_samples
            width_m = (frac_end - frac_start) * length_m

            if MIN_OPENING_WIDTH_M <= width_m <= MAX_OPENING_WIDTH_M and (
                frac_end - frac_start
            ) <= MAX_OPENING_FRACTION_OF_WALL:
                openings.append(
                    DetectedOpening(
                        type="door" if width_m <= DOOR_MAX_WIDTH_M else "window",
                        position=round((frac_start + frac_end) / 2, 3),
                        width_m=round(width_m, 2),
                    )
                )
            gap_start_idx = None

    return openings


def segment_floor_plan(image_bytes: bytes) -> SegmentationResult:
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Не удалось декодировать изображение плана")

    height, width = image.shape
    pixels_per_meter = 50.0

    # 1. Бинаризация: тёмные пиксели стен (< 140)
    _, wall_mask = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)

    wall_pixel_coords = cv2.findNonZero(wall_mask)
    if wall_pixel_coords is None:
        return SegmentationResult(
            room_polygons=[], image_width=width, image_height=height, pixels_per_meter=pixels_per_meter
        )

    ox, oy, ow, oh = cv2.boundingRect(wall_pixel_coords)
    if ow * oh < (width * height) * 0.01:
        return SegmentationResult(
            room_polygons=[], image_width=width, image_height=height, pixels_per_meter=pixels_per_meter
        )

    # 2. Выделяем прямые горизонтальные и вертикальные линии стен
    walls_clean = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))

    h_walls = cv2.morphologyEx(walls_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3)))
    v_walls = cv2.morphologyEx(walls_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20)))

    # 3. Направленное продление линий для перекрытия дверных проёмов (до 250px)
    h_ext = cv2.dilate(h_walls, cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1)))
    v_ext = cv2.dilate(v_walls, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 300)))

    grid = cv2.bitwise_or(h_ext, v_ext)
    cv2.rectangle(grid, (ox, oy), (ox + ow, oy + oh), 255, 12)

    # 4. Маскируем внутреннее пространство квартиры
    inside_mask = np.zeros_like(wall_mask)
    inside_mask[oy + 8 : oy + oh - 8, ox + 8 : ox + ow - 8] = 255
    rooms_space = cv2.bitwise_and(cv2.bitwise_not(grid), inside_mask)

    # 5. Находим связные компоненты внутренних комнат
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rooms_space, connectivity=4)

    min_area_px = int(pixels_per_meter * pixels_per_meter * 2.5)  # 2.5 кв.м
    max_area_px = int(pixels_per_meter * pixels_per_meter * 160.0) # 160 кв.м

    valid_labels = [
        l
        for l in range(1, num_labels)
        if stats[l, cv2.CC_STAT_AREA] >= min_area_px
        and stats[l, cv2.CC_STAT_WIDTH] >= 30
        and stats[l, cv2.CC_STAT_HEIGHT] >= 30
    ]

    # Если комната одна (студия или синтетический тест без внутренних стен)
    if len(valid_labels) <= 1:
        corners_px = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
        pts_m = [(round(px / pixels_per_meter, 2), round(py / pixels_per_meter, 2)) for px, py in corners_px]
        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            openings = _detect_openings(image, start_m, end_m, pixels_per_meter)
            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))
        return SegmentationResult(
            room_polygons=[RoomPolygon(points=pts_m, walls=walls)],
            image_width=width,
            image_height=height,
            pixels_per_meter=pixels_per_meter,
        )

    detected_polygons: list[RoomPolygon] = []

    for label in valid_labels:
        bx = stats[label, cv2.CC_STAT_LEFT]
        by = stats[label, cv2.CC_STAT_TOP]
        bw_room = stats[label, cv2.CC_STAT_WIDTH]
        bh_room = stats[label, cv2.CC_STAT_HEIGHT]

        # Привязка (snap) к внешним несущим стенам
        if abs(bx - ox) < 25:
            bx = ox
        if abs(by - oy) < 25:
            by = oy
        if abs((bx + bw_room) - (ox + ow)) < 25:
            bw_room = (ox + ow) - bx
        if abs((by + bh_room) - (oy + oh)) < 25:
            bh_room = (oy + oh) - by

        pts_m = [
            (round(bx / pixels_per_meter, 2), round(by / pixels_per_meter, 2)),
            (round((bx + bw_room) / pixels_per_meter, 2), round(by / pixels_per_meter, 2)),
            (round((bx + bw_room) / pixels_per_meter, 2), round((by + bh_room) / pixels_per_meter, 2)),
            (round(bx / pixels_per_meter, 2), round((by + bh_room) / pixels_per_meter, 2)),
        ]

        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            openings = _detect_openings(image, start_m, end_m, pixels_per_meter)
            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))

        detected_polygons.append(RoomPolygon(points=pts_m, walls=walls))

    return SegmentationResult(
        room_polygons=detected_polygons,
        image_width=width,
        image_height=height,
        pixels_per_meter=pixels_per_meter,
    )



