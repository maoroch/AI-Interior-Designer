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


def _is_exterior_wall_segment(
    start_m: Point2D,
    end_m: Point2D,
    ox_m: float,
    oy_m: float,
    ow_m: float,
    oh_m: float,
    tol: float = 0.6,
) -> bool:
    """Определяет, является ли стена фасадной (внешней) или межкомнатной перегородкой."""
    along_top = abs(start_m[1] - oy_m) < tol and abs(end_m[1] - oy_m) < tol
    along_bottom = abs(start_m[1] - (oy_m + oh_m)) < tol and abs(end_m[1] - (oy_m + oh_m)) < tol
    along_left = abs(start_m[0] - ox_m) < tol and abs(end_m[0] - ox_m) < tol
    along_right = abs(start_m[0] - (ox_m + ow_m)) < tol and abs(end_m[0] - (ox_m + ow_m)) < tol
    return along_top or along_bottom or along_left or along_right


def _detect_openings(
    image: np.ndarray,
    start_m: Point2D,
    end_m: Point2D,
    pixels_per_meter: float,
    is_exterior: bool = False,
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
    for i, is_wall in enumerate(is_wall_samples + [True]):
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
                # Внутренние стены между комнатами — ВСЕГДА двери (проходы).
                # Внешние фасадные стены — ОКНА.
                opening_type = "window" if is_exterior else "door"
                openings.append(
                    DetectedOpening(
                        type=opening_type,
                        position=round((frac_start + frac_end) / 2, 3),
                        width_m=round(width_m, 2),
                    )
                )
            gap_start_idx = None

    return openings


def segment_floor_plan(image_bytes: bytes) -> SegmentationResult:
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Не удалось декодировать изображение плана")

    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    pixels_per_meter = 50.0

    # 1. Бинаризация: тёмные пиксели стен (< 140)
    _, wall_mask = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)

    # Убираем тонкие размерные надписи (например, 12.0m, 9.5m), чтобы получить истинный внешний периметр здания
    structural_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)))
    coords = cv2.findNonZero(structural_walls)
    if coords is None:
        coords = cv2.findNonZero(wall_mask)
    if coords is None:
        return SegmentationResult(
            room_polygons=[], image_width=width, image_height=height, pixels_per_meter=pixels_per_meter
        )

    ox, oy, ow, oh = cv2.boundingRect(coords)
    if ow * oh < (width * height) * 0.01:
        return SegmentationResult(
            room_polygons=[], image_width=width, image_height=height, pixels_per_meter=pixels_per_meter
        )

    ox_m = ox / pixels_per_meter
    oy_m = oy / pixels_per_meter
    ow_m = ow / pixels_per_meter
    oh_m = oh / pixels_per_meter

    # 2. Детекция синих/голубых оконных блоков (Color Window Detection)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([135, 255, 255]))
    blue_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_windows_boxes: list[tuple[int, int, int, int]] = []
    for c in blue_cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw >= 25 or bh >= 25:
            detected_windows_boxes.append((bx, by, bw, bh))

    # 3. Выделяем прямые горизонтальные и вертикальные линии стен
    h_walls = cv2.morphologyEx(structural_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3)))
    v_walls = cv2.morphologyEx(structural_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 16)))

    # Продление линий для перекрытия проёмов
    h_ext = cv2.dilate(h_walls, cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1)))
    v_ext = cv2.dilate(v_walls, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150)))

    grid = cv2.bitwise_or(h_ext, v_ext)
    cv2.rectangle(grid, (ox, oy), (ox + ow, oy + oh), 255, 12)

    # 4. Маскируем внутреннее пространство квартиры (строго внутри несущих стен)
    inside_mask = np.zeros_like(wall_mask)
    inside_mask[oy + 10 : oy + oh - 10, ox + 10 : ox + ow - 10] = 255
    rooms_space = cv2.bitwise_and(cv2.bitwise_not(grid), inside_mask)

    # 5. Находим связные компоненты внутренних комнат
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rooms_space, connectivity=4)

    min_area_px = int(pixels_per_meter * pixels_per_meter * 3.8)
    max_area_px = int(pixels_per_meter * pixels_per_meter * 180.0)

    valid_labels = []
    for l in range(1, num_labels):
        area = stats[l, cv2.CC_STAT_AREA]
        rw = stats[l, cv2.CC_STAT_WIDTH]
        rh = stats[l, cv2.CC_STAT_HEIGHT]
        aspect = max(rw / max(1, rh), rh / max(1, rw))
        rx = stats[l, cv2.CC_STAT_LEFT]
        ry = stats[l, cv2.CC_STAT_TOP]

        if (
            min_area_px <= area <= max_area_px
            and rw >= 60
            and rh >= 60
            and aspect <= 3.2
            and rx >= ox
            and ry >= oy
            and (rx + rw) <= (ox + ow + 5)
            and (ry + rh) <= (oy + oh + 5)
        ):
            valid_labels.append(l)

    # Если комната одна (студия или синтетический тест без внутренних стен)
    if len(valid_labels) <= 1:
        corners_px = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
        pts_m = [(round(px / pixels_per_meter, 2), round(py / pixels_per_meter, 2)) for px, py in corners_px]
        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            is_ext = _is_exterior_wall_segment(start_m, end_m, ox_m, oy_m, ow_m, oh_m)
            openings = _detect_openings(image, start_m, end_m, pixels_per_meter, is_exterior=is_ext)
            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))
        return SegmentationResult(
            room_polygons=[RoomPolygon(points=pts_m, walls=walls)],
            image_width=width,
            image_height=height,
            pixels_per_meter=pixels_per_meter,
        )

    raw_boxes: list[list[int]] = []
    for label in valid_labels:
        bx = stats[label, cv2.CC_STAT_LEFT]
        by = stats[label, cv2.CC_STAT_TOP]
        bw_room = stats[label, cv2.CC_STAT_WIDTH]
        bh_room = stats[label, cv2.CC_STAT_HEIGHT]

        if abs(bx - ox) < 25:
            bx = ox
        if abs(by - oy) < 25:
            by = oy
        if abs((bx + bw_room) - (ox + ow)) < 25:
            bw_room = (ox + ow) - bx
        if abs((by + bh_room) - (oy + oh)) < 25:
            bh_room = (oy + oh) - by

        raw_boxes.append([int(bx), int(by), int(bw_room), int(bh_room)])

    # Привязка смежных межкомнатных перегородок
    for i in range(len(raw_boxes)):
        for j in range(i + 1, len(raw_boxes)):
            b1, b2 = raw_boxes[i], raw_boxes[j]
            r1, l2 = b1[0] + b1[2], b2[0]
            if 0 <= (l2 - r1) <= 35:
                mid_x = (r1 + l2) // 2
                b1[2] = mid_x - b1[0]
                b2[2] = (b2[0] + b2[2]) - mid_x
                b2[0] = mid_x
            bot1, top2 = b1[1] + b1[3], b2[1]
            if 0 <= (top2 - bot1) <= 35:
                mid_y = (bot1 + top2) // 2
                b1[3] = mid_y - b1[1]
                b2[3] = (b2[1] + b2[3]) - mid_y
                b2[1] = mid_y

    detected_polygons: list[RoomPolygon] = []
    for bx, by, bw_room, bh_room in raw_boxes:
        pts_m = [
            (round(bx / pixels_per_meter, 2), round(by / pixels_per_meter, 2)),
            (round((bx + bw_room) / pixels_per_meter, 2), round(by / pixels_per_meter, 2)),
            (round((bx + bw_room) / pixels_per_meter, 2), round((by + bh_room) / pixels_per_meter, 2)),
            (round(bx / pixels_per_meter, 2), round((by + bh_room) / pixels_per_meter, 2)),
        ]

        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            s_px = (start_m[0] * pixels_per_meter, start_m[1] * pixels_per_meter)
            e_px = (end_m[0] * pixels_per_meter, end_m[1] * pixels_per_meter)

            openings: list[DetectedOpening] = []
            # 1. Проверяем синие окна вдоль данного сегмента стены
            for wx, wy, ww, wh in detected_windows_boxes:
                w_mid_x = wx + ww / 2
                w_mid_y = wy + wh / 2
                # Горизонтальная стена
                if (
                    abs(s_px[1] - e_px[1]) < 10
                    and min(s_px[0], e_px[0]) <= w_mid_x <= max(s_px[0], e_px[0])
                    and abs(s_px[1] - w_mid_y) < 25
                ):
                    pos = (w_mid_x - min(s_px[0], e_px[0])) / max(1.0, abs(e_px[0] - s_px[0]))
                    if s_px[0] > e_px[0]:
                        pos = 1.0 - pos
                    openings.append(
                        DetectedOpening(
                            type="window",
                            position=round(pos, 3),
                            width_m=round(max(ww, wh) / pixels_per_meter, 2),
                        )
                    )
                # Вертикальная стена
                elif (
                    abs(s_px[0] - e_px[0]) < 10
                    and min(s_px[1], e_px[1]) <= w_mid_y <= max(s_px[1], e_px[1])
                    and abs(s_px[0] - w_mid_x) < 25
                ):
                    pos = (w_mid_y - min(s_px[1], e_px[1])) / max(1.0, abs(e_px[1] - s_px[1]))
                    if s_px[1] > e_px[1]:
                        pos = 1.0 - pos
                    openings.append(
                        DetectedOpening(
                            type="window",
                            position=round(pos, 3),
                            width_m=round(max(ww, wh) / pixels_per_meter, 2),
                        )
                    )

            # 2. Если синих окон нет — выполняем поиск дверных проёмов и разрывов
            if not openings:
                is_ext = _is_exterior_wall_segment(start_m, end_m, ox_m, oy_m, ow_m, oh_m)
                detected = _detect_openings(image, start_m, end_m, pixels_per_meter, is_exterior=is_ext)
                for d in detected:
                    # Входная дверь на нижней стене (вход в квартиру)
                    if (
                        abs(start_m[1] - (oy_m + oh_m)) < 0.5
                        and abs(end_m[1] - (oy_m + oh_m)) < 0.5
                        and min(start_m[0], end_m[0]) < ox_m + 3.5
                    ):
                        d.type = "door"
                    openings.append(d)

            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))

        detected_polygons.append(RoomPolygon(points=pts_m, walls=walls))

    return SegmentationResult(
        room_polygons=detected_polygons,
        image_width=width,
        image_height=height,
        pixels_per_meter=pixels_per_meter,
    )





