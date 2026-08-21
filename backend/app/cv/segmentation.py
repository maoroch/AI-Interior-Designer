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
    confidence: float = 0.90  # 0.0 .. 1.0 вероятность корректности
    features: list[str] = field(default_factory=list)


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
class ValidationIssue:
    severity: str  # "info" | "warning" | "error"
    code: str      # "ROOM_OVERLAP" | "NO_DOOR" | "NO_WINDOW" | "EXTREME_ASPECT" | "LOW_COVERAGE" | "TINY_DIMENSION"
    message: str
    room_index: int | None = None


@dataclass
class CVQualityScore:
    overall_score: float         # 0.0 .. 1.0 (например, 0.96)
    overlap_score: float         # 1.0 = отсутствие недопустимых наложений комнат
    coverage_score: float        # Доля полезной площади квартиры (0.0 .. 1.0)
    connectivity_score: float    # Доля комнат с дверными проходами
    geometry_score: float        # Ортогональность и регулярность стен
    issues: list[ValidationIssue] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class SegmentationResult:
    room_polygons: list[RoomPolygon]
    image_width: int
    image_height: int
    pixels_per_meter: float = 50.0  # заглушка масштаба; в реале — из LLM-разметки/аннотаций
    quality_score: CVQualityScore | None = None


def evaluate_segmentation_quality(
    room_polygons: list[RoomPolygon],
    ox_m: float,
    oy_m: float,
    ow_m: float,
    oh_m: float,
) -> CVQualityScore:
    """
    Система валидации и скоринга точности Computer Vision:
    1. Overlap Score (0..1) — проверка отсутствия пересечений и наложений комнат
    2. Coverage Score (0..1) — покрытие площади внешнего контура здания
    3. Connectivity Score (0..1) — наличие входных дверей и окон в помещениях
    4. Geometry Score (0..1) — регулярность пропорций (aspect ratio <= 3.2, мин. габариты >= 1.3м)
    """
    issues: list[ValidationIssue] = []
    n_rooms = len(room_polygons)
    if n_rooms == 0:
        return CVQualityScore(
            overall_score=0.0,
            overlap_score=0.0,
            coverage_score=0.0,
            connectivity_score=0.0,
            geometry_score=0.0,
            is_valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="NO_ROOMS",
                    message="Не обнаружено ни одной валидной комнаты",
                )
            ],
        )

    # 1. Overlap Score
    total_overlap_penalty = 0.0
    for i in range(n_rooms):
        pts1 = room_polygons[i].points
        xs1, ys1 = [p[0] for p in pts1], [p[1] for p in pts1]
        min_x1, max_x1, min_y1, max_y1 = min(xs1), max(xs1), min(ys1), max(ys1)
        area1 = (max_x1 - min_x1) * (max_y1 - min_y1)

        for j in range(i + 1, n_rooms):
            pts2 = room_polygons[j].points
            xs2, ys2 = [p[0] for p in pts2], [p[1] for p in pts2]
            min_x2, max_x2, min_y2, max_y2 = min(xs2), max(xs2), min(ys2), max(ys2)
            area2 = (max_x2 - min_x2) * (max_y2 - min_y2)

            x_ov = max(0.0, min(max_x1, max_x2) - max(min_x1, min_x2))
            y_ov = max(0.0, min(max_y1, max_y2) - max(min_y1, min_y2))
            ov_area = x_ov * y_ov
            ov_frac = ov_area / max(0.01, min(area1, area2))

            if ov_frac > 0.08:
                total_overlap_penalty += ov_frac
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ROOM_OVERLAP",
                        message=f"Пересечение комнат #{i+1} и #{j+1}: {ov_area:.2f} м² ({ov_frac*100:.1f}%)",
                        room_index=i,
                    )
                )

    overlap_score = max(0.0, 1.0 - total_overlap_penalty)

    # 2. Coverage Score
    total_room_area = sum(
        (max([p[0] for p in r.points]) - min([p[0] for p in r.points]))
        * (max([p[1] for p in r.points]) - min([p[1] for p in r.points]))
        for r in room_polygons
    )
    envelope_area = ow_m * oh_m
    cov_ratio = total_room_area / max(1.0, envelope_area)
    if cov_ratio < 0.60:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="LOW_COVERAGE",
                message=f"Низкий процент покрытия периметра квартиры: {cov_ratio*100:.1f}%",
            )
        )
        coverage_score = max(0.0, cov_ratio / 0.80)
    else:
        coverage_score = min(1.0, cov_ratio / 0.85)

    # 3. Connectivity Score
    connected_count = 0
    for idx, r in enumerate(room_polygons):
        has_door = any(any(o.type == "door" for o in w.openings) for w in r.walls)
        has_window = any(any(o.type == "window" for o in w.openings) for w in r.walls)
        if has_door or (n_rooms == 1 and (has_door or has_window)):
            connected_count += 1
        else:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="NO_DOOR",
                    message=f"В комнате #{idx+1} отсутствует входная дверь",
                    room_index=idx,
                )
            )
    connectivity_score = connected_count / max(1, n_rooms)

    # 4. Geometry Score
    geom_penalty = 0.0
    for idx, r in enumerate(room_polygons):
        xs, ys = [p[0] for p in r.points], [p[1] for p in r.points]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        aspect = max(w / max(0.1, h), h / max(0.1, w))
        if aspect > 3.2:
            geom_penalty += 0.15
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="EXTREME_ASPECT",
                    message=f"Комната #{idx+1} имеет вытянутые пропорции (aspect ratio {aspect:.1f})",
                    room_index=idx,
                )
            )
        if w < 1.3 or h < 1.3:
            geom_penalty += 0.1
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="TINY_DIMENSION",
                    message=f"Комната #{idx+1} слишком узкая ({w:.2f}м x {h:.2f}м)",
                    room_index=idx,
                )
            )
    geometry_score = max(0.0, 1.0 - geom_penalty)

    # Итоговый композитный скор
    overall = (
        (0.35 * overlap_score)
        + (0.25 * connectivity_score)
        + (0.25 * coverage_score)
        + (0.15 * geometry_score)
    )
    is_valid = overall >= 0.70 and not any(iss.severity == "error" for iss in issues)

    return CVQualityScore(
        overall_score=round(overall, 3),
        overlap_score=round(overlap_score, 3),
        coverage_score=round(coverage_score, 3),
        connectivity_score=round(connectivity_score, 3),
        geometry_score=round(geometry_score, 3),
        is_valid=is_valid,
        issues=issues,
    )



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

    n_samples = max(12, int(length_m * pixels_per_meter / 3))
    is_wall_samples = []

    for i in range(n_samples + 1):
        t = i / n_samples
        x_px = start_px[0] + (end_px[0] - start_px[0]) * t
        y_px = start_px[1] + (end_px[1] - start_px[1]) * t
        is_wall_samples.append(_sample_is_wall(image, x_px, y_px))

    # Группируем подряд идущие "не стена" (gap) участки в кандидаты на проём
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
                opening_type = "window" if is_exterior else "door"
                confidence = 0.70
                features = ["wall_gap"]

                # 1. Фактор стандартной строительной ширины
                if opening_type == "door":
                    if 0.70 <= width_m <= 1.15:
                        confidence += 0.22
                        features.append("standard_door_width")
                    elif 1.15 < width_m <= 1.80:
                        confidence += 0.15
                        features.append("double_door_width")
                else:
                    if 0.90 <= width_m <= 3.20:
                        confidence += 0.22
                        features.append("standard_window_width")

                # 2. Фактор внешнего периметра
                if is_exterior:
                    confidence += 0.05
                    features.append("facade_boundary")

                openings.append(
                    DetectedOpening(
                        type=opening_type,
                        position=round((frac_start + frac_end) / 2, 3),
                        width_m=round(width_m, 2),
                        confidence=round(min(0.99, confidence), 2),
                        features=features,
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
    structural_walls = cv2.morphologyEx(
        wall_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    )
    coords = cv2.findNonZero(structural_walls)
    if coords is None:
        coords = cv2.findNonZero(wall_mask)
    if coords is None:
        return SegmentationResult(
            room_polygons=[],
            image_width=width,
            image_height=height,
            pixels_per_meter=pixels_per_meter,
            quality_score=evaluate_segmentation_quality([], 0, 0, 0, 0),
        )

    ox, oy, ow, oh = cv2.boundingRect(coords)
    if ow * oh < (width * height) * 0.01:
        return SegmentationResult(
            room_polygons=[],
            image_width=width,
            image_height=height,
            pixels_per_meter=pixels_per_meter,
            quality_score=evaluate_segmentation_quality([], 0, 0, 0, 0),
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
    h_walls = cv2.morphologyEx(
        structural_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (14, 3))
    )
    v_walls = cv2.morphologyEx(
        structural_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 14))
    )

    # Базовая сетка с внешним периметром
    grid = np.zeros_like(wall_mask)
    cv2.rectangle(grid, (ox, oy), (ox + ow, oy + oh), 255, 12)

    # Замыкаем дверные проёмы (до 130px) вдоль физических стен
    h_closed = cv2.morphologyEx(
        h_walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (130, 1))
    )
    v_closed = cv2.morphologyEx(
        v_walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 130))
    )
    grid = cv2.bitwise_or(grid, cv2.bitwise_or(h_closed, v_closed))

    # 4. Маскируем внутреннее пространство квартиры (строго внутри несущих стен)
    inside_mask = np.zeros_like(wall_mask)
    inside_mask[oy + 10 : oy + oh - 10, ox + 10 : ox + ow - 10] = 255
    rooms_space = cv2.bitwise_and(cv2.bitwise_not(grid), inside_mask)

    # 5. Находим связные компоненты внутренних комнат и извлекаем точные полигоны
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rooms_space, connectivity=4)

    min_area_px = int(pixels_per_meter * pixels_per_meter * 3.0)

    detected_polygons: list[RoomPolygon] = []
    for l in range(1, num_labels):
        area = stats[l, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue

        mask = (labels == l).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        # Аппроксимация многоугольника комнаты (поддерживает прямоугольные, L-образные и любые многоугольные комнаты)
        peri = cv2.arcLength(cnts[0], True)
        poly = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        raw_pts = poly.reshape(-1, 2)
        if len(raw_pts) < 3:
            continue

        # Привязываем вершины контура к внешнему периметру здания и сетке стен (snapping)
        pts_m = []
        for p in raw_pts:
            px_m = float(p[0]) / pixels_per_meter
            py_m = float(p[1]) / pixels_per_meter
            sx = ox_m if abs(px_m - ox_m) < 0.35 else ((ox_m + ow_m) if abs(px_m - (ox_m + ow_m)) < 0.35 else px_m)
            sy = oy_m if abs(py_m - oy_m) < 0.35 else ((oy_m + oh_m) if abs(py_m - (oy_m + oh_m)) < 0.35 else py_m)
            pts_m.append((round(sx, 2), round(sy, 2)))

        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            s_px = (start_m[0] * pixels_per_meter, start_m[1] * pixels_per_meter)
            e_px = (end_m[0] * pixels_per_meter, end_m[1] * pixels_per_meter)
            length_m = math.hypot(end_m[0] - start_m[0], end_m[1] - start_m[1])

            # Проверяем плотность физической стены вдоль данного отрезка
            n_samples = max(10, int(length_m * pixels_per_meter / 3))
            samples = []
            tol = 8
            for i in range(n_samples + 1):
                t = i / n_samples
                x_px = int(round(s_px[0] + (e_px[0] - s_px[0]) * t))
                y_px = int(round(s_px[1] + (e_px[1] - s_px[1]) * t))
                x0, x1 = max(0, x_px - tol), min(width, x_px + tol + 1)
                y0, y1 = max(0, y_px - tol), min(height, y_px + tol + 1)
                patch = image[y0:y1, x0:x1]
                samples.append(float(patch.min()) < WALL_INTENSITY_THRESHOLD)
            wall_density = sum(samples) / max(1, len(samples))

            openings: list[DetectedOpening] = []

            # Если стена физически отсутствует (открытый проход между зонами), не строим глухую стену
            if wall_density < 0.15:
                # Открытый проход
                openings.append(
                    DetectedOpening(
                        type="door",
                        position=0.5,
                        width_m=round(length_m, 2),
                        confidence=0.95,
                        features=["open_passage", "zero_wall_density"],
                    )
                )
            else:
                # 1. Проверяем синие окна вдоль данного сегмента стены
                for wx, wy, ww, wh in detected_windows_boxes:
                    w_mid_x = wx + ww / 2
                    w_mid_y = wy + wh / 2
                    # Горизонтальная стена
                    if (
                        abs(s_px[1] - e_px[1]) < 15
                        and min(s_px[0], e_px[0]) <= w_mid_x <= max(s_px[0], e_px[0])
                        and abs(s_px[1] - w_mid_y) < 30
                    ):
                        pos = (w_mid_x - min(s_px[0], e_px[0])) / max(1.0, abs(e_px[0] - s_px[0]))
                        if s_px[0] > e_px[0]:
                            pos = 1.0 - pos
                        openings.append(
                            DetectedOpening(
                                type="window",
                                position=round(pos, 3),
                                width_m=round(max(ww, wh) / pixels_per_meter, 2),
                                confidence=0.98,
                                features=["color_window", "blue_hsv"],
                            )
                        )
                    # Вертикальная стена
                    elif (
                        abs(s_px[0] - e_px[0]) < 15
                        and min(s_px[1], e_px[1]) <= w_mid_y <= max(s_px[1], e_px[1])
                        and abs(s_px[0] - w_mid_x) < 30
                    ):
                        pos = (w_mid_y - min(s_px[1], e_px[1])) / max(1.0, abs(e_px[1] - s_px[1]))
                        if s_px[1] > e_px[1]:
                            pos = 1.0 - pos
                        openings.append(
                            DetectedOpening(
                                type="window",
                                position=round(pos, 3),
                                width_m=round(max(ww, wh) / pixels_per_meter, 2),
                                confidence=0.98,
                                features=["color_window", "blue_hsv"],
                            )
                        )

                # 2. Если синих окон нет — ищем дверные проёмы и разрывы
                if not openings:
                    is_ext = _is_exterior_wall_segment(start_m, end_m, ox_m, oy_m, ow_m, oh_m)
                    detected = _detect_openings(image, start_m, end_m, pixels_per_meter, is_exterior=is_ext)
                    for d in detected:
                        if (
                            abs(start_m[1] - (oy_m + oh_m)) < 0.5
                            and abs(end_m[1] - (oy_m + oh_m)) < 0.5
                            and min(start_m[0], end_m[0]) < ox_m + 3.5
                        ):
                            d.type = "door"
                        openings.append(d)

            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))

        detected_polygons.append(RoomPolygon(points=pts_m, walls=walls))

    # Если комната одна (студия без внутренних перегородок)
    if not detected_polygons:
        corners_px = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
        pts_m = [(round(float(px) / pixels_per_meter, 2), round(float(py) / pixels_per_meter, 2)) for px, py in corners_px]
        walls = []
        for start_m, end_m in _polygon_to_wall_segments(pts_m):
            is_ext = _is_exterior_wall_segment(start_m, end_m, ox_m, oy_m, ow_m, oh_m)
            openings = _detect_openings(image, start_m, end_m, pixels_per_meter, is_exterior=is_ext)
            walls.append(WallSegment(start=start_m, end=end_m, openings=openings))
        detected_polygons = [RoomPolygon(points=pts_m, walls=walls)]

    quality_score = evaluate_segmentation_quality(
        detected_polygons, ox_m=ox_m, oy_m=oy_m, ow_m=ow_m, oh_m=oh_m
    )

    return SegmentationResult(
        room_polygons=detected_polygons,
        image_width=width,
        image_height=height,
        pixels_per_meter=pixels_per_meter,
        quality_score=quality_score,
    )






