"""
Agent 1 — Floor Plan Analyzer.

Задача: по растровому изображению плана определить стены, комнаты (как полигоны,
без классификации типа — это делает Room Detector), двери, окна, размеры.

Подход (гибрид, см. README):
1. CV-этап (app/cv/segmentation.py) — сегментация стен/проёмов на изображении.
   На старте — placeholder на OpenCV-эвристиках (contours + анализ разрывов
   в линии стены для дверей/окон).
   TODO: заменить на модель U-Net/DeepLabV3+, обученную на CubiCasa5K
   (проверить лицензию весов перед использованием в проде — см. README).
2. LLM-этап (Groq vision) — валидация результата CV, простановка размеров
   там, где на плане есть текстовые аннотации (например "3.2m"), и разрешение
   неоднозначностей (какая линия — несущая стена, а какая — перегородка).

Вход: сырые байты изображения плана.
Выход: список Room (только polygon/walls, без type) — заготовка для Room Detector.
"""
import asyncio
import logging
import uuid


from app.core.config import get_settings
from app.core.llm import complete_vision_json
from app.cv.segmentation import segment_floor_plan
from app.models.scene import Opening, OpeningType, Room, Wall

logger = logging.getLogger("floorplan_analyzer")


async def analyze(image_bytes: bytes) -> list[Room]:
    cv_result = segment_floor_plan(image_bytes)

    if cv_result.quality_score:
        logger.info(
            "CV Segmentation Quality: overall=%.1f%% (overlap=%.1f%%, coverage=%.1f%%, connectivity=%.1f%%, geometry=%.1f%%, is_valid=%s, issues=%d)",
            cv_result.quality_score.overall_score * 100,
            cv_result.quality_score.overlap_score * 100,
            cv_result.quality_score.coverage_score * 100,
            cv_result.quality_score.connectivity_score * 100,
            cv_result.quality_score.geometry_score * 100,
            cv_result.quality_score.is_valid,
            len(cv_result.quality_score.issues),
        )

    # Построение единого векторного графа стен (устранение дубликатов и фантомных перегородок)
    from app.cv.wall_graph import build_wall_graph_from_segmentation
    wall_graph = build_wall_graph_from_segmentation(
        room_polygons=cv_result.room_polygons,
        pixels_per_meter=cv_result.pixels_per_meter,
        image_width=cv_result.image_width,
        image_height=cv_result.image_height,
    )

    rooms: list[Room] = []
    for face in wall_graph.rooms:
        walls: list[Wall] = []
        for edge in face.walls:
            openings = [
                Opening(
                    type=OpeningType.door if o.type == "door" else OpeningType.window,
                    position=o.position,
                    width=o.width_m,
                )
                for o in edge.openings
            ]
            walls.append(
                Wall(
                    id=edge.id,
                    start=edge.start,
                    end=edge.end,
                    thickness=edge.thickness,
                    openings=openings,
                )
            )

        rooms.append(
            Room(
                id=f"room_{uuid.uuid4().hex[:8]}",
                polygon=face.polygon,
                walls=walls,
            )
        )

    # Валидация через Groq vision (если указан API ключ) — не блокирует пайплайн при таймаутах
    settings = get_settings()
    if settings.groq_api_key:
        try:
            prompt = (
                "Проанализируй чертёж помещения. Верни JSON: "
                "{\"estimated_area_sqm\": float, \"rooms_detected\": [{\"name\": str, \"suggested_type\": str}]}"
            )
            vision_coro = complete_vision_json(
                system_prompt="Ты — эксперт по архитектурным чертежам. Отвечай только JSON.",
                user_prompt=prompt,
                image_bytes=image_bytes,
            )
            vision_result = await asyncio.wait_for(vision_coro, timeout=8.0)
            logger.info("Vision LLM analysis: %s", vision_result)
        except Exception as e:
            logger.info("Vision LLM skipped or timed out, using CV segmentation: %s", e)

    return rooms


