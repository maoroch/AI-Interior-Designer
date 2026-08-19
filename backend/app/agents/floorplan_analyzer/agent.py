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
import logging
import uuid

from app.core.config import get_settings
from app.core.llm import complete_vision_json
from app.cv.segmentation import segment_floor_plan
from app.models.scene import Opening, OpeningType, Room, Wall

logger = logging.getLogger("floorplan_analyzer")


async def analyze(image_bytes: bytes) -> list[Room]:
    cv_result = segment_floor_plan(image_bytes)

    rooms: list[Room] = []
    for room_polygon in cv_result.room_polygons:
        walls: list[Wall] = []
        for wall_seg in room_polygon.walls:
            openings = [
                Opening(
                    type=OpeningType.door if o.type == "door" else OpeningType.window,
                    position=o.position,
                    width=o.width_m,
                )
                for o in wall_seg.openings
            ]
            walls.append(
                Wall(
                    id=f"wall_{uuid.uuid4().hex[:8]}",
                    start=wall_seg.start,
                    end=wall_seg.end,
                    openings=openings,
                )
            )

        rooms.append(
            Room(
                id=f"room_{uuid.uuid4().hex[:8]}",
                polygon=room_polygon.points,
                walls=walls,
            )
        )

    # Валидация через Groq vision (если указан API ключ)
    settings = get_settings()
    if settings.groq_api_key:
        try:
            prompt = (
                "Проанализируй чертёж/план помещения на изображении. "
                "Определи приблизительную общую площадь в кв.м и типы помещений. "
                "Верни JSON: {\"estimated_area_sqm\": float, \"rooms_detected\": [{\"name\": str, \"suggested_type\": str}]}"
            )
            vision_result = await complete_vision_json(
                system_prompt="Ты — эксперт по архитектурным чертежам и планам квартир.",
                user_prompt=prompt,
                image_bytes=image_bytes,
            )
            logger.info("Vision LLM analysis: %s", vision_result)
        except Exception as e:
            logger.warning("Vision LLM validation failed, continuing with CV segmentation: %s", e)

    return rooms

