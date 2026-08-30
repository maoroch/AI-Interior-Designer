"""
Основной жизненный цикл проекта:

POST /projects/upload                 -> создать проект, загрузить план, запустить analysis pipeline
POST /projects/{id}/preferences       -> сохранить анкету пользователя
POST /projects/{id}/architect-choice  -> выбрать вариант перепланировки (или "as_is")
                                          -> запустить генерацию всех вариантов дизайна (A/B/C)
GET  /projects/{id}                   -> статус проекта
GET  /projects/{id}/scene             -> последняя версия СЕЙЧАС АКТИВНОГО варианта сцены
GET  /projects/{id}/scenes            -> последние версии ВСЕХ вариантов (для переключения A/B/C)
POST /projects/{id}/select-variant    -> сделать один из вариантов активным (для чата и просмотра)
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
import io

from app.core.database import projects_collection, scenes_collection
from app.core.storage import ensure_bucket, upload_bytes
from app.models.project import PipelineStage, Project, UserPreferences
from app.models.scene import Scene
from app.tasks.pipeline_tasks import run_all_design_variants, run_analysis_pipeline

router = APIRouter(prefix="/projects", tags=["projects"])

MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15MB
MAX_IMAGE_DIMENSION_PX = 8000  # защита от decompression-bomb файлов


@router.post("/upload")
async def upload_plan(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Поддерживаются только PNG и JPG на MVP-этапе")

    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, f"Файл слишком большой (максимум {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB)")
    if len(data) == 0:
        raise HTTPException(400, "Пустой файл")

    # Проверяем, что это действительно валидное изображение (а не файл с поддельным
    # расширением/content-type), и что размеры разумны — иначе CV-пайплайн и
    # decode в OpenCV может упасть или отожрать память на "изображении-бомбе".
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        raise HTTPException(400, "Файл повреждён или не является изображением")

    if width > MAX_IMAGE_DIMENSION_PX or height > MAX_IMAGE_DIMENSION_PX:
        raise HTTPException(
            400, f"Слишком большое разрешение изображения (максимум {MAX_IMAGE_DIMENSION_PX}px по стороне)"
        )
    if width < 50 or height < 50:
        raise HTTPException(400, "Изображение слишком маленькое, чтобы быть планом квартиры")

    ensure_bucket()
    extension = "png" if file.content_type == "image/png" else "jpg"
    object_key = upload_bytes(data, key_prefix="floorplans", extension=extension, content_type=file.content_type)

    project = Project(id=f"proj_{uuid.uuid4().hex[:10]}", original_plan_key=object_key)
    await projects_collection().insert_one(project.model_dump())

    await run_analysis_pipeline.kiq(project_id=project.id, plan_object_key=object_key)

    return {"project_id": project.id, "stage": project.stage.value}


@router.post("/{project_id}/preferences")
async def set_preferences(project_id: str, preferences: UserPreferences):
    result = await projects_collection().update_one(
        {"id": project_id},
        {"$set": {"preferences": preferences.model_dump(), "updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Проект не найден")
    return {"ok": True}


@router.post("/{project_id}/architect-choice")
async def choose_architect_option(project_id: str, choice_id: str | None = None):
    """choice_id = None означает 'оставить планировку как есть'. Запускает генерацию
    сразу нескольких вариантов дизайна (A/B/C, см. DESIGN_VARIANTS в pipeline_tasks.py)."""
    project_doc = await projects_collection().find_one({"id": project_id})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    if project_doc["stage"] != PipelineStage.awaiting_architect_decision.value:
        raise HTTPException(409, "Проект ещё не готов к выбору планировки")

    await run_all_design_variants.kiq(project_id=project_id, architect_choice_id=choice_id)
    return {"ok": True}


@router.get("/{project_id}")
async def get_project(project_id: str):
    project_doc = await projects_collection().find_one({"id": project_id}, {"_id": 0})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    return project_doc


@router.get("/{project_id}/scene")
async def get_scene(project_id: str):
    project_doc = await projects_collection().find_one({"id": project_id})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    variant_id = project_doc.get("active_variant_id", "variant_a")

    scene_doc = await scenes_collection().find_one(
        {"project_id": project_id, "variant_id": variant_id}, {"_id": 0}, sort=[("version", -1)]
    )
    if not scene_doc:
        raise HTTPException(404, "Сцена ещё не сгенерирована")
    return Scene(**scene_doc).model_dump()


@router.get("/{project_id}/scenes")
async def get_all_variant_scenes(project_id: str):
    """Последняя версия КАЖДОГО варианта дизайна — для переключателя A/B/C на фронтенде."""
    pipeline = [
        {"$match": {"project_id": project_id}},
        {"$sort": {"version": -1}},
        {"$group": {"_id": "$variant_id", "doc": {"$first": "$$ROOT"}}},
    ]
    variants = []
    async for row in scenes_collection().aggregate(pipeline):
        doc = row["doc"]
        doc.pop("_id", None)
        variants.append(Scene(**doc).model_dump())
    variants.sort(key=lambda s: s["variant_id"])
    return variants


@router.post("/{project_id}/select-variant")
async def select_variant(project_id: str, variant_id: str):
    exists = await scenes_collection().find_one({"project_id": project_id, "variant_id": variant_id})
    if not exists:
        raise HTTPException(404, "Такой вариант не найден")
    result = await projects_collection().update_one(
        {"id": project_id}, {"$set": {"active_variant_id": variant_id, "updated_at": datetime.utcnow()}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Проект не найден")
    return {"ok": True, "active_variant_id": variant_id}


class FurniturePatch(BaseModel):
    position: tuple[float, float, float] | None = None
    rotation_deg: float | None = None
    material: str | None = None
    color: str | None = None


@router.patch("/{project_id}/scene/furniture/{furniture_id}")
async def patch_furniture(project_id: str, furniture_id: str, patch: FurniturePatch):
    """Прямое изменение мебели из 3D-UI (drag&drop позиции, выбор материала/цвета) —
    в отличие от чата, не требует интерпретации LLM, применяется мгновенно."""
    project_doc = await projects_collection().find_one({"id": project_id})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    variant_id = project_doc.get("active_variant_id", "variant_a")

    scene_doc = await scenes_collection().find_one(
        {"project_id": project_id, "variant_id": variant_id}, sort=[("version", -1)]
    )
    if not scene_doc:
        raise HTTPException(404, "Сцена не найдена")
    scene_doc.pop("_id", None)
    scene = Scene(**scene_doc)

    item = next((f for f in scene.furniture if f.id == furniture_id), None)
    if not item:
        raise HTTPException(404, "Предмет мебели не найден")

    if patch.position is not None:
        item.position = patch.position
    if patch.rotation_deg is not None:
        item.rotation_deg = patch.rotation_deg
    if patch.material is not None:
        item.material = patch.material
    if patch.color is not None:
        item.color = patch.color

    scene.version += 1
    new_doc = scene.model_dump()
    new_doc["variant_id"] = variant_id
    new_doc["variant_label"] = scene_doc.get("variant_label", variant_id)
    await scenes_collection().insert_one(new_doc)

    return scene.model_dump()


class RoomUpdateItem(BaseModel):
    id: str
    type: str
    height: float | None = None
    label: str | None = None
    enabled: bool = True


class UpdateRoomsRequest(BaseModel):
    rooms: list[RoomUpdateItem]


@router.patch("/{project_id}/rooms")
async def update_project_rooms(project_id: str, request: UpdateRoomsRequest):
    """Обновляет экспликацию помещений (типы комнат, высоту потолков, подписи) от пользователя."""
    project_doc = await projects_collection().find_one({"id": project_id})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")

    room_map = {r.id: r for r in request.rooms}

    cursor = scenes_collection().find({"project_id": project_id})
    async for scene_doc in cursor:
        updated_rooms = []
        for r in scene_doc.get("rooms", []):
            rid = r.get("id")
            if rid in room_map:
                u = room_map[rid]
                if not u.enabled:
                    continue
                r["type"] = u.type
                if u.height is not None and u.height > 0:
                    r["height"] = round(u.height, 2)
                if u.label:
                    r["label"] = u.label
            updated_rooms.append(r)

        await scenes_collection().update_one(
            {"_id": scene_doc["_id"]},
            {"$set": {"rooms": updated_rooms}}
        )

    return {"ok": True, "updated_count": len(request.rooms)}

