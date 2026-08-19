"""
Оркестрация мультиагентного пайплайна через Taskiq.

Пайплайн разбит на ДВЕ стадии, а не на одну сплошную, из-за Architect Agent:
1. run_analysis_pipeline   — Upload -> FloorPlanAnalyzer -> RoomDetector -> ArchitectAgent
                              Останавливается на стадии "awaiting_architect_decision"
                              и ждёт выбора пользователя (см. api/routes/projects.py).
2. run_all_design_variants — (запускается после выбора варианта планировки, либо "как есть")
                              Генерирует НЕСКОЛЬКО вариантов дизайна (A/B/C — см. DESIGN_VARIANTS)
                              последовательно, каждый — отдельный документ Scene со своим
                              variant_id. Пользователь потом переключается между ними на фронтенде
                              и может продолжать чат с любым из них (Project.active_variant_id).

Это и есть чекпоинт: при повторном запуске design-пайплайна (например,
пользователь поменял пожелания) не нужно заново гонять CV и Architect Agent.

Ретраи: ошибки LLM-вызовов (LLMCallError) больше не глотаются молча —
они долетают сюда и переводят проект в статус failed с текстом ошибки,
видимым пользователю на фронтенде.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from app.agents.architect import agent as architect_agent
from app.agents.conversation import agent as conversation_agent
from app.agents.decorator import agent as decorator_agent
from app.agents.floorplan_analyzer import agent as floorplan_agent
from app.agents.furniture_planner import agent as furniture_agent
from app.agents.interior_designer import agent as interior_agent
from app.agents.lighting_designer import agent as lighting_agent
from app.agents.room_detector import agent as room_detector_agent
from app.agents.scene_generator import agent as scene_generator_agent
from app.core.config import get_settings
from app.core.database import agent_runs_collection, projects_collection, scenes_collection
from app.core.llm import LLMCallError
from app.core.redis_client import publish_status
from app.core.storage import get_s3_client
from app.models.project import PipelineStage, UserPreferences
from app.models.scene import Scene
from app.tasks.broker import broker

settings = get_settings()

def _get_design_variants(preferences: UserPreferences | None) -> list[tuple[str, str, str | None]]:
    """Генерирует 3 контрастных варианта дизайна с учётом базового стиля из анкеты."""
    base_style = (preferences.style if preferences and preferences.style else "Modern Minimalist").strip()
    
    style_mapping: dict[str, tuple[str, str]] = {
        "Modern Minimalist": ("Japandi Warm", "Contemporary Luxury"),
        "Minimalism": ("Japandi Warm", "Contemporary Luxury"),
        "Scandi": ("Japandi Organic", "Urban Industrial"),
        "Scandinavian": ("Japandi Organic", "Urban Industrial"),
        "Classic": ("Neoclassic Chic", "Art Deco Fusion"),
        "Loft": ("Dark Industrial", "Warm Wood Nordic"),
        "Industrial": ("Raw Concrete Loft", "Modern Mid-Century"),
        "Japandi": ("Zen Minimalism", "Nordic Light"),
    }

    alt1, alt2 = style_mapping.get(base_style, ("Japandi Warm", "Contemporary Luxury"))

    return [
        ("variant_a", f"Вариант A ({base_style})", None),
        ("variant_b", f"Вариант B ({alt1})", alt1),
        ("variant_c", f"Вариант C ({alt2})", alt2),
    ]


async def _log_agent_run(project_id: str, agent_name: str, status: str, **extra) -> None:
    await agent_runs_collection().insert_one(
        {
            "project_id": project_id,
            "agent_name": agent_name,
            "status": status,
            "created_at": datetime.utcnow(),
            **extra,
        }
    )


async def _set_stage(project_id: str, stage: PipelineStage, error: str | None = None) -> None:
    await projects_collection().update_one(
        {"id": project_id},
        {"$set": {"stage": stage.value, "error": error, "updated_at": datetime.utcnow()}},
    )
    await publish_status(project_id, {"stage": stage.value, "error": error})


@broker.task
async def run_analysis_pipeline(project_id: str, plan_object_key: str) -> None:
    try:
        await _set_stage(project_id, PipelineStage.analyzing_floorplan)
        image_bytes = get_s3_client().get_object(
            Bucket=settings.s3_bucket_name, Key=plan_object_key
        )["Body"].read()

        rooms = await floorplan_agent.analyze(image_bytes)
        await _log_agent_run(project_id, "floorplan_analyzer", "done", output_summary={"rooms": len(rooms)})

        await _set_stage(project_id, PipelineStage.detecting_rooms)
        rooms = await room_detector_agent.run(rooms)
        await _log_agent_run(project_id, "room_detector", "done")

        suggestions = await architect_agent.propose_suggestions(rooms)
        await _log_agent_run(project_id, "architect", "done", output_summary={"suggestions": len(suggestions)})

        scene = scene_generator_agent.assemble(
            project_id=project_id,
            version=1,
            rooms=rooms,
            furniture=[],
            lighting=[],
            decor=[],
            architect_suggestions=suggestions,
        )
        await scenes_collection().insert_one(scene.model_dump())

        await _set_stage(project_id, PipelineStage.awaiting_architect_decision)
    except Exception as exc:  # noqa: BLE001 - логируем и переводим проект в failed
        await _log_agent_run(project_id, "run_analysis_pipeline", "failed", error=str(exc))
        await _set_stage(project_id, PipelineStage.failed, error=str(exc))


async def _generate_one_variant(
    project_id: str,
    base_rooms,
    architect_suggestions,
    variant_id: str,
    variant_label: str,
    style_override: str | None,
    preferences: UserPreferences | None,
) -> None:
    prefs_obj = preferences
    if style_override:
        prefs_obj = UserPreferences(
            **{**(preferences.model_dump() if preferences else {}), "style": style_override}
        )

    rooms = [r.model_copy(deep=True) for r in base_rooms]

    rooms = await interior_agent.run(rooms, prefs_obj)
    await _log_agent_run(project_id, f"interior_designer[{variant_id}]", "done")

    furniture = await furniture_agent.run(rooms, prefs_obj)
    await _log_agent_run(project_id, f"furniture_planner[{variant_id}]", "done", output_summary={"items": len(furniture)})

    lighting = await lighting_agent.run(rooms, furniture)
    await _log_agent_run(project_id, f"lighting_designer[{variant_id}]", "done")

    decor = await decorator_agent.run(rooms, furniture)
    await _log_agent_run(project_id, f"decorator[{variant_id}]", "done")

    scene = scene_generator_agent.assemble(
        project_id=project_id,
        version=2,
        rooms=rooms,
        furniture=furniture,
        lighting=lighting,
        decor=decor,
        architect_suggestions=architect_suggestions,
        style=prefs_obj.style if prefs_obj else None,
    )
    scene_doc = scene.model_dump()
    scene_doc["variant_id"] = variant_id
    scene_doc["variant_label"] = variant_label
    await scenes_collection().insert_one(scene_doc)


@broker.task
async def run_all_design_variants(project_id: str, architect_choice_id: str | None) -> None:
    try:
        scene_doc = await scenes_collection().find_one(
            {"project_id": project_id}, sort=[("version", -1)]
        )
        if not scene_doc:
            raise ValueError("Сцена не найдена — analysis pipeline должен быть завершён раньше")
        scene_doc.pop("_id", None)
        base_scene = Scene(**scene_doc)

        project_doc = await projects_collection().find_one({"id": project_id})
        preferences_raw = project_doc.get("preferences") if project_doc else None
        preferences = UserPreferences(**preferences_raw) if preferences_raw else None

        rooms = base_scene.rooms
        if architect_choice_id:
            chosen = next(
                (s for s in base_scene.architect_suggestions if s.id == architect_choice_id), None
            )
            if chosen:
                rooms = await architect_agent.apply_suggestion(rooms, chosen)
                await _log_agent_run(project_id, "architect_apply", "done")

        await _set_stage(project_id, PipelineStage.designing_interior)

        async def _generate_and_notify(variant_id: str, variant_label: str, style_override: str | None) -> None:
            await _generate_one_variant(
                project_id=project_id,
                base_rooms=rooms,
                architect_suggestions=base_scene.architect_suggestions,
                variant_id=variant_id,
                variant_label=variant_label,
                style_override=style_override,
                preferences=preferences,
            )
            await publish_status(project_id, {"stage": "variant_ready", "variant_id": variant_id})

        variants = _get_design_variants(preferences)

        # Варианты независимы (разные variant_id, разные документы в Mongo) —
        # генерируем параллельно вместо последовательного цикла, это
        # утраивает throughput на LLM-вызовах вместо ожидания трёх раз подряд.
        await asyncio.gather(
            *[
                _generate_and_notify(variant_id, variant_label, style_override)
                for variant_id, variant_label, style_override in variants
            ]
        )

        await projects_collection().update_one(
            {"id": project_id}, {"$set": {"active_variant_id": variants[0][0]}}
        )
        await _set_stage(project_id, PipelineStage.ready)
    except Exception as exc:  # noqa: BLE001
        await _log_agent_run(project_id, "run_all_design_variants", "failed", error=str(exc))
        await _set_stage(project_id, PipelineStage.failed, error=str(exc))


@broker.task
async def run_conversation_turn(project_id: str, variant_id: str, user_message: str) -> dict:
    """Обрабатывает одну реплику чата: интерпретирует, патчит сцену, сохраняет новую версию.

    Ошибки LLM здесь НЕ должны валить весь проект (в отличие от основного пайплайна) —
    одна неудачная реплика в чате не критична, поэтому LLMCallError ловится локально
    и возвращается как понятное сообщение пользователю.
    """
    scene_doc = await scenes_collection().find_one(
        {"project_id": project_id, "variant_id": variant_id}, sort=[("version", -1)]
    )
    if not scene_doc:
        return {"error": "Сцена не найдена"}
    scene_doc.pop("_id", None)
    scene = Scene(**scene_doc)

    try:
        operations = await conversation_agent.interpret(scene, user_message)
    except LLMCallError as exc:
        return {"error": f"Не удалось обработать команду: {exc}"}

    updated_scene = conversation_agent.apply_operations(scene, operations)
    updated_doc = updated_scene.model_dump()
    updated_doc["variant_id"] = variant_id
    updated_doc["variant_label"] = scene_doc.get("variant_label", variant_id)
    await scenes_collection().insert_one(updated_doc)

    await publish_status(project_id, {"stage": "scene_updated", "variant_id": variant_id, "version": updated_scene.version})
    return {"operations": operations, "version": updated_scene.version}
