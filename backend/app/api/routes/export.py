"""Экспорт проекта — PDF-отчёт с планом, списком мебели и сметой."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core.database import projects_collection, scenes_collection
from app.export.pdf_export import generate_project_pdf
from app.models.project import Project
from app.models.scene import Scene

router = APIRouter(prefix="/projects", tags=["export"])


@router.get("/{project_id}/export/pdf")
async def export_pdf(project_id: str, variant_id: str | None = None):
    project_doc = await projects_collection().find_one({"id": project_id}, {"_id": 0})
    if not project_doc:
        raise HTTPException(404, "Проект не найден")
    project = Project(**project_doc)

    variant = variant_id or project.active_variant_id
    scene_doc = await scenes_collection().find_one(
        {"project_id": project_id, "variant_id": variant}, {"_id": 0}, sort=[("version", -1)]
    )
    if not scene_doc:
        raise HTTPException(404, "Сцена ещё не сгенерирована для этого варианта")
    scene = Scene(**scene_doc)

    pdf_bytes = generate_project_pdf(project, scene)
    filename = f"{project_id}_{variant}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
