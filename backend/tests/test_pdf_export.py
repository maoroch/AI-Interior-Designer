"""Тесты PDF-экспорта — генерируем на синтетической сцене, проверяем, что
получается валидный непустой PDF (глубокая проверка содержимого через
рендер в PNG делалась вручную при разработке, см. README)."""
from app.export.pdf_export import _estimate_price, generate_project_pdf
from app.models.project import Project, UserPreferences
from app.models.scene import FurnitureItem, Room, RoomType, Scene


def _make_project_and_scene():
    room = Room(id="r1", type=RoomType.living_room, polygon=[(0, 0), (5, 0), (5, 4), (0, 4)])
    furniture = [
        FurnitureItem(id="f1", room_id="r1", type="sofa", position=(2, 0, 1), dimensions=(2.1, 0.85, 0.9), material="grey_fabric"),
        FurnitureItem(id="f2", room_id="r1", type="unknown_type_xyz", position=(4, 0, 3), dimensions=(1, 1, 1)),
    ]
    scene = Scene(project_id="proj_1", version=1, rooms=[room], furniture=furniture, style="Modern Minimalism")
    project = Project(id="proj_1", preferences=UserPreferences(budget_usd=1000, adults=2, children=0))
    return project, scene


def test_generates_nonempty_pdf_bytes():
    project, scene = _make_project_and_scene()
    pdf_bytes = generate_project_pdf(project, scene)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_unknown_furniture_type_falls_back_to_default_price():
    assert _estimate_price("unknown_type_xyz") > 0
    assert _estimate_price("sofa") != _estimate_price("unknown_type_xyz")


def test_pdf_generation_works_with_empty_furniture():
    project, scene = _make_project_and_scene()
    scene.furniture = []
    pdf_bytes = generate_project_pdf(project, scene)
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_generation_works_without_preferences():
    project, scene = _make_project_and_scene()
    project.preferences = None
    pdf_bytes = generate_project_pdf(project, scene)
    assert pdf_bytes.startswith(b"%PDF")
