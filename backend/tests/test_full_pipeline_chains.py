"""
Интеграционные сквозные тесты полных цепочек пайплайна (End-to-End Pipeline Chains).

Тестирует сквозную цепочку:
1. 2D Чертёж (PNG)
2. CV Сегментация (OpenCV)
3. Векторный граф стен (WallGraph)
4. Семантический мост (Semantic Bridge)
5. CAD-компилятор расстановки (Compiler)
6. Сборщик сцены (Scene Generator)
7. Интерактивная модификация через чат (Conversation Agent)
8. Коммерческая смета и спецификация (PDF Export)
"""
import os
import pytest
from app.cv.segmentation import segment_floor_plan
from app.cv.wall_graph import build_wall_graph_from_segmentation
from app.cv.semantic_bridge import generate_semantic_room_brief
from app.agents.furniture_planner.compiler import compile_semantic_layout_to_3d
from app.agents.conversation.agent import apply_operations
from app.models.scene import Scene, Room, Wall, Opening, OpeningType, RoomType, FurnitureItem, LightSource
from app.export.pdf_export import generate_project_pdf
from app.models.project import Project, PipelineStage, UserPreferences


SAMPLE_PLANS = [
    "sample_plans/plan1_studio.png",
    "sample_plans/plan2_euro2k.png",
    "sample_plans/plan3_classic2bed.png",
    "sample_plans/plan4_family3bed.png",
    "sample_plans/plan5_loft.png",
]


@pytest.mark.parametrize("plan_path", SAMPLE_PLANS)
def test_end_to_end_cv_to_semantic_cad_chain(plan_path: str):
    """
    Проверяет полную цепочку: 2D Чертёж -> Сегментация -> Граф стен -> Семантический бриф -> 3D CAD компилятор.
    Выполняется для всех 5 образцов планировок.
    """
    assert os.path.exists(plan_path), f"Файл чертежа {plan_path} не найден"
    with open(plan_path, "rb") as f:
        img_bytes = f.read()

    # Шаг 1: Сегментация
    seg_res = segment_floor_plan(img_bytes)
    assert len(seg_res.room_polygons) >= 1
    assert seg_res.quality_score.coverage_score > 0.30

    # Шаг 2: Векторный граф стен
    graph = build_wall_graph_from_segmentation(seg_res.room_polygons)
    assert len(graph.rooms) == len(seg_res.room_polygons)
    assert len(graph.edges) >= 4

    # Шаг 3: Семантический мост и CAD-компилятор для каждой комнаты
    total_furniture = 0
    for room_face in graph.rooms:
        brief = generate_semantic_room_brief(room_face, room_type="living_room")
        assert brief.width_m > 0 and brief.depth_m > 0
        assert len(brief.walls) >= 3
        assert isinstance(brief.text_summary, str)

        # Конвертация в модель Room для компилятора
        walls = []
        for e in room_face.walls:
            openings = [Opening(type=OpeningType(o.type if o.type in ["door", "window"] else "door"), position=o.position, width=o.width_m) for o in e.openings]
            walls.append(Wall(id=e.id, start=e.start, end=e.end, thickness=e.thickness, openings=openings))

        room_model = Room(
            id=room_face.id,
            name="Main Room",
            type=RoomType.living_room,
            polygon=room_face.polygon,
            walls=walls,
            area_sqm=room_face.area_sqm,
        )

        semantic_items = [
            {
                "type": "sofa",
                "anchor_wall": "south",
                "placement": "center",
                "distance_from_wall_cm": 10,
                "dimensions_cm": {"width": 200, "height": 85, "depth": 90},
                "material": "fabric",
                "color": "#4A5568",
            },
            {
                "type": "coffee_table",
                "anchor_wall": "center",
                "placement": "center",
                "dimensions_cm": {"width": 100, "height": 45, "depth": 60},
                "material": "wood",
                "color": "#D4A373",
            },
        ]

        placed = compile_semantic_layout_to_3d(room_model, semantic_items)
        assert len(placed) == 2
        for item in placed:
            # Гарантия валидных 3D координат
            assert isinstance(item.position[0], float)
            assert isinstance(item.position[1], float)
            assert isinstance(item.position[2], float)
            assert item.dimensions[0] > 0
            assert item.dimensions[1] > 0
            assert item.dimensions[2] > 0
        total_furniture += len(placed)

    assert total_furniture >= len(graph.rooms) * 2


def test_full_scene_patch_to_pdf_export_chain():
    """
    Проверяет цепочку: Сцена -> Модификация через Conversation Agent -> Генерация PDF-сметы.
    """
    # 1. Исходная сцена
    initial_scene = Scene(
        project_id="proj_test_chain",
        version=1,
        style="Modern Minimalist",
        color_palette=["#F7F7F7", "#333333", "#D4A373"],
        rooms=[
            Room(
                id="room_1",
                type=RoomType.living_room,
                polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
                walls=[
                    Wall(id="w1", start=(0.0, 0.0), end=(5.0, 0.0)),
                    Wall(id="w2", start=(5.0, 0.0), end=(5.0, 4.0)),
                    Wall(id="w3", start=(5.0, 4.0), end=(0.0, 4.0)),
                    Wall(id="w4", start=(0.0, 4.0), end=(0.0, 0.0)),
                ],
                area_sqm=20.0,
            )
        ],
        furniture=[
            FurnitureItem(
                id="f_sofa_1",
                room_id="room_1",
                type="sofa",
                position=(2.5, 0.0, 3.4),
                dimensions=(2.0, 0.85, 0.9),
                material="grey_fabric",
                color="#7A8288",
            )
        ],
        lighting=[
            LightSource(
                id="l_main",
                type="ceiling",
                position=(2.5, 2.7, 2.0),
                intensity=1.0,
                color_temperature_k=3000,
            )
        ],
        decor=[],
    )

    # 2. Модификация через чат: меняем материал дивана на кожу и добавляем растение
    operations = [
        {"op": "update", "collection": "furniture", "target_id": "f_sofa_1", "fields": {"material": "leather_brown", "color": "#6F4423"}},
        {"op": "add", "collection": "decor", "target_id": None, "fields": {"type": "plant", "room_id": "room_1", "position": (0.5, 0.0, 0.5)}},
    ]
    updated_scene = apply_operations(initial_scene, operations)
    assert updated_scene.version == 2
    sofa = next(f for f in updated_scene.furniture if f.id == "f_sofa_1")
    assert sofa.material == "leather_brown"
    assert len(updated_scene.decor) == 1

    # 3. Экспорт в коммерческую PDF-смету
    project = Project(
        id="proj_test_chain",
        stage=PipelineStage.ready,
        preferences=UserPreferences(budget_usd=1000, adults=2, children=0),
    )

    pdf_bytes = generate_project_pdf(project, updated_scene)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

