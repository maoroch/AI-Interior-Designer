"""
Unit and Integration Tests for the Hybrid Architectural Layout & Validation Engine:
1. Standards Dataset Validation (Poché & Neufert Standards)
2. Vector Feature Embeddings and Rule RAG (Vector Engine & NumPy Cosine Similarity)
3. Zero-LLM Fast-Path Template Matcher (CubiCasa5K & LCSF Archetypes)
4. Layout Ergonomics Validator & Linter (Violations, Door Guard, Window Guard, Sightlines)
5. End-to-End Golden Dataset Benchmark on Sample Floorplans
"""
import math
import numpy as np
import pytest
from app.knowledge.vector_engine import (
    RoomFeatureVectorizer,
    VectorRuleRAG,
    _calc_polygon_area,
    _calc_polygon_perimeter,
)
from app.knowledge.fast_path import FastPathLayoutMatcher
from app.knowledge.validator import LayoutErgonomicsValidator, Violation, ValidationReport
from app.models.scene import (
    Scene,
    Room,
    Wall,
    Opening,
    OpeningType,
    RoomType,
    FurnitureItem,
)


# ==============================================================================
# 1. Unit Tests for Standards Dataset
# ==============================================================================

def test_standards_json_loading_and_pydantic_validation():
    rules = VectorRuleRAG.load_standards()
    assert len(rules) >= 7, "Standards dataset must contain all core ergonomic rules"
    
    rule_ids = [r["id"] for r in rules]
    assert "DOOR_SWING_CLEARANCE" in rule_ids
    assert "WINDOW_SILL_CLEARANCE" in rule_ids
    assert "TV_SIGHTLINE_ALIGNMENT" in rule_ids
    assert "MAIN_PASSAGE_CIRCULATION" in rule_ids
    assert "NON_PENETRATION_SOLID_GAP" in rule_ids


def test_standards_values_match_neufert_thresholds():
    rules = {r["id"]: r for r in VectorRuleRAG.load_standards()}
    
    door_rule = rules["DOOR_SWING_CLEARANCE"]
    assert door_rule["severity"] == "CRITICAL"
    assert door_rule["min_radius_m"] >= 0.85
    assert "Neufert" in door_rule["standard_source"]

    tv_rule = rules["TV_SIGHTLINE_ALIGNMENT"]
    assert tv_rule["max_deviation_angle_deg"] <= 25.0
    assert tv_rule["min_distance_m"] >= 1.50


# ==============================================================================
# 2. Unit Tests for Vector Engine & Cosine Similarity
# ==============================================================================

def test_room_feature_vectorizer_dimensionality_and_normalization():
    # Прямоугольная комната 5x4м
    poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]
    vec = RoomFeatureVectorizer.vectorize_room(
        room_polygon=poly,
        room_type="living_room",
        style="Modern Minimalism",
        door_count=1,
        window_count=1,
    )
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
    # Вектор должен быть L2-нормализован
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-3)


def test_cosine_similarity_identity_and_orthogonality():
    v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert math.isclose(RoomFeatureVectorizer.cosine_similarity(v1, v2), 1.0, rel_tol=1e-5)
    assert math.isclose(RoomFeatureVectorizer.cosine_similarity(v1, v3), 0.0, abs_tol=1e-5)


def test_vector_rule_rag_retrieves_correct_top_k_rules():
    living_rules = VectorRuleRAG.get_top_rules_for_room("living_room", top_k=3)
    assert len(living_rules) == 3
    rule_ids = [r["id"] for r in living_rules]
    # Для гостиной соосность ТВ должна быть среди ключевых правил
    assert "TV_SIGHTLINE_ALIGNMENT" in rule_ids or "DOOR_SWING_CLEARANCE" in rule_ids

    prompt_text = VectorRuleRAG.format_rules_for_prompt("living_room")
    assert "Poché & Neufert Standards" in prompt_text


# ==============================================================================
# 3. Unit Tests for Zero-LLM Fast-Path Matcher
# ==============================================================================

def test_fast_path_hits_standard_rectangular_living_room():
    # Стандартная гостиная 6x4м (24 кв.м)
    room = Room(
        id="test_living",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)],
        walls=[
            Wall(id="w1", start=(0.0, 0.0), end=(6.0, 0.0), thickness=0.2, openings=[Opening(type=OpeningType.door, position=0.5, width=0.9)]),
            Wall(id="w2", start=(6.0, 0.0), end=(6.0, 4.0), thickness=0.2, openings=[Opening(type=OpeningType.window, position=0.5, width=1.4)]),
            Wall(id="w3", start=(6.0, 4.0), end=(0.0, 4.0), thickness=0.2, openings=[]),
            Wall(id="w4", start=(0.0, 4.0), end=(0.0, 0.0), thickness=0.2, openings=[]),
        ],
    )
    items, similarity, archetype_id = FastPathLayoutMatcher.match_golden_template(
        room=room,
        style="Modern Minimalism",
        similarity_threshold=0.85,
    )
    assert items is not None, "Standard rectangular living room must trigger Fast-Path hit"
    assert similarity >= 0.85
    assert len(items) >= 4
    types = [it["type"] for it in items]
    assert "sofa" in types
    assert "tv_stand" in types


def test_fast_path_misses_complex_irregular_geometry():
    # Нестандартный многоугольник (7 углов с узким аппендиксом)
    room = Room(
        id="test_irregular",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (3.0, 1.0), (3.0, 6.0), (0.0, 6.0)],
        walls=[],
    )
    # Для сильно нестандартной комнаты с низким порогом сходства Fast-Path должен уступить Groq RAG
    items, similarity, archetype_id = FastPathLayoutMatcher.match_golden_template(
        room=room,
        style="Modern Minimalism",
        similarity_threshold=0.96,
    )
    assert items is None, "Irregular complex geometry must not trigger false positive fast-path"


# ==============================================================================
# 4. Unit Tests for Layout Ergonomics Validator
# ==============================================================================

def test_validator_detects_door_blockage():
    # Создаем комнату с дверью в (3.0, 0.0)
    room = Room(
        id="r1",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)],
        walls=[
            Wall(id="w1", start=(0.0, 0.0), end=(6.0, 0.0), thickness=0.2, openings=[Opening(type=OpeningType.door, position=0.5, width=0.9)]),
        ],
    )
    # Ставим шкаф прямо на дверь (3.0, 0.2)
    wardrobe = FurnitureItem(
        id="f_bad_wardrobe",
        room_id="r1",
        type="wardrobe",
        position=(3.0, 0.0, 0.2),
        rotation_deg=0.0,
        dimensions=(1.2, 2.1, 0.6),
    )
    scene = Scene(
        project_id="test_p",
        version=1,
        rooms=[room],
        furniture=[wardrobe],
        lighting=[],
        decor=[],
        architect_suggestions=[],
    )

    report = LayoutErgonomicsValidator.validate_scene(scene)
    assert not report.is_valid, "Scene with blocked door must be marked invalid"
    assert report.critical_count >= 1
    assert any(v.rule_id == "DOOR_SWING_CLEARANCE" for v in report.violations)


def test_validator_detects_window_obstruction_by_tall_furniture():
    room = Room(
        id="r1",
        type=RoomType.bedroom,
        polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
        walls=[
            Wall(id="w1", start=(5.0, 0.0), end=(5.0, 4.0), thickness=0.2, openings=[Opening(type=OpeningType.window, position=0.5, width=1.2)]),
        ],
    )
    # Высокий книжный шкаф высотой 2.0м поставлен прямо перед окном (5.0, 2.0)
    shelf = FurnitureItem(
        id="f_tall_shelf",
        room_id="r1",
        type="bookshelf",
        position=(4.8, 0.0, 2.0),
        rotation_deg=90.0,
        dimensions=(0.9, 2.0, 0.4),
    )
    scene = Scene(
        project_id="test_p",
        version=1,
        rooms=[room],
        furniture=[shelf],
        lighting=[],
        decor=[],
        architect_suggestions=[],
    )

    report = LayoutErgonomicsValidator.validate_scene(scene)
    assert not report.is_valid
    assert any(v.rule_id == "WINDOW_SILL_CLEARANCE" for v in report.violations)


def test_validator_detects_misaligned_sofa_tv_sightline():
    room = Room(
        id="r1",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        walls=[],
    )
    # Диван в (4.0, 6.0), ТВ на севере в (4.0, 2.0).
    # Диван повернут на 0 градусов (лицом на юг, спиной к ТВ!)
    sofa = FurnitureItem(
        id="f_sofa",
        room_id="r1",
        type="sofa",
        position=(4.0, 0.0, 6.0),
        rotation_deg=0.0,  # Ошибка: смотрит от ТВ в стену
        dimensions=(2.1, 0.85, 0.9),
    )
    tv = FurnitureItem(
        id="f_tv",
        room_id="r1",
        type="tv_stand",
        position=(4.0, 0.0, 2.0),
        rotation_deg=0.0,
        dimensions=(1.6, 0.5, 0.4),
    )
    scene = Scene(
        project_id="test_p",
        version=1,
        rooms=[room],
        furniture=[sofa, tv],
        lighting=[],
        decor=[],
        architect_suggestions=[],
    )

    report = LayoutErgonomicsValidator.validate_scene(scene)
    assert report.error_count >= 1
    assert any(v.rule_id == "TV_SIGHTLINE_ALIGNMENT" for v in report.violations)


def test_validator_passes_valid_ergonomic_scene():
    room = Room(
        id="r1",
        type=RoomType.living_room,
        polygon=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        walls=[
            Wall(id="w1", start=(0.0, 0.0), end=(8.0, 0.0), thickness=0.2, openings=[Opening(type=OpeningType.door, position=0.1, width=0.9)]),
        ],
    )
    # Корректная расстановка: диван на южной стене смотрит на север на ТВ (rot = 180 deg)
    sofa = FurnitureItem(
        id="f_sofa",
        room_id="r1",
        type="sofa",
        position=(4.0, 0.0, 7.2),
        rotation_deg=180.0,
        dimensions=(2.1, 0.85, 0.9),
    )
    tv = FurnitureItem(
        id="f_tv",
        room_id="r1",
        type="tv_stand",
        position=(4.0, 0.0, 1.0),
        rotation_deg=0.0,
        dimensions=(1.6, 0.5, 0.4),
    )
    coffee_table = FurnitureItem(
        id="f_table",
        room_id="r1",
        type="coffee_table",
        position=(4.0, 0.0, 5.8),
        rotation_deg=0.0,
        dimensions=(1.0, 0.45, 0.6),
    )
    scene = Scene(
        project_id="test_p",
        version=1,
        rooms=[room],
        furniture=[sofa, tv, coffee_table],
        lighting=[],
        decor=[],
        architect_suggestions=[],
    )

    report = LayoutErgonomicsValidator.validate_scene(scene)
    assert report.is_valid
    assert report.critical_count == 0
    assert report.score_percent >= 95.0


# ==============================================================================
# 5. End-to-End Golden Dataset Benchmark
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("plan_path", [
    "sample_plans/plan1_studio.png",
    "sample_plans/plan2_euro2k.png",
    "sample_plans/plan3_classic2bed.png",
    "sample_plans/plan4_family3bed.png",
    "sample_plans/plan5_loft.png",
])
async def test_golden_dataset_all_sample_plans_pass_validation(plan_path):
    from app.cv.segmentation import segment_floor_plan
    from app.cv.wall_graph import build_wall_graph_from_segmentation
    from app.agents.room_detector.agent import run as detect_rooms
    from app.agents.furniture_planner.agent import run as plan_furniture
    from app.agents.scene_generator.agent import assemble
    from app.models.project import UserPreferences

    with open(plan_path, "rb") as f:
        img_bytes = f.read()

    # Быстрая детерминированная сегментация чертежа
    seg_res = segment_floor_plan(img_bytes)
    wall_graph = build_wall_graph_from_segmentation(seg_res.room_polygons)

    rooms: list[Room] = []
    for i, face in enumerate(wall_graph.rooms):
        walls = [
            Wall(
                id=edge.id,
                start=edge.start,
                end=edge.end,
                thickness=edge.thickness,
                openings=[
                    Opening(
                        type=OpeningType.door if o.type == "door" else OpeningType.window,
                        position=o.position,
                        width=o.width_m,
                    )
                    for o in edge.openings
                ],
            )
            for edge in face.walls
        ]
        rooms.append(Room(id=f"room_{i+1}", polygon=face.polygon, walls=walls))

    rooms = await detect_rooms(rooms)
    prefs = UserPreferences(style="Modern Minimalism")
    furniture = await plan_furniture(rooms, prefs)

    scene = assemble(
        project_id="test_golden",
        version=1,
        rooms=rooms,
        furniture=furniture,
        lighting=[],
        decor=[],
        architect_suggestions=[],
    )

    report = LayoutErgonomicsValidator.validate_scene(scene)
    assert report.is_valid, f"Plan {plan_path} failed validation with critical violations: {report.violations}"
    assert report.score_percent >= 90.0, f"Plan {plan_path} score {report.score_percent}% < 90%"


