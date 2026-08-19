"""
Тесты применения патчей сцены из чата (app/agents/conversation/agent.py).
interpret() (вызов LLM) не тестируется здесь — только apply_operations,
чистая функция без сети.
"""
from app.agents.conversation.agent import apply_operations
from app.models.scene import FurnitureItem, LightSource, Scene


def _make_scene() -> Scene:
    return Scene(
        project_id="proj_1",
        version=1,
        furniture=[
            FurnitureItem(
                id="f_1", type="sofa", position=(1, 0, 1), dimensions=(2, 0.8, 0.9), material="grey_fabric"
            )
        ],
        lighting=[LightSource(id="l_1", type="ceiling", position=(2, 2.5, 2), intensity=0.9)],
    )


def test_remove_furniture_by_id():
    scene = _make_scene()
    scene = apply_operations(scene, [{"op": "remove", "collection": "furniture", "target_id": "f_1"}])
    assert scene.furniture == []


def test_update_furniture_material():
    scene = _make_scene()
    scene = apply_operations(
        scene,
        [{"op": "update", "collection": "furniture", "target_id": "f_1", "fields": {"material": "leather"}}],
    )
    assert scene.furniture[0].material == "leather"


def test_add_new_decor_item():
    scene = _make_scene()
    scene = apply_operations(
        scene,
        [
            {
                "op": "add",
                "collection": "decor",
                "target_id": None,
                "fields": {"type": "plant", "position": (3, 0, 3)},
            }
        ],
    )
    assert len(scene.decor) == 1
    assert scene.decor[0].type == "plant"


def test_update_all_lighting_intensity_for_make_lighter_command():
    scene = _make_scene()
    scene = apply_operations(
        scene,
        [{"op": "update", "collection": "lighting", "target_id": "l_1", "fields": {"intensity": 1.5}}],
    )
    assert scene.lighting[0].intensity == 1.5


def test_invalid_add_fields_are_ignored_without_crashing():
    scene = _make_scene()
    # "type" отсутствует — обязательное поле FurnitureItem; не должно падать с исключением.
    scene = apply_operations(
        scene, [{"op": "add", "collection": "furniture", "target_id": None, "fields": {"material": "wood"}}]
    )
    assert len(scene.furniture) == 1  # ничего не добавилось, но и не упало


def test_version_increments_after_patch():
    scene = _make_scene()
    original_version = scene.version
    scene = apply_operations(scene, [])
    assert scene.version == original_version + 1
