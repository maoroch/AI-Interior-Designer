"""Тесты сборки финальной сцены (Scene Generator) и ретраев LLM-обёртки."""
import pytest

from app.agents.scene_generator.agent import assemble
from app.core.llm import LLMCallError, _with_retries
from app.models.scene import ArchitectSuggestion, Room


def test_assemble_produces_valid_scene_with_all_parts():
    rooms = [Room(id="r1", polygon=[(0, 0), (3, 0), (3, 3), (0, 3)])]
    scene = assemble(
        project_id="proj_1",
        version=2,
        rooms=rooms,
        furniture=[],
        lighting=[],
        decor=[],
        architect_suggestions=[ArchitectSuggestion(id="opt_a", title="t", description="d")],
        architect_choice="opt_a",
        style="Modern Minimalism",
    )
    assert scene.project_id == "proj_1"
    assert scene.version == 2
    assert scene.architect_choice == "opt_a"
    assert scene.style == "Modern Minimalism"
    assert len(scene.rooms) == 1


async def test_with_retries_succeeds_after_transient_failures():
    calls = {"count": 0}

    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary network error")
        return "ok"

    # Ускоряем тест — не ждём реальные секунды задержки между попытками.
    import app.core.llm as llm_module

    llm_module.RETRY_BASE_DELAY_SECONDS = 0.01

    result = await _with_retries(flaky, description="test")
    assert result == "ok"
    assert calls["count"] == 3


async def test_with_retries_raises_llm_call_error_after_exhausting_attempts():
    async def always_fails():
        raise RuntimeError("permanent failure")

    import app.core.llm as llm_module

    llm_module.RETRY_BASE_DELAY_SECONDS = 0.01

    with pytest.raises(LLMCallError):
        await _with_retries(always_fails, description="test")
