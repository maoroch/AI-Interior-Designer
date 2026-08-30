"""
Движок мгновенного сопоставления планировочных архетипов (Zero-LLM Fast-Path Layout Matcher).

Использует топологии CubiCasa5K и LCSF:
- При сходстве формы комнаты с эталоном >= 0.88 генерирует расстановку за < 1 мс без единого запроса к LLM.
- При нестандартной сложной геометрии возвращает None, передавая управление генеративному агенту с Vector RAG.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.knowledge.vector_engine import RoomFeatureVectorizer, _calc_polygon_area
from app.models.scene import Room

logger = logging.getLogger("fast_path_matcher")

TOPOLOGIES_FILE = Path(__file__).parent / "patterns" / "cubicasa_topologies.json"


class FastPathLayoutMatcher:
    """Матчер эталонных планировочных топологий."""

    _archetypes_cache: list[dict[str, Any]] | None = None

    @classmethod
    def load_archetypes(cls) -> list[dict[str, Any]]:
        if cls._archetypes_cache is None:
            if TOPOLOGIES_FILE.exists():
                with open(TOPOLOGIES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls._archetypes_cache = data.get("archetypes", [])
            else:
                cls._archetypes_cache = []
        return cls._archetypes_cache

    @classmethod
    def match_golden_template(
        cls,
        room: Room,
        style: str = "Modern Minimalism",
        similarity_threshold: float = 0.88,
    ) -> tuple[list[dict[str, Any]] | None, float, str | None]:
        """
        Ищет совпадение геометрии комнаты с базой топологий CubiCasa5K.

        Возвращает:
        (matched_items_or_None, best_similarity_score, archetype_id_or_None)
        """
        archetypes = cls.load_archetypes()
        if not archetypes or len(room.polygon) < 3:
            return None, 0.0, None

        # Подсчёт дверей и окон
        door_cnt = sum(len(w.openings) for w in room.walls if any(o.type.value == "door" for o in w.openings))
        win_cnt = sum(len(w.openings) for w in room.walls if any(o.type.value == "window" for o in w.openings))

        r_type = room.type.value if hasattr(room.type, "value") else str(room.type)
        r_vec = RoomFeatureVectorizer.vectorize_room(
            room_polygon=room.polygon,
            room_type=r_type,
            style=style,
            door_count=max(1, door_cnt),
            window_count=max(1, win_cnt),
        )

        area = _calc_polygon_area(room.polygon)

        best_score = 0.0
        best_archetype: dict[str, Any] | None = None

        for arch in archetypes:
            arch_type = arch.get("room_type", "")
            # Проверяем совместимость типов (или студия для большой гостиной)
            if arch_type != r_type and not (r_type == "living_room" and arch_type == "studio" and area > 22.0):
                continue

            # Проверка диапазона площадей
            min_a = arch.get("target_area_sqm_min", 5.0)
            max_a = arch.get("target_area_sqm_max", 60.0)
            if not (min_a * 0.75 <= area <= max_a * 1.35):
                continue

            arch_vec = np.array(arch.get("feature_vector", []), dtype=np.float32)
            if len(arch_vec) != 8:
                continue

            # Нормализация вектора архетипа
            norm = np.linalg.norm(arch_vec)
            if norm > 1e-6:
                arch_vec = arch_vec / norm

            score = RoomFeatureVectorizer.cosine_similarity(r_vec, arch_vec)
            if score > best_score:
                best_score = score
                best_archetype = arch

        if best_archetype and best_score >= similarity_threshold:
            logger.info(
                "🚀 Fast-Path Hit: room '%s' (%s, %.1f sqm) matched archetype '%s' with similarity %.3f >= %.2f (Zero-LLM)",
                room.id,
                r_type,
                area,
                best_archetype.get("id"),
                best_score,
                similarity_threshold,
            )
            raw_layout = best_archetype.get("layout", [])
            adapted_layout = []
            for it in raw_layout:
                it_copy = dict(it)
                it_copy["style"] = style
                adapted_layout.append(it_copy)
            return adapted_layout, round(best_score, 3), best_archetype.get("id")

        return None, round(best_score, 3), None
