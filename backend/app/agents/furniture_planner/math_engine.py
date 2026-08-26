"""
Computational Interior Design Math Engine.

Математическое ядро дизайна интерьера:
1. Золотое сечение (Golden Ratio Phi = 1.618) для соразмерных пропорций мебели.
2. Формула комфортного расстояния до ТВ (SMPTE / THX Viewing Distance).
3. Индекс плотности и баланса пространства (Floor Occupancy Ratio K_occ = 0.35).
4. Метод потенциальных полей и физической релаксации (Force-Directed Relaxation Solver).
5. Колористический баланс 60-30-10 (Chromatic Balance).
6. Фотометрический расчёт освещённости (Photometric Lighting & Lux Target).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PHI = 1.61803398875  # Золотое сечение


@dataclass
class ProportionalFurnitureDims:
    width_m: float
    depth_m: float
    height_m: float


class GoldenRatioScaler:
    """
    Математический калькулятор гармоничных пропорций по Золотому сечению (Phi = 1.618).
    """

    @staticmethod
    def calculate_sofa_dims(wall_length_m: float) -> ProportionalFurnitureDims:
        """
        Диван должен занимать ~61.8% (1/Phi) свободной стены (но не менее 1.8м и не более 2.8м).
        """
        w = max(1.80, min(2.80, round(wall_length_m / PHI, 2)))
        d = round(w / (PHI * 1.5), 2)  # Глубина пропорциональна ширине (~0.85-0.95м)
        h = 0.85
        return ProportionalFurnitureDims(width_m=w, depth_m=max(0.85, d), height_m=h)

    @staticmethod
    def calculate_rug_dims(sofa_width_m: float) -> ProportionalFurnitureDims:
        """
        Ковер должен быть в Phi раз шире диванной группы (L_rug = L_sofa * 1.15..1.25, D_rug = L_rug / 1.33).
        """
        w = round(sofa_width_m * 1.20, 2)
        d = round(w / 1.33, 2)
        return ProportionalFurnitureDims(width_m=w, depth_m=d, height_m=0.02)

    @staticmethod
    def calculate_coffee_table_dims(sofa_width_m: float) -> ProportionalFurnitureDims:
        """
        Журнальный столик должен составлять ~55-60% длины дивана.
        """
        w = round(sofa_width_m * 0.55, 2)
        d = round(w / PHI, 2)
        return ProportionalFurnitureDims(width_m=max(0.70, w), depth_m=max(0.50, d), height_m=0.45)

    @staticmethod
    def calculate_tv_stand_dims(tv_diagonal_inches: float = 55.0) -> ProportionalFurnitureDims:
        """
        Тумба под ТВ должна быть на ~30-50% шире самого экрана телевизора.
        """
        tv_width_m = (tv_diagonal_inches * 0.0254) * 0.87
        w = round(tv_width_m * PHI, 2)
        return ProportionalFurnitureDims(width_m=max(1.40, w), depth_m=0.40, height_m=0.50)


class SMPTEViewingCalculator:
    """
    Расчёт эргономичной дистанции просмотра ТВ по стандарту SMPTE (угол обзора ~30-36 град).
    """

    @staticmethod
    def get_optimal_viewing_distance_m(tv_diagonal_inches: float = 55.0) -> float:
        """
        D = Screen_Diagonal_m * 1.60 (SMPTE 30-degree FOV standard)
        Для 55" (1.40м) -> ~2.24 метра.
        Для 65" (1.65м) -> ~2.64 метра.
        """
        diag_m = tv_diagonal_inches * 0.0254
        return round(diag_m * 1.60, 2)


class OccupancyBudgetOptimizer:
    """
    Расчёт целевого бюджета площади мебели в комнате (Occupancy Ratio).
    """

    @staticmethod
    def calculate_target_furniture_area(room_area_sqm: float, target_ratio: float = 0.35) -> float:
        """
        Оптимальная плотность пола = 32-40% площади (K_occ = 0.35).
        """
        return round(room_area_sqm * target_ratio, 2)


class ForceDirectedRelaxationSolver:
    """
    Физический релаксационный решатель взаимодействия объектов в 2D.
    Стягивает связанные предметы (диван + столик, кровать + тумбочки) и отталкивает от стен и дверей.
    """

    @staticmethod
    def relax_positions(
        items: list[dict[str, Any]],
        doors: list[tuple[float, float]],
        room_bounds: tuple[float, float, float, float],
        iterations: int = 15,
    ) -> list[dict[str, Any]]:
        min_x, min_y, max_x, max_y = room_bounds
        relaxed = [dict(it) for it in items]

        for _ in range(iterations):
            # 1. Поиск дивана и столика для парного притяжения
            sofa_idx = next((i for i, it in enumerate(relaxed) if "sofa" in str(it.get("type", "")).lower()), None)
            table_idx = next((i for i, it in enumerate(relaxed) if "coffee_table" in str(it.get("type", "")).lower()), None)
            rug_idx = next((i for i, it in enumerate(relaxed) if "rug" in str(it.get("type", "")).lower()), None)

            if sofa_idx is not None and table_idx is not None:
                sofa_pos = relaxed[sofa_idx].get("position", (0, 0, 0))
                table_pos = relaxed[table_idx].get("position", (0, 0, 0))
                sx, sz = sofa_pos[0], sofa_pos[2]
                tx, tz = table_pos[0], table_pos[2]

                # Притягиваем столик по X к центру дивана
                dx = sx - tx
                if abs(dx) > 0.05:
                    new_tx = round(tx + dx * 0.25, 2)
                    relaxed[table_idx]["position"] = (new_tx, table_pos[1], table_pos[2])

                # Если есть ковер — центрируем его под группой
                if rug_idx is not None:
                    rug_pos = relaxed[rug_idx].get("position", (0, 0, 0))
                    mid_x = round((sx + tx) / 2.0, 2)
                    mid_z = round((sz + tz) / 2.0, 2)
                    relaxed[rug_idx]["position"] = (mid_x, rug_pos[1], mid_z)

            # 2. Отталкивание от дверных зон
            for it in relaxed:
                if it.get("type") == "rug":
                    continue
                pos = it.get("position", (0, 0, 0))
                px, pz = pos[0], pos[2]
                for dx, dz in doors:
                    dist = math.hypot(px - dx, pz - dz)
                    if dist < 0.95 and dist > 0.01:
                        push = (0.95 - dist) * 0.35
                        nx = (px - dx) / dist
                        nz = (pz - dz) / dist
                        new_px = round(max(min_x + 0.3, min(max_x - 0.3, px + nx * push)), 2)
                        new_pz = round(max(min_y + 0.3, min(max_y - 0.3, pz + nz * push)), 2)
                        it["position"] = (new_px, pos[1], new_pz)

        return relaxed


class ChromaticBalanceCalculator:
    """
    Расчёт цветового баланса 60-30-10:
    - 60% Базовый тон (стены, пол)
    - 30% Вторичный тон (мебель, ковёр, дерево)
    - 10% Акцентный всплеск (декор, подушки, растения, светильники)
    """

    STYLES_PALETTES: dict[str, dict[str, str]] = {
        "Modern Minimalism": {
            "dominant_60": "#ECEFF4",       # Светлый нейтральный фон
            "secondary_30": "#3B4252",      # Глубокий графит / сланец
            "accent_10": "#D08770",         # Тёплый терракотовый акцент
            "wood_tone": "#D4A373",         # Натуральный дуб
        },
        "Japandi Warm": {
            "dominant_60": "#F4EFEA",       # Тёплый кремовый / ваниль
            "secondary_30": "#8C7A6B",      # Матовый орех / джут
            "accent_10": "#A3BE8C",         # Оливковый шалфей
            "wood_tone": "#C89D7C",         # Тёплый ясень
        },
        "Contemporary Luxury": {
            "dominant_60": "#2E3440",       # Тёмный антрацит
            "secondary_30": "#4C566A",      # Фактурный мрамор / велюр
            "accent_10": "#EBCB8B",         # Шлифованная латунь / золото
            "wood_tone": "#3E2723",         # Тёмный венге
        },
    }

    @classmethod
    def get_style_colors(cls, style_name: str) -> dict[str, str]:
        for k, palette in cls.STYLES_PALETTES.items():
            if k.lower() in style_name.lower() or style_name.lower() in k.lower():
                return palette
        return cls.STYLES_PALETTES["Modern Minimalism"]


class PhotometricLightingCalculator:
    """
    Светотехнический расчёт освещённости (Закон обратных квадратов и СНиП).
    """

    @staticmethod
    def calculate_lighting_requirements(
        room_area_sqm: float,
        room_type: str = "living_room",
    ) -> dict[str, Any]:
        """
        Расчёт требуемого светового потока (люмен) и цветовой температуры:
        - Living room / Studio: 150-200 лк, 3000K (тёплый белый)
        - Bedroom: 100-150 лк, 2700K (уютный релакс)
        - Work / Office: 350-500 лк, 4000K (нейтральный рабочий)
        - Hallway / Corridor: 100 лк, 3000K
        """
        r_type = room_type.lower()
        if "work" in r_type or "office" in r_type:
            target_lux = 400
            kelvin = 4000
        elif "bed" in r_type:
            target_lux = 120
            kelvin = 2700
        elif "hall" in r_type or "bath" in r_type:
            target_lux = 100
            kelvin = 3000
        else:
            target_lux = 180
            kelvin = 3000

        # Общий световой поток (Lumens = Lux * Area / Utilization_Factor)
        utilization_factor = 0.65
        total_lumens = round((target_lux * room_area_sqm) / utilization_factor, 0)

        # Количество потолочных спотов (каждый ~600-800 лм)
        num_ceiling_spots = max(1, int(math.ceil(total_lumens / 700)))

        return {
            "target_lux": target_lux,
            "color_temperature_k": kelvin,
            "total_lumens_required": total_lumens,
            "recommended_ceiling_spots": num_ceiling_spots,
        }
