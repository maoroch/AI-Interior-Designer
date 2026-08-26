"""
Unit and Integration tests for the Computational Interior Design Math Engine:
- GoldenRatioScaler (Phi = 1.618)
- SMPTEViewingCalculator
- OccupancyBudgetOptimizer (K_occ = 0.35)
- ForceDirectedRelaxationSolver
- ChromaticBalanceCalculator (60-30-10)
- PhotometricLightingCalculator (Lux / Lumens / Kelvin)
"""
import pytest
from app.agents.furniture_planner.math_engine import (
    ChromaticBalanceCalculator,
    ForceDirectedRelaxationSolver,
    GoldenRatioScaler,
    OccupancyBudgetOptimizer,
    PhotometricLightingCalculator,
    SMPTEViewingCalculator,
    PHI,
)


def test_golden_ratio_constant_value():
    assert round(PHI, 4) == 1.6180


def test_golden_ratio_sofa_scaling():
    # На стене 4.5м диван должен быть ~2.78м (4.5 / 1.618)
    dims = GoldenRatioScaler.calculate_sofa_dims(wall_length_m=4.5)
    assert 2.5 <= dims.width_m <= 2.8
    assert dims.depth_m >= 0.85
    assert dims.height_m == 0.85


def test_golden_ratio_rug_and_table_proportions():
    sofa_w = 2.40
    rug_dims = GoldenRatioScaler.calculate_rug_dims(sofa_width_m=sofa_w)
    table_dims = GoldenRatioScaler.calculate_coffee_table_dims(sofa_width_m=sofa_w)

    # Ковер шире дивана
    assert rug_dims.width_m > sofa_w
    assert rug_dims.height_m == 0.02

    # Столик пропорционален половине дивана
    assert 1.0 <= table_dims.width_m <= 1.4
    assert table_dims.height_m <= 0.50


def test_smpte_viewing_distance_formula():
    # Для ТВ 55" расстояние ~2.2 - 2.5 метра
    dist_55 = SMPTEViewingCalculator.get_optimal_viewing_distance_m(55.0)
    assert 2.1 <= dist_55 <= 2.6

    # Для ТВ 65" расстояние ~2.6 - 3.0 метра
    dist_65 = SMPTEViewingCalculator.get_optimal_viewing_distance_m(65.0)
    assert 2.6 <= dist_65 <= 3.0


def test_occupancy_budget_optimizer():
    area = 30.0
    budget = OccupancyBudgetOptimizer.calculate_target_furniture_area(area, target_ratio=0.35)
    assert budget == 10.5  # 30 * 0.35


def test_force_directed_relaxation_repels_from_door():
    items = [
        {"type": "sofa", "position": (2.0, 0.0, 2.0)},
        {"type": "coffee_table", "position": (2.8, 0.0, 2.0)},
    ]
    # Дверь прямо у дивана (2.1, 2.1)
    doors = [(2.1, 2.1)]
    bounds = (0.0, 0.0, 8.0, 8.0)

    relaxed = ForceDirectedRelaxationSolver.relax_positions(items, doors, bounds, iterations=10)
    new_sofa_pos = relaxed[0]["position"]

    # Диван отодвинулся от двери
    dist = ((new_sofa_pos[0] - 2.1)**2 + (new_sofa_pos[2] - 2.1)**2)**0.5
    assert dist >= 0.3


def test_chromatic_balance_palettes():
    modern = ChromaticBalanceCalculator.get_style_colors("Modern Minimalism")
    assert "dominant_60" in modern
    assert "secondary_30" in modern
    assert "accent_10" in modern
    assert modern["dominant_60"].startswith("#")


def test_photometric_lighting_requirements():
    # Гостиная 25 кв.м
    living = PhotometricLightingCalculator.calculate_lighting_requirements(25.0, "living_room")
    assert living["target_lux"] == 180
    assert living["color_temperature_k"] == 3000
    assert living["total_lumens_required"] > 5000
    assert living["recommended_ceiling_spots"] >= 4

    # Рабочий кабинет 15 кв.м
    office = PhotometricLightingCalculator.calculate_lighting_requirements(15.0, "office")
    assert office["target_lux"] == 400
    assert office["color_temperature_k"] == 4000
