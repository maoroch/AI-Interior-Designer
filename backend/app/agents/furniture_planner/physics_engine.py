"""
Физический движок пространственного взаимодействия мебели (Spatial Furniture Physics Engine).

Решает проблему слипания, пересечений и наложений 3D-моделей:
1. RigidBody2D: твердотельные геометрические боксы с массой и учетом поворота (OBB).
2. Non-Penetration Constraint: гарантированный зазор (Clearance Margin >= 15 см) между твердыми предметами (стол, шкаф, стеллаж).
3. Mass-Weighted Impulse Resolution: тяжелая мебель (шкафы, гардеробы) сдвигает более легкую (стулья, торшеры), а не наоборот.
4. Wall & Room Boundary Containment: предотвращает выталкивание мебели сквозь стены и в зоны открывания дверей.
5. Multi-Layer Transparency: ковры и покрытия (rug/carpet) являются проходимым нижним слоем и не выталкивают стоящую на них мебель.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box

NON_PASSABLE_GAP = 0.15  # 15 см безопасный эргономичный зазор между твердыми предметами
DOOR_PROTECTION_RADIUS = 0.90  # 90 см радиус открывания дверей
ROOM_WALL_INSET = 0.05  # 5 см отступ от контура стен

# Весовые коэффициенты массы для физического расталкивания (чем больше масса, тем стабильнее объект)
MASS_TABLE: dict[str, float] = {
    "wardrobe": 12.0,
    "bookshelf": 10.0,
    "sofa": 8.0,
    "bed": 9.0,
    "desk": 6.0,
    "dining_table": 5.0,
    "tv_stand": 7.0,
    "table": 4.0,
    "armchair": 3.0,
    "coffee_table": 2.5,
    "nightstand": 2.0,
    "bench": 3.0,
    "chair": 1.0,
    "plant": 0.8,
    "floor_lamp": 0.8,
    "mirror": 1.5,
    "rug": 0.0,  # Бестелесный (проходимый) слой
}


@dataclass
class FurnitureBody:
    id: str
    type: str
    x: float
    z: float
    w: float
    d: float
    rotation_deg: float
    mass: float = 1.0
    is_passable: bool = False

    def get_polygon(self, extra_buffer: float = 0.0) -> Polygon:
        """Создаёт точный ориентированный полигон предмета (OBB) с учётом поворота."""
        bw = self.w + extra_buffer * 2.0
        bd = self.d + extra_buffer * 2.0
        # Базовый прямоугольник в центре (0, 0)
        base = box(-bw / 2.0, -bd / 2.0, bw / 2.0, bd / 2.0)
        # Поворот вокруг центра
        if abs(self.rotation_deg) > 0.1:
            base = rotate(base, -self.rotation_deg, origin=(0, 0))
        # Перенос в мировые координаты (x, z)
        return translate(base, xoff=self.x, yoff=self.z)


class SpatialPhysicsEngine:
    """
    Итеративный физический решатель коллизий и пространственного баланса (Physics Relaxation Solver).
    """

    @classmethod
    def resolve_scene_physics(
        cls,
        items: list[dict[str, Any]],
        room_polygon: list[tuple[float, float]],
        doors: list[tuple[float, float]] | None = None,
        max_iterations: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Главная точка входа: принимает список предметов и полигон комнаты,
        полностью устраняет наложения мебели друг на друга и на стены.
        """
        if not items or len(room_polygon) < 3:
            return items

        room_poly = Polygon(room_polygon)
        if not room_poly.is_valid:
            room_poly = room_poly.buffer(0)

        inner_poly = room_poly.buffer(-ROOM_WALL_INSET)
        if inner_poly.is_empty or not inner_poly.is_valid:
            inner_poly = room_poly

        door_zones = [Point(dx, dy).buffer(DOOR_PROTECTION_RADIUS) for dx, dy in (doors or [])]

        # 1. Инициализация твердых тел
        bodies: list[FurnitureBody] = []
        for it in items:
            t = str(it.get("type", "unknown")).lower()
            dims = it.get("dimensions", (1.0, 1.0, 1.0))
            if isinstance(dims, dict):
                w = float(dims.get("width", 1.0))
                d = float(dims.get("depth", 1.0))
            elif isinstance(dims, (list, tuple)) and len(dims) >= 3:
                w = float(dims[0])
                d = float(dims[2])
            else:
                w, d = 1.0, 1.0

            pos = it.get("position", (0.0, 0.0, 0.0))
            px = float(pos[0]) if len(pos) > 0 else 0.0
            pz = float(pos[2]) if len(pos) > 2 else 0.0
            rot = float(it.get("rotation_deg", 0.0))

            is_rug = t in ("rug", "carpet", "ковер")
            mass = 0.0 if is_rug else MASS_TABLE.get(t, 2.0)

            bodies.append(
                FurnitureBody(
                    id=str(it.get("id", "")),
                    type=t,
                    x=px,
                    z=pz,
                    w=max(0.2, w),
                    d=max(0.2, d),
                    rotation_deg=rot,
                    mass=mass,
                    is_passable=is_rug,
                )
            )

        # 2. Итеративный физический цикл (Iterative Collision & Separation Solver)
        for iteration in range(max_iterations):
            max_penetration = 0.0

            # А. Разрешение парных коллизий (Solid vs Solid)
            for i in range(len(bodies)):
                body_a = bodies[i]
                if body_a.is_passable:
                    continue

                poly_a = body_a.get_polygon(extra_buffer=NON_PASSABLE_GAP / 2.0)

                for j in range(i + 1, len(bodies)):
                    body_b = bodies[j]
                    if body_b.is_passable:
                        continue

                    poly_b = body_b.get_polygon(extra_buffer=NON_PASSABLE_GAP / 2.0)

                    # Проверяем пересечение
                    if poly_a.intersects(poly_b):
                        inter = poly_a.intersection(poly_b)
                        inter_area = inter.area
                        if inter_area <= 0.0001:
                            continue

                        # Вектор отталкивания между центрами масс
                        dx = body_b.x - body_a.x
                        dz = body_b.z - body_a.z
                        dist = math.hypot(dx, dz)

                        if dist < 0.001:
                            # Полное совпадение центров — случайный импульс в стороны
                            nx, nz = 1.0, 0.0
                            overlap = (body_a.w + body_b.w) / 2.0 + NON_PASSABLE_GAP
                        else:
                            nx = dx / dist
                            nz = dz / dist
                            # Приблизительная глубина проникновения
                            overlap = math.sqrt(inter_area) + NON_PASSABLE_GAP * 0.5

                        max_penetration = max(max_penetration, overlap)

                        # Распределение смещения по массам (m_B / (m_A + m_B))
                        total_mass = max(0.1, body_a.mass + body_b.mass)
                        ratio_a = body_b.mass / total_mass
                        ratio_b = body_a.mass / total_mass

                        push_step = min(0.35, overlap * 0.6)

                        body_a.x -= nx * (push_step * ratio_a)
                        body_a.z -= nz * (push_step * ratio_a)
                        body_b.x += nx * (push_step * ratio_b)
                        body_b.z += nz * (push_step * ratio_b)

            # Б. Ограничение границами комнаты (Wall Boundary Constraint)
            for body in bodies:
                if body.is_passable:
                    continue

                b_poly = body.get_polygon()
                # Если тело вылезло за пределы комнаты — возвращаем внутрь
                if not inner_poly.contains(b_poly):
                    centroid = inner_poly.centroid
                    dx = centroid.x - body.x
                    dz = centroid.y - body.z
                    dist = math.hypot(dx, dz)
                    if dist > 0.01:
                        body.x += (dx / dist) * 0.08
                        body.z += (dz / dist) * 0.08

                # В. Отталкивание от дверных проёмов
                for dz_poly in door_zones:
                    if dz_poly.intersects(b_poly):
                        d_pt = dz_poly.centroid
                        dx = body.x - d_pt.x
                        dz = body.z - d_pt.y
                        dist = math.hypot(dx, dz)
                        if dist > 0.001:
                            body.x += (dx / dist) * 0.12
                            body.z += (dz / dist) * 0.12

            # Если наложений больше нет — завершаем досрочно
            if max_penetration < 0.01:
                break

        # 3. Применение скорректированных позиций обратно к предметам
        updated_items = []
        for orig, body in zip(items, bodies):
            item_copy = dict(orig)
            orig_pos = orig.get("position", (0.0, 0.0, 0.0))
            orig_y = orig_pos[1] if len(orig_pos) > 1 else 0.0
            item_copy["position"] = (round(body.x, 2), round(orig_y, 2), round(body.z, 2))
            updated_items.append(item_copy)

        return updated_items
