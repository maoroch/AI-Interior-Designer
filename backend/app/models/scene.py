"""
Схема сцены — центральный контракт между всеми агентами, MongoDB и Three.js-рендером.

Каждый агент пайплайна дополняет один и тот же объект Scene:
  FloorPlanAnalyzer  -> rooms[].polygon, walls, height
  RoomDetector       -> rooms[].type
  ArchitectAgent     -> architect_suggestions[], может переписать rooms[].polygon/walls
  InteriorDesigner   -> rooms[].floor_material, rooms[].wall_color, style
  FurniturePlanner   -> furniture[]
  LightingDesigner   -> lighting[]
  Decorator          -> decor[]
  SceneGenerator     -> собирает финальную версию, инкрементирует version

Conversation Agent модифицирует сцену точечными патчами (JSON Patch-подобные операции),
не пересобирая весь пайплайн заново.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


class RoomType(str, Enum):
    kitchen = "kitchen"
    living_room = "living_room"
    bedroom = "bedroom"
    bathroom = "bathroom"
    hallway = "hallway"
    office = "office"
    dining_room = "dining_room"
    kids_room = "kids_room"
    unknown = "unknown"


class OpeningType(str, Enum):
    door = "door"
    window = "window"


class Opening(BaseModel):
    type: OpeningType
    position: float = Field(..., ge=0, le=1, description="Позиция проёма вдоль стены, 0..1")
    width: float = Field(..., gt=0, description="Ширина проёма в метрах")
    opens_inward: bool | None = Field(default=None, description="Только для дверей")


class Wall(BaseModel):
    id: str
    start: Point2D
    end: Point2D
    thickness: float = 0.2
    openings: list[Opening] = Field(default_factory=list)


class Room(BaseModel):
    id: str
    type: RoomType = RoomType.unknown
    polygon: list[Point2D]
    height: float = 2.7
    walls: list[Wall] = Field(default_factory=list)
    floor_material: str | None = None
    wall_color: str | None = None
    label: str | None = Field(default=None, description="Человекочитаемое имя, если нужно уточнение")


class FurnitureItem(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    room_id: str | None = None
    type: str = Field(..., description="sofa, bed, dining_table, wardrobe, ...")
    style: str | None = None
    position: Point3D
    rotation_deg: float = 0
    dimensions: Point3D = Field(..., description="width, height, depth в метрах")
    material: str | None = None
    color: str | None = None
    model_ref: str | None = Field(
        default=None, description="Ссылка на GLTF-модель (фаза 2); None => рендерится примитивом"
    )


class LightSource(BaseModel):
    id: str
    type: str = Field(..., description="ceiling, floor_lamp, wall_sconce, natural")
    position: Point3D
    color_temperature_k: int = 3000
    intensity: float = 1.0


class DecorItem(BaseModel):
    id: str
    type: str = Field(..., description="plant, rug, painting, mirror, ...")
    room_id: str | None = None
    position: Point3D
    rotation_deg: float = 0
    scale: float = 1.0


class ArchitectSuggestion(BaseModel):
    id: str
    title: str
    description: str
    rooms_after: list[Room] | None = Field(
        default=None, description="Полный набор комнат, если вариант будет принят"
    )


class Scene(BaseModel):
    project_id: str
    version: int = 1
    variant_id: str = "variant_a"
    variant_label: str = "Вариант A"
    rooms: list[Room] = Field(default_factory=list)
    furniture: list[FurnitureItem] = Field(default_factory=list)
    lighting: list[LightSource] = Field(default_factory=list)
    decor: list[DecorItem] = Field(default_factory=list)
    architect_suggestions: list[ArchitectSuggestion] = Field(default_factory=list)
    architect_choice: str | None = Field(
        default=None, description="id выбранного варианта, либо null если 'оставить как есть'"
    )
    style: str | None = None
