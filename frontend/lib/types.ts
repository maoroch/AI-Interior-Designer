// Типы синхронизированы вручную с backend/app/models/scene.py и project.py.
// Это единственный "контракт", который должен обновляться при изменении Pydantic-схем.

export type Point2D = [number, number];
export type Point3D = [number, number, number];

export type RoomType =
  | "kitchen"
  | "living_room"
  | "bedroom"
  | "bathroom"
  | "hallway"
  | "office"
  | "dining_room"
  | "kids_room"
  | "unknown";

export interface Opening {
  type: "door" | "window";
  position: number; // 0..1 вдоль стены
  width: number;
  opens_inward?: boolean | null;
}

export interface Wall {
  id: string;
  start: Point2D;
  end: Point2D;
  thickness: number;
  openings: Opening[];
}

export interface Room {
  id: string;
  type: RoomType;
  polygon: Point2D[];
  height: number;
  walls: Wall[];
  floor_material: string | null;
  wall_color: string | null;
  label: string | null;
}

export interface FurnitureItem {
  id: string;
  room_id: string | null;
  type: string;
  style: string | null;
  position: Point3D;
  rotation_deg: number;
  dimensions: Point3D;
  material: string | null;
  color: string | null;
  model_ref: string | null;
}

export interface LightSource {
  id: string;
  type: string;
  position: Point3D;
  color_temperature_k: number;
  intensity: number;
}

export interface DecorItem {
  id: string;
  type: string;
  room_id: string | null;
  position: Point3D;
  rotation_deg: number;
  scale: number;
}

export interface ArchitectSuggestion {
  id: string;
  title: string;
  description: string;
  rooms_after: Room[] | null;
}

export interface Scene {
  project_id: string;
  version: number;
  variant_id: string;
  variant_label: string;
  rooms: Room[];
  furniture: FurnitureItem[];
  lighting: LightSource[];
  decor: DecorItem[];
  architect_suggestions: ArchitectSuggestion[];
  architect_choice: string | null;
  style: string | null;
}

export type PipelineStage =
  | "uploaded"
  | "analyzing_floorplan"
  | "detecting_rooms"
  | "awaiting_architect_decision"
  | "designing_interior"
  | "planning_furniture"
  | "designing_lighting"
  | "adding_decor"
  | "generating_scene"
  | "ready"
  | "failed";

export interface UserPreferences {
  style?: string | null;
  budget_usd?: number | null;
  adults: number;
  children: number;
  pets: string[];
  favorite_colors: string[];
  needs_office: boolean;
  likes_hosting_guests: boolean;
}

export interface Project {
  id: string;
  created_at: string;
  updated_at: string;
  original_plan_key: string | null;
  stage: PipelineStage;
  error: string | null;
  preferences: UserPreferences | null;
  active_variant_id: string;
}

export interface ChatMessage {
  id: string;
  project_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  applied_patch: Record<string, unknown> | null;
}
