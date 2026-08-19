"use client";

import { ContactShadows, OrbitControls, PointerLockControls, useGLTF } from "@react-three/drei";
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import React, { Suspense, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { useAppStore } from "@/lib/store";
import type { DecorItem, FurnitureItem, LightSource, Room, Scene } from "@/lib/types";

// Палитра материалов и цветов для процедурного PBR-рендера
export const MATERIAL_COLORS: Record<string, string> = {
  oak_wood: "#c29b68",
  walnut: "#4a3222",
  grey_fabric: "#7a8288",
  beige_fabric: "#d8cebf",
  navy_fabric: "#2c3e50",
  white: "#f8f7f4",
  grey: "#8e9297",
  black: "#222222",
  terracotta: "#c86446",
  marble: "#e3e3e3",
  brass: "#d4af37",
  leather_brown: "#6f4423",
};

export const MATERIAL_PRESETS = [
  { label: "Дуб", value: "oak_wood", hex: "#c29b68" },
  { label: "Орех", value: "walnut", hex: "#4a3222" },
  { label: "Серая ткань", value: "grey_fabric", hex: "#7a8288" },
  { label: "Беж", value: "beige_fabric", hex: "#d8cebf" },
  { label: "Кожа", value: "leather_brown", hex: "#6f4423" },
  { label: "Белый", value: "white", hex: "#f8f7f4" },
  { label: "Антрацит", value: "black", hex: "#222222" },
  { label: "Латунь", value: "brass", hex: "#d4af37" },
];

function colorFor(name: string | null | undefined, fallback: string): string {
  if (!name) return fallback;
  return MATERIAL_COLORS[name] ?? (name.startsWith("#") ? name : fallback);
}

function polygonCenter(polygon: [number, number][]): [number, number] {
  const xs = polygon.map((p) => p[0]);
  const ys = polygon.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
}

/** Создаёт процедурную текстуру деревянного паркета */
function createWoodTexture(): THREE.CanvasTexture {
  if (typeof document === "undefined") return new THREE.CanvasTexture(document.createElement("canvas"));
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#c29b68";
    ctx.fillRect(0, 0, 512, 512);
    // Рисуем доски
    const plankHeight = 32;
    for (let y = 0; y < 512; y += plankHeight) {
      const shift = ((y / plankHeight) % 2) * 128;
      for (let x = -128; x < 512; x += 256) {
        ctx.strokeStyle = "rgba(70, 45, 20, 0.25)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + shift, y, 256, plankHeight);
        // Тонкие древесные волокна
        ctx.fillStyle = "rgba(100, 65, 30, 0.06)";
        ctx.fillRect(x + shift + 2, y + 2, 252, plankHeight - 4);
      }
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  return tex;
}

function RoomFloor({ room }: { room: Room }) {
  const xs = room.polygon.map((p) => p[0]);
  const ys = room.polygon.map((p) => p[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const depth = Math.max(...ys) - Math.min(...ys);
  const [cx, cy] = polygonCenter(room.polygon);

  const isWood = room.floor_material?.includes("wood") || !room.floor_material;
  const woodTexture = useMemo(() => (isWood ? createWoodTexture() : null), [isWood]);

  return (
    <mesh position={[cx, 0, cy]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[width, depth]} />
      <meshStandardMaterial
        color={colorFor(room.floor_material, "#c8baa6")}
        map={woodTexture ?? undefined}
        roughness={0.4}
        metalness={0.05}
      />
    </mesh>
  );
}

function RoomWalls({ room }: { room: Room }) {
  const wallColor = colorFor(room.wall_color, "#ece7dd");
  const plinthColor = "#443428"; // Цвет напольного плинтуса

  return (
    <>
      {room.walls.map((wall) => {
        const [x1, y1] = wall.start;
        const [x2, y2] = wall.end;
        const length = Math.hypot(x2 - x1, y2 - y1);
        const angle = Math.atan2(y2 - y1, x2 - x1);

        if (wall.openings.length === 0) {
          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          return (
            <group key={wall.id}>
              {/* Основная стена */}
              <mesh position={[midX, room.height / 2, midY]} rotation={[0, -angle, 0]} castShadow receiveShadow>
                <boxGeometry args={[length, room.height, wall.thickness]} />
                <meshStandardMaterial color={wallColor} roughness={0.85} />
              </mesh>
              {/* Декоративный плинтус у основания */}
              <mesh position={[midX, 0.05, midY]} rotation={[0, -angle, 0]}>
                <boxGeometry args={[length, 0.1, wall.thickness + 0.02]} />
                <meshStandardMaterial color={plinthColor} roughness={0.5} />
              </mesh>
            </group>
          );
        }

        // Стена с вырезанными проёмами для дверей и окон
        const sorted = [...wall.openings].sort((a, b) => a.position - b.position);
        const segments: { from: number; to: number }[] = [];
        let cursor = 0;
        for (const opening of sorted) {
          const openingCenter = opening.position * length;
          const halfWidth = opening.width / 2;
          const openingStart = Math.max(0, openingCenter - halfWidth);
          const openingEnd = Math.min(length, openingCenter + halfWidth);
          if (openingStart > cursor) segments.push({ from: cursor, to: openingStart });
          cursor = Math.max(cursor, openingEnd);
        }
        if (cursor < length) segments.push({ from: cursor, to: length });

        return (
          <group key={wall.id} position={[x1, 0, y1]} rotation={[0, -angle, 0]}>
            {segments.map((seg, i) => {
              const segLength = seg.to - seg.from;
              if (segLength <= 0.02) return null;
              const segCenter = (seg.from + seg.to) / 2;
              return (
                <group key={i}>
                  <mesh position={[segCenter, room.height / 2, 0]} castShadow receiveShadow>
                    <boxGeometry args={[segLength, room.height, wall.thickness]} />
                    <meshStandardMaterial color={wallColor} roughness={0.85} />
                  </mesh>
                  {/* Плинтус на сплошных сегментах */}
                  <mesh position={[segCenter, 0.05, 0]}>
                    <boxGeometry args={[segLength, 0.1, wall.thickness + 0.02]} />
                    <meshStandardMaterial color={plinthColor} roughness={0.5} />
                  </mesh>
                </group>
              );
            })}
            {/* Верхняя перемычка над всеми проёмами (двери и окна) */}
            {sorted.map((o, i) => {
              const center = o.position * length;
              const topLintelH = Math.max(0.2, room.height - 2.1);
              const topLintelCenterY = room.height - topLintelH / 2;
              return (
                <mesh key={`lintel_${i}`} position={[center, topLintelCenterY, 0]} castShadow receiveShadow>
                  <boxGeometry args={[o.width, topLintelH, wall.thickness]} />
                  <meshStandardMaterial color={wallColor} roughness={0.85} />
                </mesh>
              );
            })}

            {/* Окна: подоконный простенок, стекло и подоконник */}
            {sorted
              .filter((o) => o.type === "window")
              .map((o, i) => {
                const center = o.position * length;
                const sillH = 0.85;
                const windowH = 1.25;
                return (
                  <group key={`win_${i}`}>
                    {/* Стена под подоконником */}
                    <mesh position={[center, sillH / 2, 0]} castShadow receiveShadow>
                      <boxGeometry args={[o.width, sillH, wall.thickness]} />
                      <meshStandardMaterial color={wallColor} roughness={0.85} />
                    </mesh>
                    {/* Плинтус под окном */}
                    <mesh position={[center, 0.05, 0]}>
                      <boxGeometry args={[o.width, 0.1, wall.thickness + 0.02]} />
                      <meshStandardMaterial color={plinthColor} roughness={0.5} />
                    </mesh>
                    {/* Стекло */}
                    <mesh position={[center, sillH + windowH / 2, 0]}>
                      <boxGeometry args={[o.width - 0.06, windowH - 0.06, wall.thickness * 0.2]} />
                      <meshPhysicalMaterial
                        color="#a8d5e5"
                        transparent
                        opacity={0.35}
                        roughness={0.08}
                        transmission={0.85}
                        thickness={0.1}
                      />
                    </mesh>
                    {/* Подоконник */}
                    <mesh position={[center, sillH + 0.02, 0]} castShadow>
                      <boxGeometry args={[o.width + 0.08, 0.04, wall.thickness + 0.08]} />
                      <meshStandardMaterial color="#ffffff" roughness={0.3} />
                    </mesh>
                  </group>
                );
              })}

            {/* Двери: наличники и открытый проход в полу */}
            {sorted
              .filter((o) => o.type === "door")
              .map((o, i) => {
                const center = o.position * length;
                const doorH = 2.1;
                const frameThickness = 0.04;
                return (
                  <group key={`door_frame_${i}`}>
                    {/* Левый косяк */}
                    <mesh position={[center - o.width / 2 + frameThickness / 2, doorH / 2, 0]} castShadow>
                      <boxGeometry args={[frameThickness, doorH, wall.thickness + 0.02]} />
                      <meshStandardMaterial color="#3d332a" roughness={0.4} />
                    </mesh>
                    {/* Правый косяк */}
                    <mesh position={[center + o.width / 2 - frameThickness / 2, doorH / 2, 0]} castShadow>
                      <boxGeometry args={[frameThickness, doorH, wall.thickness + 0.02]} />
                      <meshStandardMaterial color="#3d332a" roughness={0.4} />
                    </mesh>
                    {/* Верхняя перекладина рамы двери */}
                    <mesh position={[center, doorH - frameThickness / 2, 0]} castShadow>
                      <boxGeometry args={[o.width, frameThickness, wall.thickness + 0.02]} />
                      <meshStandardMaterial color="#3d332a" roughness={0.4} />
                    </mesh>
                  </group>
                );
              })}
          </group>
        );
      })}
    </>
  );
}

// -------------------------------------------------------------
// Высокодетализированные процедурные 3D-модели мебели
// -------------------------------------------------------------

function Sofa3D({ w, h, d, color }: { w: number; h: number; d: number; color: string }) {
  const seatH = h * 0.45;
  const backH = h * 0.55;
  const armW = Math.min(0.2, w * 0.12);
  const cushionW = (w - armW * 2) / 2;

  return (
    <group>
      {/* Основание / ножки */}
      <mesh position={[-w * 0.4, 0.05, -d * 0.35]} castShadow>
        <cylinderGeometry args={[0.025, 0.015, 0.1, 8]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[w * 0.4, 0.05, -d * 0.35]} castShadow>
        <cylinderGeometry args={[0.025, 0.015, 0.1, 8]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[-w * 0.4, 0.05, d * 0.35]} castShadow>
        <cylinderGeometry args={[0.025, 0.015, 0.1, 8]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[w * 0.4, 0.05, d * 0.35]} castShadow>
        <cylinderGeometry args={[0.025, 0.015, 0.1, 8]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Сиденье (подушки) */}
      <mesh position={[-cushionW / 2, seatH / 2 + 0.08, 0.05]} castShadow receiveShadow>
        <boxGeometry args={[cushionW - 0.02, seatH * 0.7, d * 0.8]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>
      <mesh position={[cushionW / 2, seatH / 2 + 0.08, 0.05]} castShadow receiveShadow>
        <boxGeometry args={[cushionW - 0.02, seatH * 0.7, d * 0.8]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>

      {/* Спинка */}
      <mesh position={[0, seatH + backH / 2, -d * 0.38]} castShadow receiveShadow>
        <boxGeometry args={[w - 0.02, backH, d * 0.24]} />
        <meshStandardMaterial color={color} roughness={0.75} />
      </mesh>

      {/* Подлокотники */}
      <mesh position={[-w / 2 + armW / 2, h * 0.4, 0]} castShadow>
        <boxGeometry args={[armW, h * 0.65, d * 0.95]} />
        <meshStandardMaterial color={color} roughness={0.75} />
      </mesh>
      <mesh position={[w / 2 - armW / 2, h * 0.4, 0]} castShadow>
        <boxGeometry args={[armW, h * 0.65, d * 0.95]} />
        <meshStandardMaterial color={color} roughness={0.75} />
      </mesh>
    </group>
  );
}

function Table3D({ w, h, d, color }: { w: number; h: number; d: number; color: string }) {
  const topThickness = 0.04;
  const legR = 0.025;
  const legH = h - topThickness;

  return (
    <group>
      {/* Столешница */}
      <mesh position={[0, h - topThickness / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[w, topThickness, d]} />
        <meshStandardMaterial color={color} roughness={0.3} metalness={0.1} />
      </mesh>
      {/* 4 ножки */}
      {[
        [-w / 2 + 0.08, -d / 2 + 0.08],
        [w / 2 - 0.08, -d / 2 + 0.08],
        [-w / 2 + 0.08, d / 2 - 0.08],
        [w / 2 - 0.08, d / 2 - 0.08],
      ].map(([lx, lz], i) => (
        <mesh key={i} position={[lx, legH / 2, lz]} castShadow>
          <cylinderGeometry args={[legR, legR * 0.7, legH, 12]} />
          <meshStandardMaterial color="#2b2b2b" metalness={0.6} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );
}

function Chair3D({ w, h, d, color }: { w: number; h: number; d: number; color: string }) {
  const seatH = h * 0.45;
  return (
    <group>
      {/* Сиденье */}
      <mesh position={[0, seatH, 0]} castShadow>
        <boxGeometry args={[w * 0.9, 0.04, d * 0.9]} />
        <meshStandardMaterial color={color} roughness={0.6} />
      </mesh>
      {/* Спинка */}
      <mesh position={[0, seatH + h * 0.3, -d * 0.38]} castShadow>
        <boxGeometry args={[w * 0.85, h * 0.55, 0.03]} />
        <meshStandardMaterial color={color} roughness={0.6} />
      </mesh>
      {/* Ножки */}
      {[
        [-w * 0.35, -d * 0.35],
        [w * 0.35, -d * 0.35],
        [-w * 0.35, d * 0.35],
        [w * 0.35, d * 0.35],
      ].map(([lx, lz], i) => (
        <mesh key={i} position={[lx, seatH / 2, lz]} castShadow>
          <cylinderGeometry args={[0.018, 0.012, seatH, 8]} />
          <meshStandardMaterial color="#1a1a1a" metalness={0.5} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );
}

function Bed3D({ w, h, d, color }: { w: number; h: number; d: number; color: string }) {
  const frameH = 0.25;
  const mattressH = 0.22;
  const headboardH = h;

  return (
    <group>
      {/* Каркас кровати */}
      <mesh position={[0, frameH / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[w, frameH, d]} />
        <meshStandardMaterial color="#4a3222" roughness={0.5} />
      </mesh>
      {/* Изголовье */}
      <mesh position={[0, headboardH / 2, -d / 2 + 0.05]} castShadow>
        <boxGeometry args={[w, headboardH, 0.1]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>
      {/* Матрас */}
      <mesh position={[0, frameH + mattressH / 2, 0.02]} castShadow receiveShadow>
        <boxGeometry args={[w - 0.08, mattressH, d - 0.15]} />
        <meshStandardMaterial color="#f4f3ef" roughness={0.9} />
      </mesh>
      {/* Подушки */}
      <mesh position={[-w * 0.25, frameH + mattressH + 0.06, -d * 0.3]} rotation={[0.2, 0, 0]} castShadow>
        <boxGeometry args={[w * 0.38, 0.1, d * 0.2]} />
        <meshStandardMaterial color="#ffffff" roughness={0.8} />
      </mesh>
      <mesh position={[w * 0.25, frameH + mattressH + 0.06, -d * 0.3]} rotation={[0.2, 0, 0]} castShadow>
        <boxGeometry args={[w * 0.38, 0.1, d * 0.2]} />
        <meshStandardMaterial color="#ffffff" roughness={0.8} />
      </mesh>
      {/* Покрывало */}
      <mesh position={[0, frameH + mattressH + 0.02, d * 0.12]} castShadow>
        <boxGeometry args={[w - 0.06, 0.04, d * 0.65]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>
    </group>
  );
}

function Plant3D({ w, h }: { w: number; h: number }) {
  const potH = h * 0.4;
  const potR = Math.max(0.12, w * 0.35);

  return (
    <group>
      {/* Керамический горшок */}
      <mesh position={[0, potH / 2, 0]} castShadow>
        <cylinderGeometry args={[potR, potR * 0.75, potH, 16]} />
        <meshStandardMaterial color="#d47a5b" roughness={0.4} />
      </mesh>
      {/* Земля */}
      <mesh position={[0, potH * 0.95, 0]}>
        <cylinderGeometry args={[potR * 0.95, potR * 0.95, 0.02, 16]} />
        <meshStandardMaterial color="#2d2218" roughness={0.9} />
      </mesh>
      {/* Листья / Стебли */}
      {[0, 60, 120, 180, 240, 300].map((deg, i) => {
        const rad = (deg * Math.PI) / 180;
        return (
          <mesh
            key={i}
            position={[Math.cos(rad) * 0.08, potH + 0.15 + (i % 2) * 0.08, Math.sin(rad) * 0.08]}
            rotation={[0.35, rad, 0.3]}
            castShadow
          >
            <sphereGeometry args={[w * 0.28, 8, 8]} />
            <meshStandardMaterial color="#386641" roughness={0.4} />
          </mesh>
        );
      })}
    </group>
  );
}

function ExternalGLTF({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene.clone()} />;
}

interface DraggableFurnitureProps {
  item: FurnitureItem;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onDragStart: (id: string) => void;
}

function DraggableFurniture({ item, isSelected, onSelect, onDragStart }: DraggableFurnitureProps) {
  const [x, y, z] = item.position;
  const [w, h, d] = item.dimensions;
  const color = colorFor(item.color ?? item.material, "#7a8288");

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    onSelect(item.id);
    onDragStart(item.id);
  };

  const renderModel = () => {
    if (item.model_ref) {
      return (
        <Suspense fallback={<Sofa3D w={w} h={h} d={d} color={color} />}>
          <ExternalGLTF url={item.model_ref} />
        </Suspense>
      );
    }

    const t = item.type.toLowerCase();
    if (t.includes("sofa") || t.includes("диван") || t.includes("couch")) {
      return <Sofa3D w={w} h={h} d={d} color={color} />;
    }
    if (t.includes("table") || t.includes("стол") || t.includes("desk")) {
      return <Table3D w={w} h={h} d={d} color={color} />;
    }
    if (t.includes("chair") || t.includes("стул") || t.includes("armchair") || t.includes("кресло")) {
      return <Chair3D w={w} h={h} d={d} color={color} />;
    }
    if (t.includes("bed") || t.includes("кровать")) {
      return <Bed3D w={w} h={h} d={d} color={color} />;
    }
    if (t.includes("plant") || t.includes("растение") || t.includes("цветок")) {
      return <Plant3D w={w} h={h} />;
    }

    // Универсальный стилизованный бокс для прочих предметов
    return (
      <mesh position={[0, h / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[w, h, d]} />
        <meshStandardMaterial
          color={color}
          roughness={0.5}
          metalness={0.1}
          emissive={isSelected ? "#3b82f6" : "#000000"}
          emissiveIntensity={isSelected ? 0.25 : 0}
        />
      </mesh>
    );
  };

  return (
    <group
      position={[x, y, z]}
      rotation={[0, (item.rotation_deg * Math.PI) / 180, 0]}
      onPointerDown={handlePointerDown}
    >
      {renderModel()}
      {/* Визуальная подсветка выбора */}
      {isSelected && (
        <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[Math.max(w, d) * 0.6, Math.max(w, d) * 0.65, 32]} />
          <meshBasicMaterial color="#3b82f6" side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

function DecorMesh({ item }: { item: DecorItem }) {
  const [x, y, z] = item.position;
  const size = 0.4 * (item.scale || 1);

  if (item.type === "plant") {
    return (
      <group position={[x, y, z]}>
        <Plant3D w={size} h={size * 1.5} />
      </group>
    );
  }

  if (item.type === "rug" || item.type === "carpet" || item.type === "ковер") {
    return (
      <mesh position={[x, 0.005, z]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[size * 3, size * 2]} />
        <meshStandardMaterial color="#c2b8a3" roughness={0.9} />
      </mesh>
    );
  }

  return (
    <mesh position={[x, y + size / 2, z]} castShadow>
      <boxGeometry args={[size, size, size]} />
      <meshStandardMaterial color="#c2b8a3" roughness={0.6} />
    </mesh>
  );
}

function SceneLight({ light }: { light: LightSource }) {
  const [x, y, z] = light.position;
  const color = light.color_temperature_k < 3500 ? "#ffe3b5" : "#f4f7fa";

  return (
    <group position={[x, y, z]}>
      {/* Видимый светильник на потолке */}
      <mesh position={[0, 0.02, 0]}>
        <cylinderGeometry args={[0.1, 0.1, 0.04, 16]} />
        <meshStandardMaterial color="#ffffff" emissive={color} emissiveIntensity={0.8} />
      </mesh>
      <pointLight color={color} intensity={light.intensity * 12} distance={10} decay={2} castShadow />
    </group>
  );
}

/** Невидимая плоскость пола сцены — принимает pointer-события для drag&drop мебели */
function DragPlane({
  draggingId,
  onDragMove,
  onDragEnd,
}: {
  draggingId: string | null;
  onDragMove: (id: string, point: THREE.Vector3) => void;
  onDragEnd: () => void;
}) {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -0.001, 0]}
      onPointerMove={(e) => {
        if (!draggingId) return;
        e.stopPropagation();
        onDragMove(draggingId, e.point);
      }}
      onPointerUp={(e) => {
        if (!draggingId) return;
        e.stopPropagation();
        onDragEnd();
      }}
      visible={false}
    >
      <planeGeometry args={[200, 200]} />
      <meshBasicMaterial />
    </mesh>
  );
}

interface WallObstacle {
  cx: number;
  cz: number;
  halfLength: number;
  halfWidth: number;
  angle: number;
}

function buildWallObstacles(rooms: Room[]): WallObstacle[] {
  const obstacles: WallObstacle[] = [];
  for (const room of rooms) {
    for (const wall of room.walls) {
      const [x1, z1] = wall.start;
      const [x2, z2] = wall.end;
      const length = Math.hypot(x2 - x1, z2 - z1);
      const angle = Math.atan2(z2 - z1, x2 - x1);

      if (wall.openings.length === 0) {
        obstacles.push({
          cx: (x1 + x2) / 2,
          cz: (z1 + z2) / 2,
          halfLength: length / 2,
          halfWidth: wall.thickness / 2,
          angle,
        });
        continue;
      }

      const sorted = [...wall.openings].sort((a, b) => a.position - b.position);
      const segments: { from: number; to: number }[] = [];
      let cursor = 0;
      for (const opening of sorted) {
        const center = opening.position * length;
        const half = opening.width / 2;
        const start = Math.max(0, center - half);
        const end = Math.min(length, center + half);
        if (start > cursor) segments.push({ from: cursor, to: start });
        cursor = Math.max(cursor, end);
      }
      if (cursor < length) segments.push({ from: cursor, to: length });

      for (const seg of segments) {
        const segLen = seg.to - seg.from;
        if (segLen <= 0.02) continue;
        const localCenter = (seg.from + seg.to) / 2;
        obstacles.push({
          cx: x1 + Math.cos(angle) * localCenter,
          cz: z1 + Math.sin(angle) * localCenter,
          halfLength: segLen / 2,
          halfWidth: wall.thickness / 2,
          angle,
        });
      }
    }
  }
  return obstacles;
}

function collidesAt(x: number, z: number, obstacles: WallObstacle[], radius: number): boolean {
  for (const o of obstacles) {
    const dx = x - o.cx;
    const dz = z - o.cz;
    const cos = Math.cos(-o.angle);
    const sin = Math.sin(-o.angle);
    const localX = dx * cos - dz * sin;
    const localZ = dx * sin + dz * cos;
    if (Math.abs(localX) <= o.halfLength + radius && Math.abs(localZ) <= o.halfWidth + radius) {
      return true;
    }
  }
  return false;
}

const PLAYER_RADIUS = 0.3;

function WalkMovement({ rooms }: { rooms: Room[] }) {
  const { camera } = useThree();
  const keys = useRef<Record<string, boolean>>({});
  const obstacles = useMemo(() => buildWallObstacles(rooms), [rooms]);

  useMemo(() => {
    const down = (e: KeyboardEvent) => (keys.current[e.code] = true);
    const up = (e: KeyboardEvent) => (keys.current[e.code] = false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    camera.position.set(camera.position.x, 1.6, camera.position.z);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [camera]);

  useFrame((_, delta) => {
    const speed = 2.8 * delta;
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).negate();

    let dx = 0;
    let dz = 0;
    if (keys.current["KeyW"]) {
      dx += forward.x * speed;
      dz += forward.z * speed;
    }
    if (keys.current["KeyS"]) {
      dx -= forward.x * speed;
      dz -= forward.z * speed;
    }
    if (keys.current["KeyA"]) {
      dx -= right.x * speed;
      dz -= right.z * speed;
    }
    if (keys.current["KeyD"]) {
      dx += right.x * speed;
      dz += right.z * speed;
    }

    const { x, z } = camera.position;
    if (dx !== 0 && !collidesAt(x + dx, z, obstacles, PLAYER_RADIUS)) {
      camera.position.x += dx;
    }
    if (dz !== 0 && !collidesAt(camera.position.x, z + dz, obstacles, PLAYER_RADIUS)) {
      camera.position.z += dz;
    }
    camera.position.y = 1.6;
  });

  return null;
}

type CameraMode = "orbit" | "walk";

export function SceneViewer({ scene }: { scene: Scene | null }) {
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [liveOverrides, setLiveOverrides] = useState<Record<string, [number, number, number]>>({});

  const updateFurniture = useAppStore((s) => s.updateFurniture);

  if (!scene || scene.rooms.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-neutral-400">
        Сцена ещё не готова
      </div>
    );
  }

  const selectedItem = scene.furniture.find((f) => f.id === selectedId) ?? null;

  const handleDragMove = (id: string, point: THREE.Vector3) => {
    setLiveOverrides((prev) => ({ ...prev, [id]: [point.x, 0, point.z] }));
  };

  const handleDragEnd = () => {
    if (draggingId && liveOverrides[draggingId]) {
      updateFurniture(draggingId, { position: liveOverrides[draggingId] });
    }
    setDraggingId(null);
  };

  const handleRotate = (id: string, deltaDeg: number) => {
    const item = scene.furniture.find((f) => f.id === id);
    if (!item) return;
    const newRot = (item.rotation_deg + deltaDeg + 360) % 360;
    updateFurniture(id, { rotation_deg: newRot });
  };

  return (
    <div className="relative h-full w-full bg-[#18181b]">
      <div className="absolute left-3 top-3 z-10 flex gap-2">
        <button
          onClick={() => setCameraMode(cameraMode === "orbit" ? "walk" : "orbit")}
          className="rounded-lg bg-neutral-900/90 backdrop-blur border border-neutral-700 px-3 py-1.5 text-xs font-medium text-white shadow-lg hover:bg-neutral-800 transition"
        >
          {cameraMode === "orbit" ? "🚶 Режим прогулки" : "🖱 Режим обзора"}
        </button>
        {cameraMode === "walk" && (
          <span className="rounded-lg bg-neutral-900/90 backdrop-blur border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 shadow">
            Клик — захват мыши, WASD — перемещение, Esc — выход
          </span>
        )}
      </div>

      {selectedItem && cameraMode === "orbit" && (
        <FurnitureInspector
          item={selectedItem}
          onChangeColor={(color) => updateFurniture(selectedItem.id, { color })}
          onRotate={(delta) => handleRotate(selectedItem.id, delta)}
          onClose={() => setSelectedId(null)}
        />
      )}

      <Canvas shadows camera={{ position: [7, 7, 7], fov: 45 }}>
        {/* Мягкий интерьерный свет */}
        <ambientLight intensity={0.45} />
        <directionalLight
          position={[10, 15, 8]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-bias={-0.0001}
        />
        <hemisphereLight intensity={0.25} groundColor="#2a2a2a" color="#ffffff" />

        {/* Светильники на сцене */}
        {scene.lighting.map((light) => (
          <SceneLight key={light.id} light={light} />
        ))}

        {/* Помещения (стены, пол, окна) */}
        {scene.rooms.map((room) => (
          <group key={room.id}>
            <RoomFloor room={room} />
            <RoomWalls room={room} />
          </group>
        ))}

        {/* Мебель */}
        {scene.furniture.map((item) => {
          const override = liveOverrides[item.id];
          const effectiveItem = override ? { ...item, position: override } : item;
          return (
            <DraggableFurniture
              key={item.id}
              item={effectiveItem}
              isSelected={item.id === selectedId}
              onSelect={setSelectedId}
              onDragStart={cameraMode === "orbit" ? setDraggingId : () => {}}
            />
          );
        })}

        {/* Декор */}
        {scene.decor.map((item) => (
          <DecorMesh key={item.id} item={item} />
        ))}

        {/* Мягкие контактные тени на полу */}
        <ContactShadows position={[0, 0.002, 0]} opacity={0.6} scale={25} blur={1.5} far={4} />

        {cameraMode === "orbit" && (
          <>
            <DragPlane draggingId={draggingId} onDragMove={handleDragMove} onDragEnd={handleDragEnd} />
            <OrbitControls makeDefault enabled={!draggingId} maxPolarAngle={Math.PI / 2 - 0.05} />
          </>
        )}

        {cameraMode === "walk" && (
          <>
            <PointerLockControls makeDefault />
            <WalkMovement rooms={scene.rooms} />
          </>
        )}
      </Canvas>
    </div>
  );
}

function FurnitureInspector({
  item,
  onChangeColor,
  onRotate,
  onClose,
}: {
  item: FurnitureItem;
  onChangeColor: (color: string) => void;
  onRotate: (delta: number) => void;
  onClose: () => void;
}) {
  const [w, h, d] = item.dimensions;

  return (
    <div className="absolute right-3 top-3 z-10 w-64 rounded-xl border border-neutral-700 bg-neutral-900/95 p-3.5 text-xs text-white shadow-2xl backdrop-blur">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-semibold text-sm capitalize">{item.type}</span>
        <button onClick={onClose} className="rounded p-1 text-neutral-400 hover:bg-neutral-800 hover:text-white transition">
          ✕
        </button>
      </div>

      <div className="mb-2.5 flex items-center justify-between text-neutral-400 bg-neutral-800/60 rounded px-2 py-1">
        <span>Габариты:</span>
        <span className="font-mono text-neutral-200">
          {w.toFixed(1)} × {d.toFixed(1)} × {h.toFixed(1)} м
        </span>
      </div>

      {/* Поворот предмета */}
      <div className="mb-3">
        <p className="mb-1.5 font-medium text-neutral-300">Поворот ({Math.round(item.rotation_deg)}°):</p>
        <div className="grid grid-cols-3 gap-1.5">
          <button
            onClick={() => onRotate(-45)}
            className="rounded bg-neutral-800 py-1 font-medium hover:bg-neutral-700 transition"
          >
            ↺ -45°
          </button>
          <button
            onClick={() => onRotate(45)}
            className="rounded bg-neutral-800 py-1 font-medium hover:bg-neutral-700 transition"
          >
            ↻ +45°
          </button>
          <button
            onClick={() => onRotate(90)}
            className="rounded bg-neutral-800 py-1 font-medium hover:bg-neutral-700 transition"
          >
            ⟲ 90°
          </button>
        </div>
      </div>

      {/* Материалы и цвета */}
      <div>
        <p className="mb-1.5 font-medium text-neutral-300">Материал и цвет:</p>
        <div className="grid grid-cols-4 gap-1.5">
          {MATERIAL_PRESETS.map((preset) => (
            <button
              key={preset.value}
              onClick={() => onChangeColor(preset.value)}
              title={preset.label}
              className="flex flex-col items-center gap-1 rounded p-1 hover:bg-neutral-800 transition"
            >
              <span
                className="h-5 w-5 rounded-full border border-neutral-600 shadow-sm"
                style={{ backgroundColor: preset.hex }}
              />
              <span className="text-[10px] text-neutral-400 truncate w-full text-center">{preset.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
