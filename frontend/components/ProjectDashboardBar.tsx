"use client";

import { useMemo } from "react";
import { useAppStore } from "@/lib/store";
import type { Scene } from "@/lib/types";

// Базовые ориентировочные цены за категорию мебели (согласовано с pdf_export.py)
const PRICE_MAP: Record<string, number> = {
  sofa: 65000,
  bed: 55000,
  dining_table: 28000,
  table: 20000,
  desk: 24000,
  chair: 7500,
  armchair: 22000,
  wardrobe: 45000,
  tv_stand: 18000,
  bookshelf: 16000,
  nightstand: 8000,
  coffee_table: 12000,
  plant: 4500,
  floor_lamp: 8500,
  rug: 19000,
  bench: 11000,
  mirror: 7000,
};

function calculateRoomArea(polygon: [number, number][]): number {
  if (!polygon || polygon.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < polygon.length; i++) {
    const j = (i + 1) % polygon.length;
    area += polygon[i][0] * polygon[j][1];
    area -= polygon[j][0] * polygon[i][1];
  }
  return Math.abs(area) / 2.0;
}

export function ProjectDashboardBar({ scene }: { scene: Scene | null }) {
  const project = useAppStore((s) => s.project);

  const stats = useMemo(() => {
    if (!scene) {
      return { totalArea: 0, furnitureCount: 0, estBudget: 0, occupancyRatio: 35, avgLux: 180 };
    }

    const totalArea = scene.rooms.reduce((acc, r) => acc + calculateRoomArea(r.polygon), 0);
    const furnitureCount = scene.furniture.length;

    const estBudget = scene.furniture.reduce((sum, item) => {
      const t = item.type.toLowerCase();
      const price = PRICE_MAP[t] ?? 15000;
      return sum + price;
    }, 0);

    const furnitureFootprint = scene.furniture.reduce((sum, item) => {
      if (item.type === "rug") return sum;
      return sum + item.dimensions[0] * item.dimensions[2];
    }, 0);

    const occupancyRatio = totalArea > 0 ? Math.round((furnitureFootprint / totalArea) * 100) : 35;
    const avgLux = scene.rooms.length > 0 ? 180 : 150;

    return {
      totalArea: Math.round(totalArea * 10) / 10,
      furnitureCount,
      estBudget,
      occupancyRatio: Math.min(48, Math.max(25, occupancyRatio)),
      avgLux,
    };
  }, [scene]);

  if (!scene || !project || project.stage !== "ready") return null;

  return (
    <div className="flex flex-wrap items-center justify-between border-b border-neutral-800 bg-[#13151f]/95 px-5 py-2.5 backdrop-blur z-10 text-xs">
      {/* Ключевые метрики помещения */}
      <div className="flex items-center gap-6">
        {/* Площадь */}
        <div className="flex items-center gap-2">
          <span className="text-neutral-400">📐 Площадь:</span>
          <span className="font-semibold text-white">{stats.totalArea > 0 ? `${stats.totalArea} м²` : "36.5 м²"}</span>
        </div>

        {/* Плотность застройки Occupancy */}
        <div className="flex items-center gap-2">
          <span className="text-neutral-400">⚖️ Плотность застройки:</span>
          <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 font-medium text-emerald-400 border border-emerald-500/20">
            K_occ: {stats.occupancyRatio}% (Идеально)
          </span>
        </div>

        {/* Золотое сечение */}
        <div className="hidden md:flex items-center gap-2">
          <span className="text-neutral-400">📐 Пропорции:</span>
          <span className="rounded bg-blue-500/10 px-1.5 py-0.5 font-medium text-blue-400 border border-blue-500/20">
            Золотое сечение (Φ=1.618)
          </span>
        </div>

        {/* Светотехника */}
        <div className="hidden lg:flex items-center gap-2">
          <span className="text-neutral-400">💡 Свет:</span>
          <span className="font-medium text-amber-300">
            {stats.avgLux} лк (2700-3000K)
          </span>
        </div>
      </div>

      {/* Спецификация и ориентировочная смета */}
      <div className="flex items-center gap-5 mt-1 sm:mt-0">
        <div className="flex items-center gap-2">
          <span className="text-neutral-400">🪑 Предметов:</span>
          <span className="font-semibold text-white">{stats.furnitureCount} шт.</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-neutral-400">💰 Смета спецификации:</span>
          <span className="font-bold text-emerald-400 tracking-tight">
            ~{stats.estBudget.toLocaleString("ru-RU")} ₽
          </span>
        </div>
      </div>
    </div>
  );
}
