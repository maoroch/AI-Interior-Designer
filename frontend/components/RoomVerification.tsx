"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { RoomType } from "@/lib/types";

const ROOM_TYPE_OPTIONS: Array<{ value: RoomType; label: string; icon: string }> = [
  { value: "living_room", label: "Гостиная / Студия", icon: "🛋" },
  { value: "bedroom", label: "Спальня", icon: "🛏" },
  { value: "kitchen", label: "Кухня", icon: "🍳" },
  { value: "dining_room", label: "Кухня-столовая", icon: "🍽" },
  { value: "kids_room", label: "Детская комната", icon: "🧸" },
  { value: "office", label: "Кабинет / Рабочая зона", icon: "💼" },
  { value: "hallway", label: "Прихожая / Коридор", icon: "🚪" },
  { value: "bathroom", label: "Санузел / Ванная", icon: "🚿" },
];

function calculatePolygonArea(polygon: [number, number][]): number {
  if (!polygon || polygon.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < polygon.length; i++) {
    const j = (i + 1) % polygon.length;
    area += polygon[i][0] * polygon[j][1];
    area -= polygon[j][0] * polygon[i][1];
  }
  return Math.round((Math.abs(area) / 2.0) * 10) / 10;
}

interface RoomVerificationProps {
  onConfirm: () => void;
}

export function RoomVerification({ onConfirm }: RoomVerificationProps) {
  const project = useAppStore((s) => s.project);
  const scene = useAppStore((s) => s.scene);
  const refreshProject = useAppStore((s) => s.refreshProject);

  const initialRooms = scene?.rooms ?? [];
  const [roomsState, setRoomsState] = useState(
    initialRooms.map((r, i) => ({
      id: r.id,
      index: i + 1,
      type: r.type,
      height: r.height || 2.7,
      label: r.label || `Помещение ${i + 1}`,
      area: calculatePolygonArea(r.polygon),
      enabled: true,
    }))
  );
  const [isSubmitting, setIsSubmitting] = useState(false);

  const totalArea = roomsState
    .filter((r) => r.enabled)
    .reduce((sum, r) => sum + r.area, 0);

  const handleTypeChange = (id: string, newType: RoomType) => {
    setRoomsState((prev) =>
      prev.map((r) => (r.id === id ? { ...r, type: newType } : r))
    );
  };

  const handleHeightChange = (id: string, newHeight: number) => {
    setRoomsState((prev) =>
      prev.map((r) => (r.id === id ? { ...r, height: newHeight } : r))
    );
  };

  const handleToggleRoom = (id: string) => {
    setRoomsState((prev) =>
      prev.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r))
    );
  };

  const handleSaveAndProceed = async () => {
    if (!project) return;
    setIsSubmitting(true);
    try {
      await api.updateRooms(
        project.id,
        roomsState.map((r) => ({
          id: r.id,
          type: r.type,
          height: r.height,
          label: r.label,
          enabled: r.enabled,
        }))
      );
      await refreshProject();
      onConfirm();
    } catch (e) {
      console.error("Failed to update rooms:", e);
      onConfirm();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-5 text-neutral-100">
      <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
        <div>
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <span>📐</span> Экспликация помещений
          </h2>
          <p className="text-xs text-neutral-400 mt-0.5">
            Проверьте распознанные комнаты, назначение и высоту потолков
          </p>
        </div>
        <div className="text-right">
          <span className="text-[11px] text-neutral-400">Общая площадь:</span>
          <p className="text-sm font-bold text-emerald-400">
            {Math.round(totalArea * 10) / 10} м²
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-3 max-h-[50vh] overflow-y-auto pr-1">
        {roomsState.map((room) => (
          <div
            key={room.id}
            className={`flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-xl border p-3.5 transition-all ${
              room.enabled
                ? "border-neutral-700/80 bg-neutral-800/60 shadow-sm"
                : "border-neutral-800/40 bg-neutral-900/30 opacity-50"
            }`}
          >
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={room.enabled}
                onChange={() => handleToggleRoom(room.id)}
                className="h-4 w-4 rounded border-neutral-700 bg-neutral-900 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
              <div>
                <span className="text-xs font-semibold text-white">
                  Комната #{room.index}
                </span>
                <span className="ml-2 rounded bg-neutral-700/70 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                  {room.area > 0 ? `${room.area} м²` : "—"}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto">
              {/* Выбор назначения комнаты */}
              <select
                value={room.type}
                disabled={!room.enabled}
                onChange={(e) => handleTypeChange(room.id, e.target.value as RoomType)}
                className="flex-1 sm:flex-none rounded-lg border border-neutral-700 bg-neutral-900/90 px-3 py-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none cursor-pointer"
              >
                {ROOM_TYPE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.icon} {opt.label}
                  </option>
                ))}
              </select>

              {/* Высота потолка */}
              <div className="flex items-center gap-1 bg-neutral-900/90 border border-neutral-700 rounded-lg px-2 py-1 text-xs">
                <span className="text-[11px] text-neutral-400">h:</span>
                <input
                  type="number"
                  step="0.1"
                  min="2.2"
                  max="4.5"
                  value={room.height}
                  disabled={!room.enabled}
                  onChange={(e) => handleHeightChange(room.id, parseFloat(e.target.value) || 2.7)}
                  className="w-12 bg-transparent text-center font-medium text-white focus:outline-none"
                />
                <span className="text-[11px] text-neutral-400">м</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={handleSaveAndProceed}
        disabled={isSubmitting || totalArea === 0}
        className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 py-3 text-xs font-semibold text-white shadow-lg shadow-indigo-500/25 hover:from-blue-500 hover:to-indigo-500 transition disabled:opacity-50"
      >
        {isSubmitting ? "Сохранение параметров…" : "Подтвердить экспликацию и перейти к стилю →"}
      </button>
    </div>
  );
}
