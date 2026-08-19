"use client";

import { useState } from "react";
import { useAppStore } from "@/lib/store";
import type { UserPreferences } from "@/lib/types";

const DEFAULT_PREFS: UserPreferences = {
  style: "Modern Minimalism",
  budget_usd: 15000,
  adults: 2,
  children: 0,
  pets: [],
  favorite_colors: [],
  needs_office: false,
  likes_hosting_guests: false,
};

export function PreferencesForm() {
  const [prefs, setPrefs] = useState<UserPreferences>(DEFAULT_PREFS);
  const [colorsInput, setColorsInput] = useState("серый, белый, дерево");
  const [petsInput, setPetsInput] = useState("");
  const submitPreferences = useAppStore((s) => s.submitPreferences);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitPreferences({
      ...prefs,
      favorite_colors: colorsInput.split(",").map((c) => c.trim()).filter(Boolean),
      pets: petsInput.split(",").map((p) => p.trim()).filter(Boolean),
    });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 text-sm">
      <label className="flex flex-col gap-1">
        Стиль
        <input
          className="rounded border border-neutral-200 px-3 py-2"
          value={prefs.style ?? ""}
          onChange={(e) => setPrefs({ ...prefs, style: e.target.value })}
        />
      </label>

      <label className="flex flex-col gap-1">
        Бюджет, $
        <input
          type="number"
          className="rounded border border-neutral-200 px-3 py-2"
          value={prefs.budget_usd ?? 0}
          onChange={(e) => setPrefs({ ...prefs, budget_usd: Number(e.target.value) })}
        />
      </label>

      <div className="flex gap-4">
        <label className="flex flex-1 flex-col gap-1">
          Взрослые
          <input
            type="number"
            min={0}
            className="rounded border border-neutral-200 px-3 py-2"
            value={prefs.adults}
            onChange={(e) => setPrefs({ ...prefs, adults: Number(e.target.value) })}
          />
        </label>
        <label className="flex flex-1 flex-col gap-1">
          Дети
          <input
            type="number"
            min={0}
            className="rounded border border-neutral-200 px-3 py-2"
            value={prefs.children}
            onChange={(e) => setPrefs({ ...prefs, children: Number(e.target.value) })}
          />
        </label>
      </div>

      <label className="flex flex-col gap-1">
        Домашние животные (через запятую)
        <input
          className="rounded border border-neutral-200 px-3 py-2"
          value={petsInput}
          onChange={(e) => setPetsInput(e.target.value)}
          placeholder="кот, собака"
        />
      </label>

      <label className="flex flex-col gap-1">
        Любимые цвета (через запятую)
        <input
          className="rounded border border-neutral-200 px-3 py-2"
          value={colorsInput}
          onChange={(e) => setColorsInput(e.target.value)}
        />
      </label>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={prefs.needs_office}
          onChange={(e) => setPrefs({ ...prefs, needs_office: e.target.checked })}
        />
        Нужен кабинет
      </label>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={prefs.likes_hosting_guests}
          onChange={(e) => setPrefs({ ...prefs, likes_hosting_guests: e.target.checked })}
        />
        Люблю принимать гостей
      </label>

      <button
        type="submit"
        className="mt-2 rounded bg-neutral-900 px-4 py-2 text-white hover:bg-neutral-700"
      >
        Продолжить
      </button>
    </form>
  );
}
