"use client";

import { useAppStore } from "@/lib/store";

export function ArchitectChoice() {
  const scene = useAppStore((s) => s.scene);
  const chooseArchitectOption = useAppStore((s) => s.chooseArchitectOption);
  const suggestions = scene?.architect_suggestions ?? [];

  return (
    <div className="flex flex-col gap-3 text-sm">
      <p className="text-neutral-600">
        Architect Agent проанализировал планировку и предлагает варианты. Можно
        принять один из них — тогда геометрия комнат пересчитается, — или
        оставить планировку как есть.
      </p>

      {suggestions.map((s) => (
        <button
          key={s.id}
          onClick={() => chooseArchitectOption(s.id)}
          className="rounded-lg border border-neutral-200 p-3 text-left hover:border-neutral-400"
        >
          <p className="font-medium">{s.title}</p>
          <p className="mt-1 text-xs text-neutral-500">{s.description}</p>
        </button>
      ))}

      <button
        onClick={() => chooseArchitectOption(null)}
        className="rounded-lg border border-dashed border-neutral-300 p-3 text-left text-neutral-500 hover:border-neutral-400"
      >
        Оставить планировку как есть
      </button>
    </div>
  );
}
