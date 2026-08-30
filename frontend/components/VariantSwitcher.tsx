"use client";

import { useAppStore } from "@/lib/store";

export function VariantSwitcher() {
  const variants = useAppStore((s) => s.variants);
  const project = useAppStore((s) => s.project);
  const selectVariant = useAppStore((s) => s.selectVariant);

  if (variants.length <= 1) return null;

  return (
    <div className="flex items-center gap-2 border-b border-neutral-800 bg-[#161822] px-5 py-2.5 z-10">
      <span className="text-[11px] font-medium text-neutral-400 mr-1 hidden sm:inline">
        Варианты интерьера:
      </span>
      <div className="flex items-center gap-2 overflow-x-auto">
        {variants.map((v) => {
          const isActive = project?.active_variant_id === v.variant_id;
          return (
            <button
              key={v.variant_id}
              onClick={() => selectVariant(v.variant_id)}
              className={`flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-medium transition-all ${
                isActive
                  ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-indigo-500/25 ring-1 ring-white/20"
                  : "bg-neutral-800/80 text-neutral-300 hover:bg-neutral-700 hover:text-white border border-neutral-700/50"
              }`}
            >
              <span className={`h-2 w-2 rounded-full ${isActive ? "bg-white" : "bg-neutral-500"}`} />
              <span>{v.variant_label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
