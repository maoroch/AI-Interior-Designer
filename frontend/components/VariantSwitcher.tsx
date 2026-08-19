"use client";

import { useAppStore } from "@/lib/store";

export function VariantSwitcher() {
  const variants = useAppStore((s) => s.variants);
  const project = useAppStore((s) => s.project);
  const selectVariant = useAppStore((s) => s.selectVariant);

  if (variants.length <= 1) return null;

  return (
    <div className="flex gap-2 border-b border-neutral-200 bg-white px-4 py-2">
      {variants.map((v) => (
        <button
          key={v.variant_id}
          onClick={() => selectVariant(v.variant_id)}
          className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
            project?.active_variant_id === v.variant_id
              ? "bg-neutral-900 text-white"
              : "bg-neutral-100 text-neutral-600 hover:bg-neutral-200"
          }`}
        >
          {v.variant_label}
        </button>
      ))}
    </div>
  );
}
