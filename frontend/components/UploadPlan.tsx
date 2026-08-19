"use client";

import { useRef, useState } from "react";
import { useAppStore } from "@/lib/store";

export function UploadPlan() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const startProject = useAppStore((s) => s.startProject);
  const isLoading = useAppStore((s) => s.isLoading);

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      alert("Поддерживаются только PNG и JPG");
      return;
    }
    startProject(file);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFile(e.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
      className={`flex h-64 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
        dragOver ? "border-neutral-400 bg-neutral-50" : "border-neutral-200"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {isLoading ? (
        <p className="text-sm text-neutral-500">Загружаем и анализируем план…</p>
      ) : (
        <>
          <p className="text-sm font-medium text-neutral-700">
            Перетащите план квартиры сюда или нажмите, чтобы выбрать файл
          </p>
          <p className="mt-1 text-xs text-neutral-400">PNG или JPG</p>
        </>
      )}
    </div>
  );
}
