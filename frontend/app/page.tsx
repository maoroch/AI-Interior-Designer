"use client";

import { useEffect, useState } from "react";
import { ArchitectChoice } from "@/components/ArchitectChoice";
import { ChatPanel } from "@/components/ChatPanel";
import { PreferencesForm } from "@/components/PreferencesForm";
import { ProjectDashboardBar } from "@/components/ProjectDashboardBar";
import { RoomVerification } from "@/components/RoomVerification";
import { SceneViewer } from "@/components/SceneViewer";
import { UploadPlan } from "@/components/UploadPlan";
import { VariantSwitcher } from "@/components/VariantSwitcher";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store";

const STAGE_LABELS: Record<string, string> = {
  uploaded: "План загружен",
  analyzing_floorplan: "Распознаём стены, двери, окна (CV + Vision)…",
  detecting_rooms: "Определяем функциональные зоны комнат…",
  awaiting_architect_decision: "Готовы варианты перепланировки",
  designing_interior: "Подбираем стили, текстуры и материалы…",
  planning_furniture: "Расставляем 3D-мебель и проходы…",
  designing_lighting: "Продумываем световой сценарий…",
  adding_decor: "Добавляем декор и озеленение…",
  generating_scene: "Финальная сборка интерактивной 3D-сцены…",
  ready: "3D-проект готов к просмотру и редактированию",
  failed: "Ошибка генерации",
};

export default function Home() {
  const project = useAppStore((s) => s.project);
  const scene = useAppStore((s) => s.scene);
  const refreshProject = useAppStore((s) => s.refreshProject);
  const [roomsConfirmed, setRoomsConfirmed] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const pid = params.get("project_id");
      if (pid) {
        useAppStore.setState({ projectId: pid });
        refreshProject().then(() => {
          useAppStore.getState().refreshScene();
          useAppStore.getState().refreshVariants();
        });
      }
    }
  }, []);

  useEffect(() => {
    if (!project || project.stage === "ready" || project.stage === "failed") return;
    const interval = setInterval(refreshProject, 3000);
    return () => clearInterval(interval);
  }, [project, refreshProject]);

  const stage = project?.stage;

  return (
    <main className="flex h-screen w-screen flex-col bg-[#0f1117] text-neutral-100 antialiased select-none overflow-hidden">
      {/* Шапка с брендингом и CTA */}
      <header className="flex items-center justify-between border-b border-neutral-800 bg-[#161822]/80 backdrop-blur px-6 py-3 z-20">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-white font-bold shadow-md shadow-indigo-500/20">
            ✦
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-tight text-white">AI Interior Designer</h1>
              <span className="rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-medium text-indigo-400 border border-indigo-500/20">
                Showcase
              </span>
            </div>
            {stage && (
              <p className="text-[11px] text-neutral-400 flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {STAGE_LABELS[stage] ?? stage}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {project && stage === "ready" && (
            <a
              href={api.exportPdfUrl(project.id, project.active_variant_id)}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 rounded-lg border border-neutral-700 bg-neutral-800/80 px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-neutral-700 hover:text-white transition shadow-sm"
            >
              <span>📄</span>
              <span>Экспорт PDF</span>
            </a>
          )}

          {/* Авторский бейдж */}
          <div className="hidden sm:flex items-center gap-2 border-l border-neutral-800 pl-4">
            <span className="text-[11px] text-neutral-400">Created by</span>
            <span className="text-xs font-medium text-neutral-200">Ilyas Salimov</span>
            <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400 border border-emerald-500/20">
              Available for Freelance
            </span>
          </div>
        </div>
      </header>

      {/* Экран загрузки плана */}
      {!project && (
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="w-full max-w-xl">
            <UploadPlan />
          </div>
        </div>
      )}

      {/* Шаг 1: Экспликация помещений */}
      {project && !roomsConfirmed && !project.preferences && stage !== "failed" && (
        <div className="flex-1 flex items-center justify-center p-4 overflow-y-auto">
          <div className="w-full max-w-xl rounded-2xl border border-neutral-800 bg-[#161822] p-6 shadow-2xl">
            <RoomVerification onConfirm={() => setRoomsConfirmed(true)} />
          </div>
        </div>
      )}

      {/* Шаг 2: Анкета предпочтений */}
      {project && roomsConfirmed && !project.preferences && stage !== "failed" && (
        <div className="flex-1 flex items-center justify-center p-4 overflow-y-auto">
          <div className="w-full max-w-lg rounded-2xl border border-neutral-800 bg-[#161822] p-6 shadow-2xl">
            <h2 className="mb-1 text-lg font-semibold text-white">Параметры дизайн-проекта</h2>
            <p className="mb-5 text-xs text-neutral-400">
              Настройте стиль, атмосферу и особенности вашей будущей планировки
            </p>
            <PreferencesForm />
          </div>
        </div>
      )}

      {/* Выбор архитектурной перепланировки */}
      {project && project.preferences && stage === "awaiting_architect_decision" && (
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl border border-neutral-800 bg-[#161822] p-6 shadow-2xl">
            <ArchitectChoice />
          </div>
        </div>
      )}

      {/* Основной интерактивный 3D-экран + Чат */}
      {project &&
        project.preferences &&
        stage &&
        !["uploaded", "analyzing_floorplan", "detecting_rooms", "awaiting_architect_decision"].includes(stage) &&
        stage !== "failed" && (
          <div className="flex flex-1 flex-col overflow-hidden">
            <VariantSwitcher />
            <ProjectDashboardBar scene={scene} />
            <div className="flex flex-1 overflow-hidden">
              <div className="flex-1 relative">
                <SceneViewer scene={scene} />
              </div>
              <aside className="w-80 border-l border-neutral-800 bg-[#161822]">
                <ChatPanel />
              </aside>
            </div>
          </div>
        )}

      {/* Сообщение об ошибке */}
      {stage === "failed" && (
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="max-w-md rounded-xl border border-red-500/30 bg-red-950/30 p-5 text-center text-sm text-red-300">
            <p className="font-semibold mb-1">Ошибка обработки проекта</p>
            <p className="text-xs text-red-400">{project?.error}</p>
          </div>
        </div>
      )}
    </main>
  );
}
