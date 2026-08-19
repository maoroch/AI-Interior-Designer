import { create } from "zustand";
import { api } from "./api";
import type { ChatMessage, PipelineStage, Project, Scene } from "./types";

interface AppState {
  projectId: string | null;
  project: Project | null;
  scene: Scene | null;
  variants: Scene[];
  chatHistory: ChatMessage[];
  isLoading: boolean;
  errorMessage: string | null;
  ws: WebSocket | null;

  startProject: (file: File) => Promise<void>;
  submitPreferences: (prefs: Project["preferences"]) => Promise<void>;
  chooseArchitectOption: (choiceId: string | null) => Promise<void>;
  refreshProject: () => Promise<void>;
  refreshScene: () => Promise<void>;
  refreshVariants: () => Promise<void>;
  selectVariant: (variantId: string) => Promise<void>;
  sendChatMessage: (message: string) => Promise<void>;
  updateFurniture: (
    furnitureId: string,
    patch: { position?: [number, number, number]; rotation_deg?: number; material?: string; color?: string }
  ) => Promise<void>;
  connectWebSocket: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  projectId: null,
  project: null,
  scene: null,
  variants: [],
  chatHistory: [],
  isLoading: false,
  errorMessage: null,
  ws: null,

  startProject: async (file: File) => {
    set({ isLoading: true, errorMessage: null });
    try {
      const { project_id } = await api.uploadPlan(file);
      set({ projectId: project_id, isLoading: false });
      get().connectWebSocket();
      await get().refreshProject();
    } catch (err) {
      set({ isLoading: false, errorMessage: (err as Error).message });
    }
  },

  submitPreferences: async (prefs) => {
    const { projectId } = get();
    if (!projectId || !prefs) return;
    await api.setPreferences(projectId, prefs);
    await get().refreshProject();
  },

  chooseArchitectOption: async (choiceId) => {
    const { projectId } = get();
    if (!projectId) return;
    set({ isLoading: true });
    await api.chooseArchitectOption(projectId, choiceId);
    set({ isLoading: false });
  },

  refreshProject: async () => {
    const { projectId } = get();
    if (!projectId) return;
    try {
      const project = await api.getProject(projectId);
      set({ project });
      if (project.stage === "ready") {
        await get().refreshScene();
        await get().refreshVariants();
      }
    } catch (err) {
      set({ errorMessage: (err as Error).message });
    }
  },

  refreshScene: async () => {
    const { projectId } = get();
    if (!projectId) return;
    try {
      const scene = await api.getScene(projectId);
      set({ scene });
    } catch {
      // Сцена ещё не готова — не считаем это ошибкой пользователя
    }
  },

  refreshVariants: async () => {
    const { projectId } = get();
    if (!projectId) return;
    try {
      const variants = await api.getAllVariants(projectId);
      set({ variants });
    } catch {
      // варианты ещё не сгенерированы
    }
  },

  selectVariant: async (variantId: string) => {
    const { projectId } = get();
    if (!projectId) return;
    await api.selectVariant(projectId, variantId);
    await get().refreshProject();
  },

  sendChatMessage: async (message: string) => {
    const { projectId, chatHistory } = get();
    if (!projectId) return;
    const optimisticUser: ChatMessage = {
      id: `local_${Date.now()}`,
      project_id: projectId,
      role: "user",
      content: message,
      created_at: new Date().toISOString(),
      applied_patch: null,
    };
    set({ chatHistory: [...chatHistory, optimisticUser] });
    const assistantMsg = await api.sendChatMessage(projectId, message);
    set({ chatHistory: [...get().chatHistory, assistantMsg] });
    await get().refreshScene();
  },

  updateFurniture: async (furnitureId, patch) => {
    const { projectId } = get();
    if (!projectId) return;
    const updatedScene = await api.patchFurniture(projectId, furnitureId, patch);
    set({ scene: updatedScene });
  },

  connectWebSocket: () => {
    const { projectId, ws } = get();
    if (!projectId || ws) return;

    let retryTimeout: NodeJS.Timeout | null = null;
    let attempts = 0;

    const createSocket = () => {
      const currentProjectId = get().projectId;
      if (!currentProjectId) return;

      const socket = new WebSocket(api.wsUrl(currentProjectId));
      socket.onopen = () => {
        attempts = 0;
        set({ ws: socket });
      };
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as { stage?: PipelineStage };
          if (payload.stage) {
            get().refreshProject();
          }
        } catch {
          // игнорируем нераспознанные сообщения
        }
      };
      socket.onclose = () => {
        set({ ws: null });
        if (get().projectId === currentProjectId) {
          // Экспоненциальный reconnect от 1с до максимум 10с
          const delay = Math.min(1000 * Math.pow(1.5, attempts), 10000);
          attempts++;
          retryTimeout = setTimeout(createSocket, delay);
        }
      };
    };

    createSocket();
  },
}));
