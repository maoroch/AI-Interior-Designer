import type { ChatMessage, Project, Scene, UserPreferences } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} -> ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  uploadPlan: async (file: File): Promise<{ project_id: string; stage: string }> => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_URL}/projects/upload`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  setPreferences: (projectId: string, preferences: UserPreferences) =>
    request<{ ok: boolean }>(`/projects/${projectId}/preferences`, {
      method: "POST",
      body: JSON.stringify(preferences),
    }),

  chooseArchitectOption: (projectId: string, choiceId: string | null) =>
    request<{ ok: boolean }>(
      `/projects/${projectId}/architect-choice${choiceId ? `?choice_id=${choiceId}` : ""}`,
      { method: "POST" }
    ),

  getProject: (projectId: string) => request<Project>(`/projects/${projectId}`),

  getScene: (projectId: string) => request<Scene>(`/projects/${projectId}/scene`),

  sendChatMessage: (projectId: string, message: string) =>
    request<ChatMessage>(`/projects/${projectId}/chat`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  getChatHistory: (projectId: string) =>
    request<ChatMessage[]>(`/projects/${projectId}/chat`),

  patchFurniture: (
    projectId: string,
    furnitureId: string,
    patch: { position?: [number, number, number]; rotation_deg?: number; material?: string; color?: string }
  ) =>
    request<Scene>(`/projects/${projectId}/scene/furniture/${furnitureId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  getAllVariants: (projectId: string) => request<Scene[]>(`/projects/${projectId}/scenes`),

  selectVariant: (projectId: string, variantId: string) =>
    request<{ ok: boolean; active_variant_id: string }>(
      `/projects/${projectId}/select-variant?variant_id=${variantId}`,
      { method: "POST" }
    ),

  wsUrl: (projectId: string) =>
    `${API_URL.replace(/^http/, "ws")}/ws/projects/${projectId}`,

  updateRooms: (
    projectId: string,
    rooms: Array<{ id: string; type: string; height?: number; label?: string; enabled?: boolean }>
  ) =>
    request<{ ok: boolean; updated_count: number }>(`/projects/${projectId}/rooms`, {
      method: "PATCH",
      body: JSON.stringify({ rooms }),
    }),

  exportPdfUrl: (projectId: string, variantId?: string) =>
    `${API_URL}/projects/${projectId}/export/pdf${variantId ? `?variant_id=${variantId}` : ""}`,
};
