"use client";

import { useState } from "react";
import { useAppStore } from "@/lib/store";

export function ChatPanel() {
  const [input, setInput] = useState("");
  const chatHistory = useAppStore((s) => s.chatHistory);
  const sendChatMessage = useAppStore((s) => s.sendChatMessage);
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || sending) return;
    setSending(true);
    const message = input;
    setInput("");
    try {
      await sendChatMessage(message);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {chatHistory.length === 0 && (
          <p className="text-xs text-neutral-400">
            Попробуйте: &laquo;Замени диван&raquo;, &laquo;Сделай интерьер светлее&raquo;,
            &laquo;Добавь растения&raquo;.
          </p>
        )}
        {chatHistory.map((msg) => (
          <div
            key={msg.id}
            className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
              msg.role === "user"
                ? "ml-auto bg-neutral-900 text-white"
                : "bg-neutral-100 text-neutral-800"
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>
      <div className="flex gap-2 border-t border-neutral-200 p-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Напишите, что изменить…"
          className="flex-1 rounded border border-neutral-200 px-3 py-2 text-sm"
        />
        <button
          onClick={handleSend}
          disabled={sending}
          className="rounded bg-neutral-900 px-3 py-2 text-sm text-white disabled:opacity-50"
        >
          Отправить
        </button>
      </div>
    </div>
  );
}
