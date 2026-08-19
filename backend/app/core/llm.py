"""
Тонкая обёртка над Groq API, общая для всех LLM-агентов.

Ретраи: LLM-вызовы нестабильны (rate limits, временные сбои, иногда — невалидный
JSON в ответе). Раньше ошибки тихо проглатывались агентами (возвращали {}),
что маскировало реальные проблемы. Теперь: до RETRY_ATTEMPTS попыток с
экспоненциальной задержкой, а если все попытки провалились — поднимается
LLMCallError с понятным сообщением, которое долетает до pipeline_tasks.py
и переводит проект в статус failed с текстом ошибки, видимым пользователю.
"""
from __future__ import annotations

import asyncio
import json
import logging

from groq import AsyncGroq

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("llm")

_client: AsyncGroq | None = None

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.5


class LLMCallError(Exception):
    """Поднимается, когда все попытки вызова LLM исчерпаны."""


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def _with_retries(coro_factory, description: str):
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - любые сбои Groq/парсинга JSON
            last_error = exc
            logger.warning("LLM call failed (%s), attempt %s/%s: %s", description, attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    raise LLMCallError(f"Не удалось получить ответ от LLM ({description}) после {RETRY_ATTEMPTS} попыток: {last_error}")


def _parse_json_safely(content: str) -> dict:
    import re

    cleaned = (content or "").strip()
    # Удаляем reasoning-теги <think>...</think> (характерно для моделей вроде Qwen 3.6 / DeepSeek)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Сначала пробуем прямой json.loads
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Если есть markdown или пояснительный текст вокруг JSON — находим первую фигурную скобку и парсим
    idx = cleaned.find("{")
    if idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response: {content[:200]}")



async def complete_json(system_prompt: str, user_prompt: str, model: str | None = None) -> dict:
    """Просит модель вернуть строго JSON и парсит ответ. Ретраит и на сетевых ошибках, и на невалидном JSON."""

    async def _call():
        response = await get_client().chat.completions.create(
            model=model or settings.groq_text_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + "\nОтвечай ТОЛЬКО валидным JSON, без markdown-обрамления.",
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        content = response.choices[0].message.content
        return _parse_json_safely(content)

    return await _with_retries(_call, description="complete_json")


async def complete_vision_json(system_prompt: str, user_prompt: str, image_bytes: bytes, mime_type: str = "image/png", model: str | None = None) -> dict:
    """Отправляет изображение + промпт в Vision LLM и парсит ответ в JSON."""
    import base64
    import io
    from PIL import Image

    # Сжимаем изображение до макс 512px, чтобы не упираться в TPM (Tokens Per Minute) лимиты Groq
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.thumbnail((512, 512))
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        optimized_bytes = buf.getvalue()
        mime_type = "image/jpeg"
    except Exception:
        optimized_bytes = image_bytes

    b64 = base64.b64encode(optimized_bytes).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{b64}"

    async def _call():
        response = await get_client().chat.completions.create(
            model=model or settings.groq_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + "\nВсегда отвечай в формате строгого JSON объекта.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return _parse_json_safely(content)

    return await _with_retries(_call, description="complete_vision_json")



