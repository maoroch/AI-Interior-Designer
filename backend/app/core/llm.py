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

import re

RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY_SECONDS = 3.0
_llm_lock = asyncio.Lock()


class LLMCallError(Exception):
    """Поднимается, когда все попытки вызова LLM исчерпаны."""


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


def _extract_retry_delay(error_msg: str, attempt: int) -> float:
    """Извлекает точное время ожидания из ответа Groq 429 RateLimit (например: 'try again in 2.75s')."""
    min_wait = min(2.5, RETRY_BASE_DELAY_SECONDS)
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)(?:ms|s)", error_msg, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if "ms" in match.group(0).lower():
            val = val / 1000.0
        return max(min_wait, val + 0.8)
    return max(min_wait, RETRY_BASE_DELAY_SECONDS * attempt)



async def _with_retries(coro_factory, description: str):
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with _llm_lock:
                result = await coro_factory()
                # Небольшая пауза для соблюдения лимитов TPM на Free Tier
                await asyncio.sleep(0.3)
                return result
        except Exception as exc:  # noqa: BLE001 - любые сбои Groq/парсинга JSON
            last_error = exc
            error_str = str(exc)
            delay = _extract_retry_delay(error_str, attempt)
            logger.warning(
                "LLM call failed (%s), attempt %s/%s. Sleeping %.2fs. Error: %s",
                description,
                attempt,
                RETRY_ATTEMPTS,
                delay,
                exc,
            )
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(delay)

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

    # Сжимаем изображение до макс 512px, чтобы не упираться в TPM лимиты Groq
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.thumbnail((512, 512))
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=80)
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
                    "content": system_prompt + "\nОтвечай строго валидным JSON объектом. Не используй теги <think>.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        return _parse_json_safely(content)

    return await _with_retries(_call, description="complete_vision_json")




