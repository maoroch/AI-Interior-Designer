"""
Подключение к MongoDB через Motor (async-драйвер).

Коллекции (см. README -> "Схема данных MongoDB"):
- projects        — метаданные проекта, статус пайплайна
- scenes          — версии сцены (JSON-описание 3D-сцены)
- chat_messages   — история диалога с Conversation Agent
- agent_runs      — трейс выполнения каждого агента (для статуса/дебага/чекпоинтов)
- furniture_library — внутренний реестр мебельных моделей (не пользовательский каталог)
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

settings = get_settings()

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[settings.mongo_db_name]


def projects_collection():
    return get_database()["projects"]


def scenes_collection():
    return get_database()["scenes"]


def chat_messages_collection():
    return get_database()["chat_messages"]


def agent_runs_collection():
    return get_database()["agent_runs"]


def furniture_library_collection():
    return get_database()["furniture_library"]


async def ping() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False


async def ensure_indexes() -> None:
    """Индексы, которые понадобятся с первых запросов."""
    await scenes_collection().create_index([("project_id", 1), ("version", -1)])
    await chat_messages_collection().create_index([("project_id", 1), ("created_at", 1)])
    await agent_runs_collection().create_index([("project_id", 1), ("created_at", -1)])
    await furniture_library_collection().create_index([("category", 1), ("style", 1)])
