"""
Централизованная конфигурация приложения.
Все секреты и адреса сервисов берутся из переменных окружения (.env).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Общие ---
    app_name: str = "AI Interior Designer"
    environment: str = "development"
    debug: bool = True

    # --- MongoDB ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "ai_interior_designer"

    # --- Redis (брокер Taskiq + pub/sub для WebSocket) ---
    redis_uri: str = "redis://localhost:6379/0"

    # --- S3 / MinIO (хранение планов, рендеров, экспортов) ---
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "ai-interior-designer"
    s3_region: str = "us-east-1"

    # --- Groq API (LLM-агенты) ---
    groq_api_key: str = ""
    groq_text_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str = "qwen/qwen3.6-27b"  # актуальная vision-модель Groq (проверено на console.groq.com/docs/vision); поддерживает JSON mode, до 5 изображений/запрос, лимит 20MB на изображение

    # --- CORS ---
    cors_allow_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
