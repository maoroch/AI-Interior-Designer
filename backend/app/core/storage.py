"""
Обёртка над S3-совместимым хранилищем (MinIO локально, R2/S3 в проде).
Хранит загруженные планы, экспортированные рендеры и PDF.
"""
import uuid
from functools import lru_cache

import boto3
from botocore.client import Config

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name=settings.s3_region,
    )


def ensure_bucket() -> None:
    client = get_s3_client()
    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if settings.s3_bucket_name not in existing:
        client.create_bucket(Bucket=settings.s3_bucket_name)


def upload_bytes(data: bytes, key_prefix: str, extension: str, content_type: str) -> str:
    """Загружает файл и возвращает object key (не URL — URL строится отдельно, т.к. может быть подписанным)."""
    key = f"{key_prefix}/{uuid.uuid4().hex}.{extension}"
    get_s3_client().put_object(
        Bucket=settings.s3_bucket_name, Key=key, Body=data, ContentType=content_type
    )
    return key


def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    return get_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )
