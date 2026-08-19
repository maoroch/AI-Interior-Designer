"""
Taskiq-брокер на Redis. Каждый тяжёлый агент — отдельная задача в очереди,
что позволяет пайплайну переживать рестарты и перезапускаться с чекпоинта
(нужно, т.к. Architect Agent может пересчитать геометрию и потребовать
повторного прогона части пайплайна).

Запуск воркера:
    taskiq worker app.tasks.broker:broker
"""
from taskiq import TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

from app.core.config import get_settings

settings = get_settings()

result_backend = RedisAsyncResultBackend(
    redis_url=settings.redis_uri,
    socket_timeout=None,
    socket_connect_timeout=10,
    socket_keepalive=True,
    health_check_interval=30,
)

broker = ListQueueBroker(
    url=settings.redis_uri,
    socket_timeout=None,
    socket_connect_timeout=10,
    socket_keepalive=True,
    health_check_interval=30,
).with_result_backend(result_backend)

scheduler = TaskiqScheduler(broker=broker, sources=[LabelScheduleSource(broker)])

