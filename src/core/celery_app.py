from celery import Celery

from src.core.config import settings

celery_app = Celery(
    "ragprobe",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.tasks.jobs"],
)

celery_app.conf.update(task_track_started=True)
