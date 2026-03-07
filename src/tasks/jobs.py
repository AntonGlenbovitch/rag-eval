from src.core.celery_app import celery_app


@celery_app.task(name="tasks.ping")
def ping() -> str:
    return "pong"
