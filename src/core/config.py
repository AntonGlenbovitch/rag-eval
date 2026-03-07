from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ragprobe"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    postgres_db: str = "ragprobe"
    postgres_user: str = "ragprobe"
    postgres_password: str = "ragprobe"
    postgres_host: str = "db"
    postgres_port: int = 5432
    database_url: str = "postgresql+asyncpg://ragprobe:ragprobe@db:5432/ragprobe"

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_url: str = "redis://redis:6379/0"

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


settings = Settings()
