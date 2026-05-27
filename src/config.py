from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # API Keys
    openrouter_api_key: str

    # Database
    database_url: str

    # Models
    llm_model: str = "openrouter/owl-alpha"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 4

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_domain: str = "localhost:8000"

    # STT
    groq_api_key: str = ""

    # Observability
    sentry_dsn: str = ""
    environment: str = "dev"

    # Admin UI
    admin_password: str = "changeme"


settings = Settings()
