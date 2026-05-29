from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # API Keys
    openrouter_api_key: str

    # Database
    database_url: str

    # Models
    llm_model: str = "openrouter/free"
    llm_fallback_model: str = "openrouter/owl-alpha"  # empty string = disabled
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

    # WhatsApp (optional — per-tenant credentials stored in DB)
    wa_phone_number_id: str = ""
    wa_access_token: str = ""
    wa_app_secret: str = ""
    wa_business_id: str = ""
    wa_verify_token: str = ""


settings = Settings()
