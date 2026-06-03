from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # ─── Chat LLM Provider ──────────────────────────────────────────────────────
    # Used for chat/completion calls (generate_answer, triage).
    # Defaults match current OpenRouter behavior for backwards compatibility.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""  # Takes precedence over openrouter_api_key if both set
    llm_model: str = "openrouter/free"
    llm_fallback_model: str = "openrouter/owl-alpha"  # empty string = disabled

    # ─── Embedding Provider ──────────────────────────────────────────────────────
    # Separate config because Ollama Cloud doesn't offer embeddings.
    # Must point to an OpenAI-compatible embeddings endpoint.
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""  # Must be set explicitly — no fallback to LLM_API_KEY
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 512  # MRL: text-embedding-3-small supports 512-dim truncation

    # ─── Deprecated aliases (backwards compat) ───────────────────────────────────
    # openrouter_api_key still works but LLM_API_KEY takes precedence.
    openrouter_api_key: str = ""

    # ─── Database ────────────────────────────────────────────────────────────────
    database_url: str

    # ─── RAG ─────────────────────────────────────────────────────────────────────
    chunk_size: int = 768
    chunk_overlap: int = 77
    top_k_results: int = 4

    # ─── HNSW (pgvector) ────────────────────────────────────────────────────────
    # Defaults match ~98% recall per pgvector benchmarks (ef_search=160 vs 40 default ~85%).
    # Overridable at runtime via SystemConfig (config_overlay).
    hnsw_ef_search: int = 160
    hnsw_iterative_scan: str = "on"  # "on" | "off" — enables exact re-ranking after HNSW

    # ─── App ─────────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_domain: str = "localhost:8000"

    # ─── Vision ──────────────────────────────────────────────────────────────────
    # Empty = fall back to LLM_MODEL. Set to a vision-capable model (e.g. llava).
    llm_vision_model: str = ""

    # ─── Contextual retrieval ──────────────────────────────────────────────────
    # Fast/cheap model for generating chunk context summaries. Empty = disabled.
    llm_context_model: str = ""

    # ─── Web search (E3) ─────────────────────────────────────────────────────────
    # Provider-neutral. Validate Ollama endpoint before use (see TODOS PREREQ-WS).
    web_search_url: str = ""

    # ─── STT ─────────────────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # ─── Observability ──────────────────────────────────────────────────────────
    sentry_dsn: str = ""
    environment: str = "dev"

    # ─── Admin UI ────────────────────────────────────────────────────────────────
    admin_password: str = "changeme"

    # ─── WhatsApp (optional — per-tenant credentials stored in DB) ────────────────
    wa_phone_number_id: str = ""
    wa_access_token: str = ""
    wa_app_secret: str = ""
    wa_business_id: str = ""
    wa_verify_token: str = ""

    # ─── Resolved properties ─────────────────────────────────────────────────────

    @property
    def effective_llm_api_key(self) -> str:
        """LLM_API_KEY takes precedence over openrouter_api_key."""
        return self.llm_api_key or self.openrouter_api_key

    @property
    def effective_embedding_api_key(self) -> str:
        """Embedding API key. Falls back to openrouter_api_key for backwards compat."""
        return self.embedding_api_key or self.openrouter_api_key


settings = Settings()