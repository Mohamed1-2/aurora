from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    kb_dir: str = "services/rag-service/knowledge_base"
    corpus_file: str | None = None
    port: int = 8103
    top_k: int = 4
    log_level: str = "INFO"
    gateway_url: str = "http://llm-gateway:8102"

    model_config = {"env_prefix": "AURORA_RAG_", "env_file": ".env"}


settings = Settings()