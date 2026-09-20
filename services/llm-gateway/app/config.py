from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8102
    log_level: str = "INFO"

    local_enabled: bool = False
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "qwen2.5-coder:7b"

    # Ordered failover chain; "mock" is the always-available final fallback.
    provider_priority: str = "azure-openai,openai,mock"
    provider_weights: str = "azure-openai:3,openai:2,mock:1"

    # OpenAI-compatible providers (Azure OpenAI included via base_url).
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "aurora-chat"
    azure_openai_api_version: str = "2024-06-01"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"

    # PEFT (v2 LoRA) provider — serves the validated adapter2 adapter locally.
    # Gated by AURORA_GATEWAY_PEFT_ADAPTER_DIR; the old Azure/OpenAI/mock chain
    # is untouched when this is empty (the ORION-X path stays intact).
    peft_adapter_dir: str = ""
    peft_base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    peft_use_lora: bool = True
    peft_max_new_tokens: int = 64

    cache_ttl_seconds: int = 300
    cache_max_entries: int = 512
    circuitbreaker_failures: int = 3
    circuitbreaker_cooldown_s: int = 30

    default_temperature: float = 0.2
    default_max_tokens: int = 600

    model_config = {"env_prefix": "AURORA_GATEWAY_", "env_file": ".env"}


settings = Settings()