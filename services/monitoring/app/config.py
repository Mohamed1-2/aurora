from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8104
    log_level: str = "INFO"
    telemetry_ws: str = "ws://localhost:8101/ws"
    baseline_file: str = "baseline.json"
    baseline_frames: int = 600
    eval_frames: int = 300
    psi_threshold: float = 0.25
    ks_threshold: float = 0.01
    services: str = "assistant-api:http://localhost:8100,llm-gateway:http://localhost:8102,rag-service:http://localhost:8103,telemetry-simulator:http://localhost:8101"

    model_config = {"env_prefix": "AURORA_MONITOR_", "env_file": ".env"}


settings = Settings()