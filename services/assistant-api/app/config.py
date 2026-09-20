from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8100
    log_level: str = "INFO"
    environment: str = "local"

    telemetry_ws: str = "ws://localhost:8101/ws"
    telemetry_http: str = "http://localhost:8101"
    telemetry_replay: str = ""  # optional JSON snapshot for offline/eval runs

    rag_url: str = "http://localhost:8103"
    gateway_url: str = "http://localhost:8102"

    max_context_documents: int = 4
    prompt_version: str = "aurora-prompt-v9"
    min_windows_frames: int = 30

    # WS3 auto-fetch of the deterministic prediction explanation: the assistant
    # keeps enough 10 Hz frames for one (or, for a delta, two) prediction windows.
    prediction_window_seconds: float = 300.0
    prediction_min_frames: int = 300  # 30 s == 3 x 10 s bins (physics minimum)

    # WS4 proactive intelligence: deterministic alert detection on a cadence.
    proactive_enabled: bool = True
    proactive_poll_interval_s: float = 5.0
    proactive_max_alerts: int = 5  # cap on events worded per /v1/proactive/ask

    model_config = {"env_prefix": "AURORA_ASSISTANT_", "env_file": ".env"}


settings = Settings()