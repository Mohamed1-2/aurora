from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vehicle_id: str = "ORION-X-TEST-01"
    hz: int = 10
    seed: int = 20240627
    buffer_frames: int = 1500
    port: int = 8101
    log_level: str = "INFO"
    tum_uds_dir: str = ""

    model_config = {"env_prefix": "AURORA_TELEMETRY_", "env_file": ".env"}


settings = Settings()