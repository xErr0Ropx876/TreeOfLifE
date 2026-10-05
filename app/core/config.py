"""Typed settings and configuration management for Tree of Life."""
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    PORT: int = 8000

    # Storage & Privacy
    UPLOAD_DIR: str = "data/uploads_tmp"
    CROPS_DIR: str = "data/processed/crops"
    RETENTION_HOURS: int = 24
    ESCALATION_RETENTION_DAYS: int = 180

    # Service Adapter Selection
    USE_REAL_FLORENCE: bool = False
    USE_REAL_QWEN: bool = False
    USE_REAL_EFFICIENTNET: bool = False

    # Model Residency & Hardware
    MODEL_RESIDENCY: str = "on_demand"
    DEVICE: str = "cpu"
    HF_HOME: str = "data/models/huggingface"
    HF_HUB_OFFLINE: int = 0

    # Policy Settings
    QWEN_POLICY: str = "always"
    QUALITY_MIN: float = 0.60
    GENOMIC_ALLOW_PLACEHOLDER: bool = True

    # MongoDB Settings
    DATABASE_TYPE: str = "mongodb"
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DB_NAME: str = "tree_of_life"


# Load configurations helper
def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    p = Path(file_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


settings = Settings()
