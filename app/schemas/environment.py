"""District weather and environment risk schemas."""
from typing import Dict, Optional
from pydantic import Field
from app.schemas.common import BaseSchema


class EnvironmentResult(BaseSchema):
    used: bool = False
    district: Optional[str] = None
    temperature_celsius: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    risk_vector: Dict[str, float] = Field(default_factory=dict)
    source: str = "open_meteo_cached"
    is_rule_based_provisional: bool = True
