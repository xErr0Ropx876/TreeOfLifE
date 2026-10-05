"""EfficientNet-B0 visual classifier schemas."""
from typing import List
from pydantic import Field
from app.schemas.common import BaseSchema


class DiseaseProbability(BaseSchema):
    disease: str
    probability: float = Field(ge=0.0, le=1.0)


class VisualResult(BaseSchema):
    top_3: List[DiseaseProbability] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    lesion_count: int = Field(default=0, ge=0)
    unknown_ood_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    disagreement_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Jensen-Shannon divergence between full image and crop predictions")
    entropy: float = Field(default=0.0, ge=0.0, description="Normalized prediction entropy")
    temperature_applied: bool = True
