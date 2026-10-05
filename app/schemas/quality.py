"""Image quality schemas for Tree of Life."""
from typing import List
from pydantic import Field
from app.schemas.common import BaseSchema


class ImageQualityResult(BaseSchema):
    image_index: int
    is_usable: bool
    blur_score: float = Field(ge=0.0, description="Laplacian variance blur score")
    is_blurry: bool
    mean_brightness: float = Field(ge=0.0, le=255.0)
    is_underexposed: bool
    is_overexposed: bool
    is_duplicate: bool
    leaf_coverage: float = Field(ge=0.0, le=1.0, description="Green vegetation ratio")
    leaf_mask_area: int = Field(ge=0, description="Pixel count of green leaf mask for severity calculation")
    quality_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)


class AggregateQuality(BaseSchema):
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    passes_gate: bool
    reasons: List[str] = Field(default_factory=list)
    recapture_guidance: List[str] = Field(default_factory=list)
