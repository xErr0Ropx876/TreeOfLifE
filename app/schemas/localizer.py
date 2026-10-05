"""Lesion localization and OCR schemas."""
from typing import List, Optional
from pydantic import Field
from app.schemas.common import BaseSchema


class BoundingBox(BaseSchema):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float = Field(ge=0.0, le=1.0)
    label: str = "lesion"


class LocalizerResult(BaseSchema):
    image_index: int
    detections: List[BoundingBox] = Field(default_factory=list)
    lesion_count: int = Field(ge=0)
    crops_available: bool = False
    ocr_text: Optional[str] = None
    lesion_localization_unavailable: bool = False
