"""Variety resolution schemas."""
from typing import Optional
from pydantic import Field
from app.schemas.common import BaseSchema, VarietySource


class VarietyResolution(BaseSchema):
    canonical_name: Optional[str] = None
    canonical_id: Optional[str] = None
    matched_alias: Optional[str] = None
    source: VarietySource = VarietySource.UNKNOWN
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_user_confirmation: bool = False
