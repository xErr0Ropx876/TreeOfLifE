"""Genomic susceptibility prior schemas."""
from typing import Dict, Optional
from pydantic import Field
from app.schemas.common import BaseSchema, GenomicSourceType

MANDATORY_GENOMIC_DISCLAIMER = (
    "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant."
)


class GenomicResult(BaseSchema):
    used: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    canonical_variety_id: Optional[str] = None
    susceptibility_vector: Dict[str, float] = Field(default_factory=dict)
    source_type: GenomicSourceType = GenomicSourceType.VARIETY_LEVEL_PUBLIC_PROFILE
    source_database: str = "local_curated"
    source_version: str = "2024.1"
    is_placeholder: bool = True
    disclaimer: str = MANDATORY_GENOMIC_DISCLAIMER
