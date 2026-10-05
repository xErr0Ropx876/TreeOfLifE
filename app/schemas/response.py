"""Full API Response schemas for Tree of Life."""
from typing import Optional
from pydantic import Field
from app.schemas.common import BaseSchema, RequestStatus, CropType
from app.schemas.variety import VarietyResolution
from app.schemas.visual import VisualResult
from app.schemas.phenotype import PhenotypePayload
from app.schemas.genomic import GenomicResult
from app.schemas.environment import EnvironmentResult
from app.schemas.fusion import FusionResult, RecommendedAction, AuditReport

MANDATORY_SAFETY_NOTICE = (
    "This result is decision support, not a laboratory diagnosis. "
    "Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist."
)


class PredictResponse(BaseSchema):
    request_id: str
    status: RequestStatus
    crop: CropType = CropType.RICE
    variety: VarietyResolution
    visual_result: VisualResult
    phenotype_observations: Optional[PhenotypePayload] = None
    genomic_result: GenomicResult
    environment_result: Optional[EnvironmentResult] = None
    fusion_result: FusionResult
    recommended_action: RecommendedAction
    safety_notice: str = MANDATORY_SAFETY_NOTICE
    audit: AuditReport


class PartialResult(BaseSchema):
    request_id: str
    status: str = "processing"
    visual_top_class: Optional[str] = None
    visual_confidence: Optional[float] = None
    quality_score: float = Field(ge=0.0, le=1.0)
    passes_quality_gate: bool
