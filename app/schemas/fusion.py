"""Mathematical evidence fusion and routing schemas."""
from typing import List, Dict, Optional
from pydantic import Field
from app.schemas.common import BaseSchema, ConfidenceLevel, SeverityLevel, ActionType


class FusionResult(BaseSchema):
    disease: str = "unknown"
    severity: SeverityLevel = SeverityLevel.NOT_ESTIMATED
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    calibrated_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    model_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    effective_branch_weights: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    prediction_set: Optional[List[str]] = Field(
        default=None,
        description="Reserved for split-conformal prediction sets (null in Phase 1)"
    )


class RecommendedAction(BaseSchema):
    action_type: ActionType
    message: str


class AuditReport(BaseSchema):
    model_versions: Dict[str, str] = Field(default_factory=dict)
    data_source_versions: Dict[str, str] = Field(default_factory=dict)
    timestamp: str
    privacy_actions: List[str] = Field(default_factory=lambda: [
        "EXIF stripped (GPS and device metadata removed)",
        "Location reduced to district/taluk level only",
        "Ephemeral image retention timer initialized (24h default)"
    ])
    degraded_modes: List[str] = Field(default_factory=list)
