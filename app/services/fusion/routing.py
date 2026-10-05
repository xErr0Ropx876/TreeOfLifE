"""Decision routing and safety recommendations for Tree of Life Phase 1."""
from typing import Tuple
from app.schemas.common import RequestStatus, ConfidenceLevel, ActionType
from app.schemas.fusion import RecommendedAction


def route_decision(
    fused_prob: float,
    quality_score: float,
    uncertainty_score: float,
    ood_prob: float,
    passes_gate: bool,
    top_disease: str
) -> Tuple[RequestStatus, ConfidenceLevel, RecommendedAction]:
    """Pure routing function mapping fused outputs to verified decision outcomes."""
    if not passes_gate or quality_score < 0.50:
        return (
            RequestStatus.RECAPTURE_REQUIRED,
            ConfidenceLevel.LOW,
            RecommendedAction(
                action_type=ActionType.RECAPTURE_IMAGES,
                message="Image quality is insufficient for decision support. Please retake steady, in-focus photos of the affected leaves in indirect natural daylight."
            )
        )

    if ood_prob >= 0.30 or top_disease == "unknown_or_ood":
        return (
            RequestStatus.ESCALATED,
            ConfidenceLevel.LOW,
            RecommendedAction(
                action_type=ActionType.CONSULT_EXPERT,
                message="Visual patterns show high uncertainty or potential non-target conditions. Escalate to an agricultural extension officer or plant pathologist for field inspection."
            )
        )

    if top_disease == "healthy":
        return (
            RequestStatus.SUCCESS,
            ConfidenceLevel.HIGH if fused_prob >= 0.80 else ConfidenceLevel.MODERATE,
            RecommendedAction(
                action_type=ActionType.MONITOR,
                message="Crop leaves appear healthy with no significant lesion symptoms detected. Continue standard monitoring and nutrient management practices."
            )
        )

    # Disease classes routing
    if fused_prob >= 0.80 and quality_score >= 0.70 and uncertainty_score <= 0.40:
        return (
            RequestStatus.SUCCESS,
            ConfidenceLevel.HIGH,
            RecommendedAction(
                action_type=ActionType.TREATMENT_ADVISORY_PENDING_VALIDATION,
                message="High-confidence decision support. Verified disease symptoms detected. Consult your local Krishi Vigyan Kendra (KVK) or extension officer for approved integrated pest management (IPM) guidelines."
            )
        )
    elif fused_prob >= 0.60:
        return (
            RequestStatus.SUCCESS,
            ConfidenceLevel.MODERATE,
            RecommendedAction(
                action_type=ActionType.TREATMENT_ADVISORY_PENDING_VALIDATION,
                message="Moderate-confidence decision support. Monitor symptom progression closely. Confirm suspected symptoms with a local agricultural extension officer."
            )
        )
    else:
        return (
            RequestStatus.ESCALATED,
            ConfidenceLevel.LOW,
            RecommendedAction(
                action_type=ActionType.CONSULT_EXPERT,
                message="Low-confidence decision support. Symptoms are inconclusive or multiple stress factors may be present. Expert diagnosis recommended before taking intervention."
            )
        )
