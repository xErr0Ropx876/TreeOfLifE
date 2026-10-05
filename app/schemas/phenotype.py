"""Strict Qwen2.5-VL Phenotype Observation Schema."""
from typing import List, Dict, Optional
from pydantic import Field
from app.schemas.common import BaseSchema, DiseaseFamily


class LesionObservation(BaseSchema):
    region_id: str
    bbox: List[int] = Field(description="[ymin, xmin, ymax, xmax] relative to region")
    lesion_color: str
    lesion_shape: str
    lesion_margin: str
    surface_texture: str
    tissue_response: str = Field(description="chlorotic | necrotic | water_soaked | powdery | wilted | unknown")
    estimated_region_affected_percent: float = Field(ge=0.0, le=100.0)
    distribution: str = Field(description="isolated | scattered | clustered | streaked | diffuse | unknown")
    possible_cause_family: DiseaseFamily
    visible_symptoms: List[str] = Field(default_factory=list)
    observation_confidence: float = Field(ge=0.0, le=1.0)


class WholeLeafObservations(BaseSchema):
    overall_affected_percent: float = Field(ge=0.0, le=100.0)
    dominant_symptom_pattern: str
    needs_more_images: bool = False
    requested_images: List[str] = Field(default_factory=list)


class VisualQualityAssessment(BaseSchema):
    is_usable: bool
    issues: List[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)


class PhenotypePayload(BaseSchema):
    crop_guess: str = "rice"
    visual_quality: VisualQualityAssessment
    lesion_observations: List[LesionObservation] = Field(default_factory=list)
    whole_leaf_observations: WholeLeafObservations
    limitations: List[str] = Field(default_factory=list)
    family_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores per disease family (fungal, bacterial, viral, nutrient_deficiency, pest_damage, unknown)"
    )
    mean_observation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
