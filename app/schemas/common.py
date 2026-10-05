"""Common enums and primitive schemas for Tree of Life."""
from enum import Enum
from pydantic import BaseModel, ConfigDict


class CropType(str, Enum):
    RICE = "rice"
    UNKNOWN = "unknown"


class DiseaseFamily(str, Enum):
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    NUTRIENT_DEFICIENCY = "nutrient_deficiency"
    PEST_DAMAGE = "pest_damage"
    NONE = "none"
    UNKNOWN = "unknown"


class RequestStatus(str, Enum):
    SUCCESS = "success"
    RECAPTURE_REQUIRED = "recapture_required"
    INSUFFICIENT_DATA = "insufficient_data"
    ESCALATED = "escalated"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    NOT_ESTIMATED = "not_estimated"


class ActionType(str, Enum):
    MONITOR = "monitor"
    RECAPTURE_IMAGES = "recapture_images"
    CONSULT_EXPERT = "consult_expert"
    TREATMENT_ADVISORY_PENDING_VALIDATION = "treatment_advisory_pending_validation"


class VarietySource(str, Enum):
    SEED_PACKET_OCR = "seed_packet_ocr"
    BARCODE = "barcode"
    FARMER_DROPDOWN = "farmer_dropdown"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class GenomicSourceType(str, Enum):
    VARIETY_LEVEL_PUBLIC_PROFILE = "variety_level_public_profile"
    LOCAL_CURATED = "local_curated"
    LAB_ALLELE_ASSAY = "lab_allele_assay"  # Reserved in schema, unimplemented in Phase 1


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )
