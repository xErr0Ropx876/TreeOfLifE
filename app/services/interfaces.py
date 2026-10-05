"""Service Protocols for Tree of Life.

Every model-backed service defines async load() and unload() lifecycle methods.
"""
from typing import Protocol, List, Dict, Optional, runtime_checkable
from PIL import Image
from app.schemas.quality import ImageQualityResult, AggregateQuality
from app.schemas.localizer import LocalizerResult
from app.schemas.phenotype import PhenotypePayload
from app.schemas.visual import VisualResult
from app.schemas.variety import VarietyResolution
from app.schemas.genomic import GenomicResult
from app.schemas.environment import EnvironmentResult


@runtime_checkable
class LoadableService(Protocol):
    """Lifecycle protocol for heavy services requiring GPU/model load and release."""
    async def load(self) -> None:
        """Load model weights into memory/device."""
        ...

    async def unload(self) -> None:
        """Unload model and invoke CUDA garbage collection."""
        ...


@runtime_checkable
class QualityGate(Protocol):
    """Protocol for image quality checking and vegetation masking."""
    def evaluate_image(self, image: Image.Image, index: int) -> ImageQualityResult:
        ...

    def evaluate_batch(self, images: List[Image.Image]) -> AggregateQuality:
        ...


@runtime_checkable
class LesionLocalizer(LoadableService, Protocol):
    """Protocol for Florence-2 or classical CV lesion localization and OCR."""
    async def localize_and_ocr(
        self,
        leaf_images: List[Image.Image],
        seed_packet_image: Optional[Image.Image] = None
    ) -> List[LocalizerResult]:
        ...


@runtime_checkable
class VisualClassifier(LoadableService, Protocol):
    """Protocol for EfficientNet-B0 visual disease classification."""
    async def classify(
        self,
        full_images: List[Image.Image],
        crop_images: List[Image.Image]
    ) -> VisualResult:
        ...


@runtime_checkable
class PhenotypeExtractor(LoadableService, Protocol):
    """Protocol for Qwen2.5-VL structured symptom observation."""
    async def extract_phenotype(
        self,
        full_images: List[Image.Image],
        crop_images: List[Image.Image],
        localizer_results: List[LocalizerResult]
    ) -> Optional[PhenotypePayload]:
        ...


@runtime_checkable
class VarietyResolver(Protocol):
    """Protocol for fuzzy matching rice varieties from OCR/dropdown/barcode."""
    def resolve(
        self,
        variety_text: Optional[str] = None,
        ocr_text: Optional[str] = None,
        barcode: Optional[str] = None
    ) -> VarietyResolution:
        ...


@runtime_checkable
class GenomicStore(Protocol):
    """Protocol for retrieving variety-level susceptibility priors."""
    def get_profile(self, canonical_variety_id: str) -> Optional[GenomicResult]:
        ...


@runtime_checkable
class EnvironmentService(Protocol):
    """Protocol for district weather fetching and risk vector computation."""
    async def get_district_risk(self, district: str) -> Optional[EnvironmentResult]:
        ...
