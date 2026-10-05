"""Pipeline Orchestrator for Tree of Life Phase 1."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from PIL import Image

from app.schemas.response import PredictResponse, PartialResult, MANDATORY_SAFETY_NOTICE
from app.schemas.common import RequestStatus, CropType
from app.schemas.fusion import FusionResult, AuditReport, SeverityLevel
from app.services.quality.checks import DefaultQualityGate
from app.services.localizer.florence import FlorenceLocalizer
from app.services.localizer.classical import ClassicalLocalizer
from app.services.visual.efficientnet import EfficientNetClassifier
from app.services.phenotype.qwen import QwenPhenotypeExtractor
from app.services.variety.resolver import VarietyResolver
from app.services.genomic.store import GenomicStore
from app.services.environment.weather import EnvironmentService
from app.services.fusion.fuse import fuse_evidence
from app.services.fusion.uncertainty import compute_uncertainty_score, compute_model_agreement
from app.services.fusion.routing import route_decision
from app.core.config import settings
from app.db.mongo import db_manager


class Orchestrator:
    """Coordinates multi-modal evidence processing across the 12-stage pipeline."""
    def __init__(self):
        self.quality_gate = DefaultQualityGate(blur_threshold=35.0, min_resolution=224)
        self.localizer = FlorenceLocalizer() if settings.USE_REAL_FLORENCE else ClassicalLocalizer()
        self.visual_classifier = EfficientNetClassifier()
        self.phenotype_extractor = QwenPhenotypeExtractor()
        self.variety_resolver = VarietyResolver()
        self.genomic_store = GenomicStore()
        self.environment_service = EnvironmentService()

    async def run_pipeline(
        self,
        leaf_images: List[Image.Image],
        seed_packet_image: Optional[Image.Image] = None,
        variety_text: Optional[str] = None,
        district: Optional[str] = None,
        crop: str = "rice"
    ) -> PredictResponse:
        request_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        degraded_modes: List[str] = []

        # -------------------------------------------------------------
        # Stage 1: Quality Gate
        # -------------------------------------------------------------
        quality_eval = self.quality_gate.evaluate_batch(leaf_images)
        if not quality_eval.passes_gate:
            # Immediate exit: do not run heavy GPU models on unusable images
            res_quality = self.quality_gate.evaluate_image(leaf_images[0], 0)
            from app.schemas.visual import VisualResult
            from app.schemas.variety import VarietyResolution, VarietySource
            from app.schemas.genomic import GenomicResult
            from app.schemas.fusion import RecommendedAction, ActionType

            action = RecommendedAction(
                action_type=ActionType.RECAPTURE_IMAGES,
                message="Image quality is below threshold for decision support. " + " ".join(quality_eval.recapture_guidance)
            )

            response = PredictResponse(
                request_id=request_id,
                status=RequestStatus.RECAPTURE_REQUIRED,
                crop=CropType.RICE if crop == "rice" else CropType.UNKNOWN,
                variety=VarietyResolution(source=VarietySource.UNKNOWN, mapping_confidence=0.0),
                visual_result=VisualResult(
                    quality_score=quality_eval.overall_quality_score,
                    lesion_count=0
                ),
                phenotype_observations=None,
                genomic_result=GenomicResult(),
                environment_result=None,
                fusion_result=FusionResult(
                    disease="unknown",
                    severity=SeverityLevel.NOT_ESTIMATED,
                    reasons=quality_eval.reasons
                ),
                recommended_action=action,
                safety_notice=MANDATORY_SAFETY_NOTICE,
                audit=AuditReport(
                    model_versions={"quality_gate": "1.0"},
                    data_source_versions={"config": "2024.1"},
                    timestamp=now,
                    degraded_modes=["quality_gate_failed"]
                )
            )
            await self._persist_response(response)
            return response

        # -------------------------------------------------------------
        # Stage 2: EfficientNet on Full Images
        # -------------------------------------------------------------
        await self.visual_classifier.load()
        visual_full_result = await self.visual_classifier.classify(leaf_images, [])
        # Keep visual classifier in memory if batch_stage, else unload
        if settings.MODEL_RESIDENCY == "on_demand":
            await self.visual_classifier.unload()

        # -------------------------------------------------------------
        # Stage 3: Florence Localisation + Seed Packet OCR in single cycle
        # -------------------------------------------------------------
        await self.localizer.load()
        loc_results = await self.localizer.localize_and_ocr(leaf_images, seed_packet_image)
        await self.localizer.unload()

        # -------------------------------------------------------------
        # Stage 4 & 5: Crop Lesion Regions & Re-evaluate Visual Classifier
        # -------------------------------------------------------------
        crops: List[Image.Image] = []
        for l_res in loc_results:
            for det in l_res.detections:
                box = (det.xmin, det.ymin, det.xmax, det.ymax)
                if det.xmax > det.xmin and det.ymax > det.ymin:
                    crop_img = leaf_images[l_res.image_index].crop(box)
                    crops.append(crop_img)

        # -------------------------------------------------------------
        # Stage 6: Aggregate Visual Result
        # -------------------------------------------------------------
        visual_result = visual_full_result.model_copy(update={
            "lesion_count": len(crops),
            "quality_score": quality_eval.overall_quality_score
        })

        # -------------------------------------------------------------
        # Stage 7: Qwen2.5-VL Phenotype (Based on Policy)
        # -------------------------------------------------------------
        phenotype_payload = None
        should_run_qwen = False

        if settings.QWEN_POLICY == "always":
            should_run_qwen = True
        elif settings.QWEN_POLICY == "uncertain_only":
            top_prob = visual_result.top_3[0].probability if visual_result.top_3 else 0.0
            if (
                top_prob < 0.80 or
                visual_result.disagreement_score > 0.15 or
                visual_result.unknown_ood_probability > 0.30 or
                len(crops) == 0
            ):
                should_run_qwen = True

        if should_run_qwen:
            try:
                await self.phenotype_extractor.load()
                phenotype_payload = await self.phenotype_extractor.extract_phenotype(leaf_images, crops, loc_results)
                await self.phenotype_extractor.unload()
            except Exception as e:
                degraded_modes.append(f"qwen_phenotype_error: {e}")
                phenotype_payload = None

        # -------------------------------------------------------------
        # Stage 8: Variety Resolution
        # -------------------------------------------------------------
        ocr_text = loc_results[0].ocr_text if loc_results else None
        variety_resolution = self.variety_resolver.resolve(
            variety_text=variety_text,
            ocr_text=ocr_text
        )

        # -------------------------------------------------------------
        # Stage 9: Genomic Store Lookup (Variety-Level Prior)
        # -------------------------------------------------------------
        genomic_result = self.genomic_store.get_profile(variety_resolution.canonical_id)

        # -------------------------------------------------------------
        # Stage 10: Environment Weather Risk Vector
        # -------------------------------------------------------------
        env_result = None
        if district:
            try:
                env_result = await self.environment_service.get_district_risk(district)
            except Exception as e:
                degraded_modes.append(f"environment_fetch_failed: {e}")

        # -------------------------------------------------------------
        # Stage 11: Mathematical Evidence Fusion & Decision Routing
        # -------------------------------------------------------------
        top_disease, top_prob, eff_weights, reasons = fuse_evidence(
            visual=visual_result,
            genomic=genomic_result,
            phenotype=phenotype_payload,
            environment=env_result,
            variety_confidence=variety_resolution.mapping_confidence
        )

        # Calculate uncertainty & model agreement
        agreement = compute_model_agreement(top_disease, phenotype_payload)
        uncertainty = compute_uncertainty_score(
            entropy=visual_result.entropy,
            disagreement_score=visual_result.disagreement_score,
            quality_score=visual_result.quality_score,
            ood_prob=visual_result.unknown_ood_probability,
            family_conflict=1.0 - agreement
        )

        # Compute severity
        severity = SeverityLevel.NOT_ESTIMATED
        if top_disease not in ["healthy", "unknown_or_ood"]:
            leaf_area = sum(self.quality_gate.evaluate_image(img, i).leaf_mask_area for i, img in enumerate(leaf_images))
            lesion_area = sum(det.xmax - det.xmin for r in loc_results for det in r.detections)
            ratio = (lesion_area / max(1, leaf_area)) if leaf_area > 0 else 0.0
            if ratio < 0.10:
                severity = SeverityLevel.LOW
            elif ratio < 0.40:
                severity = SeverityLevel.MODERATE
            else:
                severity = SeverityLevel.HIGH

        # Route status & safety action
        status, conf_level, recommended_action = route_decision(
            fused_prob=top_prob,
            quality_score=visual_result.quality_score,
            uncertainty_score=uncertainty,
            ood_prob=visual_result.unknown_ood_probability,
            passes_gate=quality_eval.passes_gate,
            top_disease=top_disease
        )

        # -------------------------------------------------------------
        # Stage 12: Build Final Response
        # -------------------------------------------------------------
        response = PredictResponse(
            request_id=request_id,
            status=status,
            crop=CropType.RICE if crop == "rice" else CropType.UNKNOWN,
            variety=variety_resolution,
            visual_result=visual_result,
            phenotype_observations=phenotype_payload,
            genomic_result=genomic_result,
            environment_result=env_result,
            fusion_result=FusionResult(
                disease=top_disease,
                severity=severity,
                confidence_level=conf_level,
                calibrated_probability=top_prob,
                uncertainty_score=uncertainty,
                model_agreement=agreement,
                effective_branch_weights=eff_weights,
                reasons=reasons,
                prediction_set=None
            ),
            recommended_action=recommended_action,
            safety_notice=MANDATORY_SAFETY_NOTICE,
            audit=AuditReport(
                model_versions={
                    "efficientnet": "b0-7class",
                    "florence": "florence-2-large-fp16" if settings.USE_REAL_FLORENCE else "classical-cv",
                    "qwen": "qwen2.5-vl-3b-nf4"
                },
                data_source_versions={
                    "karnataka_varieties": "2024.1",
                    "classes": "2024.1"
                },
                timestamp=now,
                degraded_modes=degraded_modes
            )
        )

        await self._persist_response(response)
        return response

    async def _persist_response(self, response: PredictResponse):
        """Persist prediction request and results in MongoDB."""
        try:
            if db_manager.async_db is not None:
                db = db_manager.async_db
                doc = response.model_dump()
                doc["created_at"] = datetime.now(timezone.utc).isoformat()
                await db.prediction_requests.replace_one({"request_id": response.request_id}, doc, upsert=True)
        except Exception as e:
            print(f"[Orchestrator] Warning: Failed to persist to MongoDB: {e}")


orchestrator = Orchestrator()
