# Tasks Document: Phase 1 Foundation

This document defines the actionable, dependency-ordered task breakdown for **TREE OF LIFE Phase 1 Foundation**, adhering strictly to `.kiro/steering/*`, `documentation/phase1-plan-revisions.md`, `requirements.md`, and `design.md`.

All tasks are structured such that **a fully mocked end-to-end pipeline (Milestones 1–8) works completely on CPU** before any real heavy model adapter (Milestones 11–15) is loaded.

---

## Task Roadmap & Milestones

```
[Milestone 1: Configs & Schemas]
       │
       ▼
[Milestone 2: Database & Seed]
       │
       ▼
[Milestone 3: Storage & Janitor]
       │
       ▼
[Milestone 4: Protocols & Deterministic Mocks]
       │
       ▼
[Milestone 5: Quality Gate & Leaf Masking]
       │
       ▼
[Milestone 6: Mathematical Fusion & Routing (Pure Functions)]
       │
       ▼
[Milestone 7: Pipeline Orchestrator (Mocked + Basic GpuLock)]
       │
       ▼
[Milestone 8: FastAPI Endpoints, Async Queue & Expert Portal]
       │
       ★ MIDPOINT: 100% CPU Mocked Pipeline Green
       │
       ▼
[Milestone 9: Variety Resolution & Genomic Prior Lookup]
       │
       ▼
[Milestone 10: Environment Weather & Risk Vector (Optional)]
       │
       ▼
[Milestone 11: Florence-2 Localizer + Classical CV Fallback]
       │
       ▼
[Milestone 12: Qwen2.5-VL 4-Bit Phenotype & JSON Repair]
       │
       ▼
[Milestone 13: EfficientNet-B0 Classifier + Temperature Scaling + OOD]
       │
       ▼
[Milestone 14: Hardened GPU Lock, Residency Policies & OOM Fallback]
       │
       ▼
[Milestone 15: Hardware Smoke Suite, Honesty Grep & Windows README]
```

---

## Milestone 1: Configuration, Registries, and Typed Schemas
- [ ] **Task 1.1: Core Configuration Files**
  - Create `configs/classes.yaml` defining 7 target classes (`healthy`, `rice_blast`, `rice_brown_spot`, `bacterial_leaf_blight`, `rice_leaf_scald`, `rice_tungro`, `unknown_or_ood`) and their disease family mapping (`fungal`, `bacterial`, `viral`).
  - Create `configs/varieties_seed.yaml` defining canonical Karnataka varieties (IR64, BPT-5204/Sona Masuri, Swarna/MTU-7029, MTU-1010, RNR-15048, Pusa Basmati 1121) and aliases.
  - Create `configs/fusion_config.yaml` specifying default weights (visual: 0.70, genomic: 0.15, phenotype: 0.10, environment: 0.05), `DISAGREEMENT_THRESHOLD: 0.15`, `OOD_THRESHOLD: 0.30`, `VISUAL_UNCERTAIN_THRESHOLD: 0.80`, and `GENOMIC_VARIETY_CONF_PENALTY: 0.5`.
  - Create `configs/classical_lesion.yaml` with HSV bounds for green leaf mask (H: 35–85, S > 0.2, V > 0.2) and lesion candidate mask (H: 10–40, S > 0.3).
  - Create `.env.example` with runtime settings (`APP_ENV=development`, `MODEL_RESIDENCY=on_demand`, `QUALITY_MIN=0.60`, `GENOMIC_ALLOW_PLACEHOLDER=true`).

- [ ] **Task 1.2: Settings Management**
  - Implement `app/core/config.py` using `pydantic-settings` to parse environment variables and load YAML configuration registries into typed immutable dataclasses.

- [ ] **Task 1.3: Pydantic v2 Schemas**
  - Implement `app/schemas/common.py` (Base schemas, enums: `CropType`, `DiseaseFamily`, `ConfidenceLevel`, `ActionType`, `RequestStatus`).
  - Implement `app/schemas/quality.py` (Per-image quality checks, blur score, exposure, duplicate distance, leaf coverage, aggregate score).
  - Implement `app/schemas/localizer.py` (Bounding box coordinates, detection confidence, label, crop metadata).
  - Implement `app/schemas/phenotype.py` (Strict Qwen output schema: `crop_guess`, `visual_quality`, `lesion_observations`, `whole_leaf_observations`, `family_distribution`, `limitations`).
  - Implement `app/schemas/visual.py` (Top-3 predictions, crop aggregations, disagreement score, entropy, OOD score).
  - Implement `app/schemas/variety.py` (Canonical variety name, alias matched, source, confidence, `requires_user_confirmation`).
  - Implement `app/schemas/genomic.py` (Susceptibility vector, source database, `is_placeholder`, mandatory honesty disclaimer).
  - Implement `app/schemas/fusion.py` (Fused disease, severity, confidence level, calibrated probability, uncertainty score, model agreement, contributing reasons, reserved `prediction_set: list[str] | None = None`).
  - Implement `app/schemas/response.py` (`PredictResponse`, `PartialResult`, `AuditReport`, `ErrorResponse`).

- [ ] **Task 1.4: Unit Tests for Milestone 1**
  - `tests/unit/test_config.py`: Verify settings load YAML configs without magic numbers.
  - `tests/unit/test_schemas.py`: Verify round-trip validation and field serialization for all schemas.

---

## Milestone 2: Relational Database & Seed Management
- [ ] **Task 2.1: SQLAlchemy 2.0 Base & Tables**
  - Implement `app/db/base.py` and `app/db/models.py`:
    - `prediction_requests`: ID, status, crop, variety metadata, timestamps, retention expiry.
    - `uploaded_images`: Foreign key to request, storage path, original filename, EXIF stripped flag.
    - `image_quality_results`: Foreign key to image, blur, exposure, duplicate, leaf coverage, leaf mask area.
    - `florence_detections`: Detection bounding boxes, crops storage path.
    - `qwen_phenotypes`: Validated phenotype JSON payload, repair attempts.
    - `efficientnet_predictions`: Full and crop logits, probabilities, entropy, OOD score.
    - `variety_aliases` & `variety_accession_map`: Canonical names and aliases index.
    - `genomic_profiles`: Variety susceptibility vectors, placeholder flags, data source versions.
    - `fusion_results`: Final fused class, probability, severity, uncertainty score, reasons list.
    - `expert_escalations`: Self-contained snapshot for MedHive, `consent_given` flag, `images_included` flag, 180-day retention.
    - `model_versions` & `audit_events`: Model hashes, devices, degraded mode logs.

- [ ] **Task 2.2: Session Management & Repositories**
  - Implement `app/db/session.py` with sync engine configured for threadpool execution.
  - Implement `app/db/repositories.py` containing all query and persistence methods (keeping services free from direct DB queries).

- [ ] **Task 2.3: Idempotent Database Seeding**
  - Implement `scripts/seed_db.py` reading `configs/varieties_seed.yaml` to populate varieties, aliases, and placeholder genomic profiles.

- [ ] **Task 2.4: Unit Tests for Milestone 2**
  - `tests/unit/test_db_models.py`: Verify table creation and relationships.
  - `tests/unit/test_seed.py`: Verify seed idempotency and placeholder tags.

---

## Milestone 3: Privacy-Compliant Storage & Janitor
- [ ] **Task 3.1: File Storage with Privacy Stripping**
  - Implement `app/pipeline/storage.py`:
    - File upload handling with magic byte validation (JPEG/PNG/WebP, Pillow verification).
    - EXIF metadata stripping (eliminating GPS, camera serials, timestamps).
    - Temporary artifact management with unique UUIDs.

- [ ] **Task 3.2: Automated Retention Janitor**
  - Implement `app/pipeline/janitor.py`:
    - Background task scanning for records with `retention_expires_at < now()`.
    - Deletes expired uploads, temporary lesion crops, and cascaded DB records.
    - **Critical Privacy Protection**: NEVER deletes `expert_escalations` with `status="open"`, and NEVER deletes prediction requests referenced by open escalations.

- [ ] **Task 3.3: Unit Tests for Milestone 3**
  - `tests/unit/test_storage.py`: Verify EXIF stripping and magic byte checks.
  - `tests/unit/test_janitor.py`: Verify expired records cleanup and escalation preservation.

---

## Milestone 4: Service Interfaces & Deterministic Mocks
- [ ] **Task 4.1: Service Protocols with Lifecycle**
  - Implement `app/services/interfaces.py` defining type-safe Protocols:
    - `LoadableService`: Protocol with `async def load()` and `async def unload()`.
    - `VisualClassifier(LoadableService)`
    - `LesionLocalizer(LoadableService)`
    - `PhenotypeExtractor(LoadableService)`
    - `VarietyResolver`
    - `GenomicStore`
    - `EnvironmentService`

- [ ] **Task 4.2: Deterministic Mocks**
  - Implement `app/services/visual/mock.py`: Returns deterministic logits, entropy, and OOD scores.
  - Implement `app/services/localizer/mock.py`: Returns synthetic lesion boxes and seed-packet OCR text.
  - Implement `app/services/phenotype/mock.py`: Returns schema-compliant phenotype JSON without external calls.
  - Implement `app/services/variety/mock.py`: Returns deterministic resolution for seed varieties.
  - Implement `app/services/genomic/mock.py`: Returns curated susceptibility vectors with placeholder tags.
  - Implement `app/services/environment/mock.py`: Returns synthetic weather risk vectors.

- [ ] **Task 4.3: Service Factory**
  - Implement `app/services/factory.py`: Instantiates mocks or real adapters based on configuration flags (`USE_REAL_FLORENCE`, `USE_REAL_QWEN`, `USE_REAL_EFFICIENTNET`).

- [ ] **Task 4.4: Unit Tests for Milestone 4**
  - `tests/unit/test_mocks.py`: Verify every mock executes on CPU and returns schema-compliant data.

---

## Milestone 5: Quality Gate & Severity Mask Pre-computation
- [ ] **Task 5.1: Image Quality Checks**
  - Implement `app/services/quality/checks.py`:
    - Resolution check (reject < 224x224).
    - Blur check (Laplacian variance vs threshold).
    - Exposure check (mean brightness in [30, 225]).
    - Perceptual hash check (detect near-duplicate images with Hamming distance < 5).
    - Green vegetation segmentation (HSV green mask: H: 35–85, S > 0.2, V > 0.2).
    - Leaf coverage ratio (green mask / total area; flag < 30%).
    - Aggregate quality calculation (0.0 to 1.0).
    - Cache `leaf_mask_area` in pixels for downstream severity calculations.

- [ ] **Task 5.2: Recapture Routing Trigger**
  - If aggregate quality < `QUALITY_MIN`, flag request as `recapture_required` and abort GPU execution immediately with actionable feedback.

- [ ] **Task 5.3: Unit Tests for Milestone 5**
  - `tests/unit/test_quality_gate.py`: Test on blurry, overexposed, and synthetic leaf images.

---

## Milestone 6: Mathematical Fusion & Routing (Pure Functions)
- [ ] **Task 6.1: Evidence Term Computations**
  - Implement `app/services/fusion/fuse.py`:
    - Visual term: temperature-scaled logits.
    - Genomic term: `term_c = logit(clip(s_c, eps, 1-eps)) - mean_over_diseases(logit(s))` (centred over 5 diseases; healthy and unknown get 0).
    - Phenotype term: full family probability assigned to each member disease class (no splitting), clamped with `eps=1e-8`, log-transformed, and centred.
    - Environment term: weather risk vector centred over disease classes.
    - Weight re-normalization: missing/disabled modalities set to 0 and active weights renormalised.
    - Variety confidence scaling: medium-confidence variety matches scale genomic weight by `0.5`.

- [ ] **Task 6.2: Visual Guard & Uncertainty Metric**
  - Implement `app/services/fusion/uncertainty.py`:
    - Hard visual guard: if visual top probability >= 0.95 and quality >= 0.70, top fused class cannot be overridden by genomic/env priors.
    - Uncertainty metric: bounded combination of entropy, Jensen-Shannon full-crop divergence, family conflict, (1 - quality), and OOD probability.

- [ ] **Task 6.3: Severity Computation**
  - Implement `app/services/visual/aggregate.py`:
    - Compute union of lesion candidate boxes on leaf mask.
    - `severity_ratio = lesion_union_area / leaf_mask_area`.
    - Ratio < 10% -> `low`; 10–40% -> `moderate`; > 40% -> `high`; fallback to Qwen `overall_affected_percent`; `not_estimated` for healthy/unknown.

- [ ] **Task 6.4: Decision Routing & Action Assignment**
  - Implement `app/services/fusion/routing.py`:
    - Status routing: `success` | `recapture_required` | `insufficient_data` | `escalated`.
    - Confidence level: `high` | `moderate` | `low`.
    - Recommended action: `monitor` | `recapture_images` | `consult_expert` | `treatment_advisory_pending_validation` (strict safety notices; zero pesticide brands/quantities).

- [ ] **Task 6.5: Unit Tests for Milestone 6**
  - `tests/unit/test_fusion.py`: Test weight renormalisation, epsilon clamping, family-to-class mapping, visual guard override, and routing rules.

---

## Milestone 7: Pipeline Orchestrator (Mocked + Basic GpuLock)
- [ ] **Task 7.1: Async GPU Synchronization Primitive**
  - Implement `app/pipeline/gpu_lock.py`: Simple async lock serializing heavy processing stages.

- [ ] **Task 7.2: Execution Orchestrator**
  - Implement `app/pipeline/orchestrator.py` following strict sequence:
    1. Quality Gate (extract & save leaf mask area).
    2. EfficientNet on Full Images (early visual signal).
    3. Florence-2: Localisation + Seed Packet OCR in single load -> immediate unload.
    4. Generate padded lesion crops.
    5. EfficientNet on Crops -> aggregate with full image predictions.
    6. Emit `partial_result`.
    7. Qwen2.5-VL Phenotype extraction (per `QWEN_POLICY: always | uncertain_only | off`) -> immediate unload.
    8. Variety Resolution (fuzzy match from OCR / dropdown).
    9. Genomic Store lookup.
    10. Environment risk calculation.
    11. Fusion & Routing (compute severity, calibrated probability, actions).
    12. Build complete `PredictResponse`.

- [ ] **Task 7.3: Graceful Degradation Handling**
  - Failure in any optional component (Qwen, Florence, Genomic, Weather) logs audit event, sets `degraded_modes`, and completes prediction without raising 500 error.

- [ ] **Task 7.4: Unit Tests for Milestone 7**
  - `tests/unit/test_orchestrator.py`: Test mocked end-to-end flow, degraded mode handling, and lock serialization.

---

## Milestone 8: FastAPI Endpoints, Async Queue & Expert Portal
- [ ] **Task 8.1: API Factory & Lifespan**
  - Implement `app/main.py`: Lifespan startup (DB initialization, seed check, queue worker initialization) and shutdown.

- [ ] **Task 8.2: REST Endpoints**
  - Implement `app/api/v1/predict.py`: `POST /api/v1/disease/predict` with multipart image uploads, optional query `wait=true`.
  - Implement `app/api/v1/jobs.py`: `GET /api/v1/disease/result/{request_id}` with support for fast `partial_result`.
  - Implement `app/api/v1/variety.py`: `POST /api/v1/variety/resolve`.
  - Implement `app/api/v1/genotype.py`: `GET /api/v1/genotype/profile/{canonical_variety_id}` (metadata only, no raw genomic data).
  - Implement `app/api/v1/expert.py`: `POST /api/v1/expert/escalate` (requires explicit `consent_given`).
  - Implement `app/api/v1/health.py`: `GET /api/v1/models/health` (device, queue depth, memory).

- [ ] **Task 8.3: Bounded Queue & Rate Limiter**
  - Implement `app/pipeline/queue.py`: In-process async queue with bounded size; returns 429/503 when full.

- [ ] **Task 8.4: Integration Tests for Milestone 8 (Midpoint)**
  - `tests/integration/test_api_mocked.py`: Verify all endpoints with mocked services on CPU. All tests green!

---

## Milestone 9: Variety Resolution & Genomic Prior Lookup
- [ ] **Task 9.1: Variety Resolver with RapidFuzz**
  - Implement `app/services/variety/fuzzy.py` and `resolver.py`:
    - Priority: Seed-packet OCR > Barcode > Controlled dropdown > Expert.
    - Confidence bands: >= 0.85 (High), 0.60–0.85 (Medium, requires user confirmation), < 0.60 (Unknown).

- [ ] **Task 9.2: Curated Genomic Store**
  - Implement `app/services/genomic/store.py` and `lookup.py`:
    - Precomputed 5-disease susceptibility vector lookup.
    - Enforces placeholder flag and disclaimers.
    - Disabled in production unless `GENOMIC_ALLOW_PLACEHOLDER=true`.

- [ ] **Task 9.3: Unit Tests for Milestone 9**
  - `tests/unit/test_variety_genomic.py`: Test fuzzy matching aliases, medium confidence penalties, and placeholder enforcement.

---

## Milestone 10: Environment Weather & Disease Risk
- [ ] **Task 10.1: Weather Service & Disk Cache**
  - Implement `app/services/environment/weather.py`:
    - District-level Open-Meteo API client with 24-hour file-backed cache.

- [ ] **Task 10.2: Agronomic Risk Rules**
  - Implement `app/services/environment/risk_rules.py`:
    - Rule-based risk vector from humidity, temperature, and rainfall (marked pending agronomist validation).

- [ ] **Task 10.3: Unit Tests for Milestone 10**
  - `tests/unit/test_environment.py`: Test caching, network fallback, and rule vector computation.

---

## Milestone 11: Florence-2 Localizer + Classical CV Fallback
- [ ] **Task 11.1: Florence-2 Real Adapter**
  - Implement `app/services/localizer/florence.py`:
    - FP16 precision, `<OD>` lesion detection, `<OCR_WITH_REGION>` seed packet OCR in single load.
    - IoU NMS deduplication, top K=5 lesion crops.
    - Explicit unload and CUDA memory release.

- [ ] **Task 11.2: Classical CV Fallback**
  - Implement `app/services/localizer/classical.py`:
    - Green vegetation mask segmentation.
    - Yellow/brown lesion contour extraction inside green mask.

- [ ] **Task 11.3: Tests for Milestone 11**
  - `tests/unit/test_classical_localizer.py`: Test CPU classical fallback.
  - Smoke tests with `@pytest.mark.gpu` for Florence-2.

---

## Milestone 12: Qwen2.5-VL 4-Bit Phenotype & JSON Repair
- [ ] **Task 12.1: Qwen2.5-VL Adapter**
  - Implement `app/services/phenotype/qwen.py`:
    - 4-bit NF4 bitsandbytes quantization, greedy decoding, max_new_tokens=512.
    - Strict system prompt returning observation schema without disease names.

- [ ] **Task 12.2: JSON Validation & Repair**
  - Implement `app/services/phenotype/json_repair.py`:
    - Pydantic validation with single repair retry before falling back to `qwen_phenotype_available=false`.

- [ ] **Task 12.3: Tests for Milestone 12**
  - `tests/unit/test_json_repair.py`: Test repair logic on malformed JSON.
  - Smoke tests with `@pytest.mark.gpu` for Qwen2.5-VL.

---

## Milestone 13: EfficientNet-B0 Classifier + Temperature Scaling + OOD
- [ ] **Task 13.1: EfficientNet Real Adapter**
  - Implement `app/services/visual/efficientnet.py`:
    - Torchvision EfficientNet-B0 backbone with 7-class head.
    - Checkpoint validation; raises `ModelUnavailableError` if weights missing.

- [ ] **Task 13.2: Temperature Scaling & OOD Scoring**
  - Implement `app/services/visual/calibration.py` and `ood.py`:
    - Temperature scaling artifact loading.
    - Energy-based out-of-distribution calculation.

- [ ] **Task 13.3: Tests for Milestone 13**
  - `tests/unit/test_calibration_ood.py`: Test scaling and OOD threshold triggers.

---

## Milestone 14: Hardened GPU Lock, Residency Policies & OOM Fallback
- [ ] **Task 14.1: Hardened GpuLock**
  - Enhance `app/pipeline/gpu_lock.py`:
    - Track active model residency (`on_demand` vs `batch_stage`).
    - VRAM logging before and after invocations.
    - Catch CUDA OOM errors and trigger automatic fallback to CPU.

- [ ] **Task 14.2: Tests for Milestone 14**
  - `tests/unit/test_gpu_lock_hardened.py`: Test residency transitions and OOM fallback handling.

---

## Milestone 15: Hardware Smoke Suite, Honesty Grep & Windows Setup
- [ ] **Task 15.1: Honesty Grep Test**
  - Implement `tests/unit/test_honesty.py`:
    - Pure Python scanner verifying codebase does NOT contain forbidden claims (e.g. "DNA of this plant", "sequencing this plant", raw pesticide quantities).

- [ ] **Task 15.2: Hardware Benchmark & Smoke Scripts**
  - Implement `scripts/download_models.py` and `scripts/benchmark_latency.py`.

- [ ] **Task 15.3: Production README & Setup Validation**
  - Create `README.md` and `docs/setup-validation.md` with explicit Windows CUDA instructions and `[VERIFIED_VERSION]` placeholders.
