# Requirements Document: Phase 1 Foundation

## Introduction

Tree of Life is a local, open-source crop-disease decision-support backend for Indian smallholder farmers. Phase 1 targets rice diseases in Karnataka, operating on consumer-grade Windows hardware (RTX 4050 Laptop, 6GB VRAM) without paid APIs or cloud dependencies. The system analyzes 3–5 rice leaf photographs plus optional metadata to provide evidence-based disease identification through three branches: phenotype extraction from visual analysis, variety-level genomic susceptibility lookups, and optional environment-based risk assessment. All outputs are decision support, not laboratory diagnoses.

## Glossary

- **System**: The Tree of Life backend application
- **Quality_Gate**: Image quality validation subsystem
- **Lesion_Localizer**: Component that detects and crops disease regions in leaf images
- **Visual_Classifier**: EfficientNet-B0 deep learning model for disease classification
- **Phenotype_Extractor**: Qwen2.5-VL vision-language model for symptom description
- **Variety_Resolver**: Component that identifies rice variety from multiple sources
- **Genomic_Store**: Local database of variety-level disease susceptibility profiles
- **Environment_Service**: Component that fetches weather data and computes disease risk
- **Fusion_Engine**: Component that combines evidence from multiple branches
- **Router**: Component that determines confidence level and recommended actions
- **Orchestrator**: Main pipeline controller that sequences all processing stages
- **GPU_Lock**: Synchronization mechanism for serialized GPU model execution
- **Janitor**: Background service that deletes expired temporary data
- **API**: FastAPI REST interface
- **User**: End user submitting prediction requests (typically farmers or extension workers)
- **Expert**: Agricultural extension officer or plant pathologist
- **Candidate_Region**: A detected lesion area or full image used for classification

## Requirements

### Requirement 1: Image Ingest and Quality Validation

**User Story:** As a farmer, I want the system to validate my leaf photos before analysis, so that I receive reliable results and clear guidance when image quality is insufficient.

#### Acceptance Criteria

1.1. WHEN a prediction request is received, THE Quality_Gate SHALL validate that between 3 and 5 leaf images are provided

1.2. THE Quality_Gate SHALL reject images with resolution below 224x224 pixels

1.3. THE Quality_Gate SHALL compute Laplacian variance for each image and flag images below the configured blur threshold as blurry

1.4. THE Quality_Gate SHALL detect underexposed images where mean pixel brightness is below 30 on a 0-255 scale

1.5. THE Quality_Gate SHALL detect overexposed images where mean pixel brightness exceeds 225 on a 0-255 scale

1.6. THE Quality_Gate SHALL compute perceptual hash for each image and flag duplicate images with hash distance below 5

1.7. THE Quality_Gate SHALL compute a green vegetation mask using HSV thresholding (H: 35-85, S > 0.2, V > 0.2)

1.8. THE Quality_Gate SHALL calculate leaf coverage as the ratio of green mask area to total image area

1.9. THE Quality_Gate SHALL flag images where leaf coverage is below 30 percent as insufficient leaf coverage

1.10. THE Quality_Gate SHALL strip all EXIF metadata including GPS coordinates from uploaded images

1.11. THE Quality_Gate SHALL compute an aggregate quality score from 0.0 to 1.0 based on weighted combination of blur, exposure, duplicate, and leaf coverage checks

1.12. THE Quality_Gate SHALL store the leaf mask area in pixels for later severity computation

1.13. IF the aggregate quality score is below the configured QUALITY_MIN threshold, THEN THE System SHALL return status recapture_required with specific recapture instructions without running GPU models

1.13.1. WHEN image_quality_valid is false, THE System SHALL always include recapture guidance in the response and SHALL never suppress this guidance

### Requirement 2: Lesion Localization and OCR

**User Story:** As a developer, I want the system to detect and crop disease regions efficiently, so that classification models receive focused input while conserving GPU memory.

#### Acceptance Criteria

2.1. THE Lesion_Localizer SHALL provide a protocol interface with implementations for Florence-2-large, classical computer vision, and mock

2.1.1. THE Lesion_Localizer protocol interface SHALL define async load() and async unload() methods

2.2. WHEN the Florence-2-large localizer is invoked, THE Lesion_Localizer SHALL load the model in FP16 precision

2.3. THE Lesion_Localizer SHALL run object detection using the Florence-2 `<OD>` or `<CAPTION_TO_PHRASE_GROUNDING>` task on each leaf image

2.4. THE Lesion_Localizer SHALL apply Non-Maximum Suppression with IoU threshold loaded from configuration to deduplicate overlapping detections

2.5. THE Lesion_Localizer SHALL select up to 5 highest-confidence lesion detections per image

2.6. THE Lesion_Localizer SHALL convert detection coordinates to original image dimensions

2.7. THE Lesion_Localizer SHALL pad each detection bounding box by 10 to 20 percent

2.8. THE Lesion_Localizer SHALL save cropped lesion regions as temporary artifacts with retention expiry

2.9. WHEN a seed packet image is provided, THE Lesion_Localizer SHALL perform OCR using the Florence-2 `<OCR_WITH_REGION>` task in the same model load cycle

2.10. THE Lesion_Localizer SHALL extract text strings from seed packet OCR results

2.11. THE Lesion_Localizer SHALL unload the Florence-2 model and clear CUDA cache after completing both localization and OCR

2.12. WHERE Florence-2 is unavailable, THE Lesion_Localizer SHALL use the classical computer vision fallback implementation

2.13. THE classical fallback SHALL create a green vegetation mask using HSV thresholding (H: 35-85, S > 0.2, V > 0.2)

2.14. WHILE operating in classical fallback mode, THE Lesion_Localizer SHALL detect yellow-to-brown lesion regions within the green mask using HSV range (H: 10-40, S > 0.3)

2.15. THE classical fallback SHALL apply morphological operations and contour detection to extract bounding boxes

2.16. IF no credible lesion regions are detected, THEN THE System SHALL set lesion_localization_unavailable flag to true and use full images for classification

2.17. WHEN lesion is detected but bounding box is invalid OR detection confidence is below threshold, THE Lesion_Localizer SHALL return no crops and set lesion_localization_unavailable to true without raising an error

### Requirement 3: Visual Disease Classification

**User Story:** As a researcher, I want visual disease classification to produce calibrated probabilities with uncertainty quantification, so that downstream fusion can weight evidence appropriately.

#### Acceptance Criteria

3.1. THE Visual_Classifier SHALL provide a protocol interface with implementations for EfficientNet-B0, and mock

3.1.1. THE Visual_Classifier protocol interface SHALL define async load() and async unload() methods

3.2. WHEN the EfficientNet classifier is loaded, THE Visual_Classifier SHALL load an EfficientNet-B0 model with ImageNet-pretrained backbone weights

3.3. THE Visual_Classifier SHALL replace the final classifier layer with a 7-class output layer

3.4. THE Visual_Classifier SHALL load trained checkpoint weights from the configured model registry path

3.5. IF no trained checkpoint file exists at the registry path, THEN THE Visual_Classifier SHALL raise ModelUnavailableError

3.6. THE Visual_Classifier SHALL run inference in FP16 precision

3.7. THE Visual_Classifier SHALL resize input images to 224x224 pixels with appropriate normalization

3.8. THE Visual_Classifier SHALL compute classification logits for the full image

3.9. WHEN lesion crops are available, THE Visual_Classifier SHALL compute classification logits for each crop

3.10. THE Visual_Classifier SHALL aggregate crop predictions using mean-crop and max-crop strategies

3.11. THE Visual_Classifier SHALL compute disagreement score as the maximum probability difference between full-image and crop predictions

3.12. THE Visual_Classifier SHALL apply temperature scaling using calibration parameters loaded from the model artifact

3.13. THE Visual_Classifier SHALL convert scaled logits to probabilities using softmax

3.14. THE Visual_Classifier SHALL compute prediction entropy as a measure of uncertainty

3.15. THE Visual_Classifier SHALL compute an out-of-distribution score using energy-based or max-logit method

3.16. THE Visual_Classifier SHALL assign scores above the OOD_THRESHOLD to the unknown_or_ood class

3.17. THE Visual_Classifier SHALL return the top 3 disease predictions with probabilities

3.18. THE Visual_Classifier SHALL remain loaded in GPU memory if residency policy is batch_stage and more queued jobs exist

3.19. IF temperature scaling fails, THEN THE Visual_Classifier SHALL not pass results to fusion and SHALL mark visual_result as unavailable

### Requirement 4: Phenotype Extraction from Visual Observations

**User Story:** As a plant pathologist, I want the system to extract structured symptom descriptions, so that I can understand the visual evidence supporting each prediction.

#### Acceptance Criteria

4.1. THE Phenotype_Extractor SHALL provide a protocol interface with implementations for Qwen2.5-VL-3B-Instruct and mock

4.1.1. THE Phenotype_Extractor protocol interface SHALL define async load() and async unload() methods

4.2. WHEN the Qwen extractor is loaded, THE Phenotype_Extractor SHALL load Qwen2.5-VL-3B-Instruct in 4-bit NF4 quantization using bitsandbytes

4.3. THE Phenotype_Extractor SHALL configure min_pixels to 256 times 28 times 28 (200,704 pixels)

4.4. THE Phenotype_Extractor SHALL configure max_pixels to 512 times 28 times 28 (401,408 pixels)

4.5. THE Phenotype_Extractor SHALL set max_new_tokens to 512

4.6. THE Phenotype_Extractor SHALL use greedy decoding for deterministic output

4.7. THE Phenotype_Extractor SHALL pass the full image plus up to 5 lesion crops with bounding box coordinates as input

4.8. THE Phenotype_Extractor SHALL request structured JSON output matching the phenotype schema

4.9. THE Phenotype_Extractor SHALL validate the JSON output against the Pydantic phenotype schema

4.10. IF JSON validation fails, THEN THE Phenotype_Extractor SHALL attempt one repair using a JSON repair library

4.11. IF validation fails after repair, THEN THE Phenotype_Extractor SHALL set qwen_phenotype_available to false and continue pipeline execution

4.12. THE Phenotype_Extractor SHALL extract crop_guess from Qwen output

4.13. THE Phenotype_Extractor SHALL extract visual_quality assessment with is_usable flag and quality_score

4.14. THE Phenotype_Extractor SHALL extract lesion_observations array with per-region attributes (color, shape, margin, texture, tissue_response, affected_percent, distribution, possible_cause_family, observation_confidence)

4.15. THE Phenotype_Extractor SHALL extract whole_leaf_observations with overall_affected_percent and dominant_symptom_pattern

4.16. THE Phenotype_Extractor SHALL aggregate lesion count, mean and maximum affected area ratio, and dominant symptom attributes across regions

4.17. THE Phenotype_Extractor SHALL extract family_distribution mapping disease families to confidence scores

4.18. THE Phenotype_Extractor SHALL compute mean observation confidence from all region observations

4.19. THE Phenotype_Extractor SHALL unload the model and clear CUDA cache after extraction

4.20. WHERE the Qwen policy is set to uncertain_only, THE Phenotype_Extractor SHALL run only if visual top probability is below 0.80 OR disagreement score exceeds DISAGREEMENT_THRESHOLD OR OOD probability exceeds OOD_THRESHOLD OR no lesion localization is available

4.21. WHERE the Qwen policy is set to off, THE Phenotype_Extractor SHALL not run

### Requirement 5: Rice Variety Resolution

**User Story:** As a farmer, I want the system to identify my rice variety from multiple sources, so that variety-specific genomic information can improve prediction accuracy.

#### Acceptance Criteria

5.1. THE Variety_Resolver SHALL accept variety evidence from multiple sources: seed_packet_ocr, barcode, farmer_dropdown, expert, and unknown

5.2. THE Variety_Resolver SHALL prioritize sources in order: seed_packet_ocr, barcode, farmer_dropdown, expert

5.3. WHEN seed packet OCR text is available, THE Variety_Resolver SHALL perform fuzzy string matching against variety names and aliases using rapidfuzz library

5.4. THE Variety_Resolver SHALL return the best fuzzy match with a matching confidence score from 0.0 to 1.0

5.5. THE Variety_Resolver SHALL assign high confidence to matches with score 0.85 or above

5.6. THE Variety_Resolver SHALL assign medium confidence to matches with score between 0.60 and 0.85

5.7. THE Variety_Resolver SHALL assign low confidence to matches with score below 0.60

5.8. IF variety confidence is below 0.60, THEN THE Variety_Resolver SHALL set variety to unknown

5.9. WHEN variety source is , THE Variety_Resolver SHALL cap confidence at 0.40

5.10. THE Variety_Resolver SHALL check consistency between resolved crop and user-provided crop parameter

5.11. IF crop and variety are inconsistent, THEN THE Variety_Resolver SHALL set a crop_variety_inconsistency flag

5.12. THE Variety_Resolver SHALL return canonical_variety_name, source, and mapping_confidence

5.13. WHERE variety confidence is between 0.60 and 0.85, THE Variety_Resolver SHALL set requires_user_confirmation flag to true

5.14. IF user-provided variety_text fails to match any variety, THEN THE Variety_Resolver SHALL fall back to metadata sources in priority order: seed_packet_ocr, barcode

### Requirement 6: Genomic Susceptibility Lookup

**User Story:** As a developer, I want variety-level genomic susceptibility data to inform predictions, so that known resistance or susceptibility traits contribute to decision support while clearly distinguishing this from plant-level DNA analysis.

#### Acceptance Criteria

6.1. THE Genomic_Store SHALL maintain a local database of variety-level disease susceptibility profiles keyed by canonical variety identifier

6.2. THE Genomic_Store SHALL store a 5-disease susceptibility vector with values from 0.0 (resistant) to 1.0 (highly susceptible) for rice_blast, rice_brown_spot, bacterial_leaf_blight, rice_leaf_scald, and rice_tungro

6.3. THE Genomic_Store SHALL store an optional 64 or 128 dimensional feature vector for future learned fusion

6.4. THE Genomic_Store SHALL record source_database field indicating data provenance

6.5. THE Genomic_Store SHALL record source_version field for traceability

6.6. THE Genomic_Store SHALL mark seed data with is_placeholder flag set to true

6.7. WHERE seed data has is_placeholder set to true AND APP_ENV is production AND GENOMIC_ALLOW_PLACEHOLDER is false, THE Genomic_Store SHALL refuse to return the profile

6.8. THE Genomic_Store SHALL attach the genomic disclaimer text to every response: "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant"

6.9. THE Genomic_Store SHALL set source_type to variety_level_public_profile for all Phase 1 lookups

6.10. THE Genomic_Store SHALL reserve source_type enum value lab_allele_assay for future laboratory-based genotyping (unimplemented in Phase 1)

6.11. IF no profile exists for the requested variety, THEN THE Genomic_Store SHALL return profile_not_found status

6.12. IF variety confidence is below 0.60, THEN THE System SHALL not perform genomic lookup

### Requirement 7: Environment-Based Risk Assessment

**User Story:** As an agricultural extension worker, I want environmental conditions to contribute weak additional evidence, so that disease risk in favorable weather conditions can inform recommendations.

#### Acceptance Criteria

7.1. THE Environment_Service SHALL provide optional environment-based risk assessment

7.2. WHEN district or taluk is provided, THE Environment_Service SHALL fetch recent weather data from Open-Meteo API

7.3. THE Environment_Service SHALL cache weather data on disk keyed by district and date

7.4. THE Environment_Service SHALL compute disease risk scores based on rule-based logic defined in configuration

7.5. THE Environment_Service SHALL load disease risk rules from configs/environment_risk_rules.yaml

7.6. THE Environment_Service SHALL mark all risk rules as pending agronomist validation

7.7. THE Environment_Service SHALL compute a 5-disease risk vector with values from 0.0 (low risk) to 1.0 (high risk)

7.8. IF weather data is unavailable, THEN THE Environment_Service SHALL set environment_available to false and assign weight 0.0

7.9. THE Environment_Service SHALL default environment branch weight to 0.05

### Requirement 8: Multi-Modal Evidence Fusion

**User Story:** As a system architect, I want to fuse evidence from visual, genomic, phenotype, and environment branches mathematically, so that predictions leverage all available information while accounting for missing modalities and maintaining interpretability.

#### Acceptance Criteria

8.1. THE Fusion_Engine SHALL implement fusion as pure functions without side effects

8.2. THE Fusion_Engine SHALL load default branch weights from configuration: visual 0.70, genomic 0.15, phenotype 0.10, environment 0.05

8.3. THE Fusion_Engine SHALL operate on 7 disease classes: healthy, rice_blast, rice_brown_spot, bacterial_leaf_blight, rice_leaf_scald, rice_tungro, unknown_or_ood

8.4. THE Fusion_Engine SHALL use temperature-scaled visual logits as the visual term

8.5. THE Fusion_Engine SHALL compute genomic term as logit(clip(susceptibility, epsilon, 1-epsilon)) minus mean logit over the 5 disease classes

8.6. THE Fusion_Engine SHALL apply epsilon clamping with epsilon equal to 1e-8 before logarithm operations to prevent log(0)

8.7. THE Fusion_Engine SHALL assign genomic term value 0.0 for healthy and unknown_or_ood classes

8.8. THE Fusion_Engine SHALL compute environment term using the same centering formula as genomic term

8.9. THE Fusion_Engine SHALL assign phenotype term by mapping Qwen family probabilities to disease classes through the class-to-family map

8.10. WHEN mapping family probability to class, THE Fusion_Engine SHALL assign the full family probability to each class in that family without splitting

8.11. THE Fusion_Engine SHALL compute phenotype term as log(clip(family_prob, epsilon, 1-epsilon)) minus mean log probability over disease classes

8.12. THE Fusion_Engine SHALL scale phenotype term by mean Qwen observation confidence

8.13. THE Fusion_Engine SHALL assign phenotype term value 0.0 when Qwen output is unavailable

8.14. THE Fusion_Engine SHALL set branch weight to 0.0 for missing modalities

8.15. THE Fusion_Engine SHALL renormalize remaining branch weights to sum to 1.0

8.15.1. THE Fusion_Engine SHALL compute fused evidence when at least one modality is available

8.16. THE Fusion_Engine SHALL multiply genomic weight by variety confidence factor: 1.0 if confidence is 0.85 or above, GENOMIC_VARIETY_CONF_PENALTY (default 0.5) if confidence is between 0.60 and 0.85, and 0.0 if below 0.60

8.17. THE Fusion_Engine SHALL compute fused logits as weighted sum of branch terms

8.18. THE Fusion_Engine SHALL convert fused logits to probabilities using softmax

8.19. IF visual top probability is 0.95 or above AND quality score is 0.70 or above, THEN THE Fusion_Engine SHALL override fused top class to match visual top class and append visual_guard_override reason

8.20. THE Fusion_Engine SHALL apply quality penalty to confidence level using monotonic decreasing function

8.21. THE Fusion_Engine SHALL compute uncertainty_score as bounded weighted sum of normalized entropy, full-vs-crop disagreement, family conflict between Qwen family and class family, (1 minus quality score), and OOD probability

8.22. THE Fusion_Engine SHALL append machine-readable reason codes for each contributing factor

8.23. THE Fusion_Engine SHALL compute model_agreement score based on consistency between visual, phenotype family, and genomic susceptibility

### Requirement 9: Severity Estimation

**User Story:** As a farmer, I want to know how severely my crop is affected, so that I can prioritize treatment decisions.

#### Acceptance Criteria

9.1. THE System SHALL compute severity from the union area of lesion detection regions divided by leaf mask area

9.2. THE System SHALL compute lesion union area by creating a binary mask and setting all detected lesion regions to 1

9.3. THE System SHALL handle overlapping detections by computing pixel-wise union

9.4. THE System SHALL classify severity as low when lesion ratio is below 0.10

9.5. THE System SHALL classify severity as moderate when lesion ratio is between 0.10 and 0.40

9.6. THE System SHALL classify severity as high when lesion ratio is above 0.40

9.7. WHERE lesion detections are unavailable AND Qwen overall_affected_percent is available, THE System SHALL compute severity ratio from Qwen percentage divided by 100

9.8. THE System SHALL set severity to not_estimated for healthy or unknown_or_ood classes

9.9. THE System SHALL set severity to not_estimated when neither detections nor Qwen percentage are available

9.10. THE System SHALL label severity estimates as approximate in response field descriptions

### Requirement 10: Routing and Action Recommendation

**User Story:** As a user, I want the system to recommend appropriate actions based on confidence level, so that I know whether to monitor, recapture, or consult an expert.

#### Acceptance Criteria

10.1. THE Router SHALL assign confidence_level high when fused top probability is 0.80 or above AND quality score is 0.70 or above AND entropy is below configured threshold AND no severe modality contradiction exists AND OOD probability is below threshold

10.2. THE Router SHALL assign confidence_level moderate when fused top probability is between 0.60 and 0.80 OR quality or disagreement are moderate

10.3. THE Router SHALL assign confidence_level low and status escalated when fused top probability is below 0.60 OR quality score is below 0.50 OR OOD probability exceeds threshold OR strong disagreement exists between visual and phenotype or genomic evidence OR crop variety inconsistency flag is set OR no usable lesion detection exists with low visual confidence

10.3.1. WHEN confidence_level is below 0.80, THE Router SHALL limit recommended_action to monitor, recapture_images, or consult_expert only

10.4. THE Router SHALL set status to insufficient_data when crop cannot be identified OR images are unusable OR no class exceeds minimum probability threshold

10.5. THE Router SHALL recommend action_type monitor for high confidence healthy or low-severity disease predictions

10.6. THE Router SHALL recommend action_type recapture_images when status is recapture_required

10.7. THE Router SHALL recommend action_type consult_expert when status is escalated

10.8. THE Router SHALL recommend action_type treatment_advisory_pending_validation for moderate or high severity disease predictions with sufficient confidence, with message indicating agronomist validation is pending

10.9. THE Router SHALL never include pesticide brand names or quantities in recommended actions

### Requirement 11: Orchestration and GPU Management

**User Story:** As a system administrator, I want GPU resources to be managed safely, so that concurrent requests do not cause out-of-memory errors on limited hardware.

#### Acceptance Criteria

11.1. THE Orchestrator SHALL execute processing stages in order: Quality_Gate, Visual_Classifier on full images, Lesion_Localizer (localization and OCR), crop generation, Visual_Classifier on crops, partial result emission, Phenotype_Extractor per policy, Variety_Resolver, Genomic_Store lookup, Environment_Service, Fusion_Engine, Router, response construction

11.2. THE Orchestrator SHALL use GPU_Lock to serialize GPU-intensive stages

11.3. THE GPU_Lock SHALL provide an async-safe lock mechanism

11.4. THE Orchestrator SHALL wrap blocking model inference calls in asyncio.to_thread or executor with max_workers equal to 1

11.5. THE Orchestrator SHALL never block the event loop with synchronous model calls

11.6. THE Orchestrator SHALL unload Florence-2 model immediately after completing localization and OCR

11.7. THE Orchestrator SHALL unload Qwen model immediately after phenotype extraction

11.8. THE Orchestrator SHALL call garbage collection and torch.cuda.empty_cache after unloading heavy models

11.9. THE Orchestrator SHALL log GPU VRAM usage before and after each heavy model operation

11.10. THE GPU_Lock SHALL support residency policies: on_demand (load and unload per job) and batch_stage (load once and process queued jobs)

11.11. THE GPU_Lock SHALL detect out-of-memory errors and retry on CPU

11.12. WHERE a model fails on GPU with OOM error, THE Orchestrator SHALL mark device as cpu and degraded flag as true in audit log

11.13. THE Orchestrator SHALL continue execution when any optional stage fails, marking that stage unavailable in the response

11.14. IF Visual_Classifier fails on GPU, THEN THE Orchestrator SHALL retry on CPU before returning insufficient_data status

### Requirement 12: API Endpoints and Request Handling

**User Story:** As a client application developer, I want RESTful endpoints with clear contracts, so that I can integrate disease prediction into user-facing applications.

#### Acceptance Criteria

12.1. THE API SHALL expose all endpoints under the /api/v1 prefix

12.2. THE API SHALL provide endpoint POST /disease/predict accepting multipart form data

12.3. THE predict endpoint SHALL accept 3 to 5 leaf image files in images parameter

12.4. THE predict endpoint SHALL accept optional seed_packet image file

12.5. THE predict endpoint SHALL accept required crop parameter (default: rice)

12.6. THE predict endpoint SHALL accept optional variety_text parameter

12.7. THE predict endpoint SHALL accept optional district and taluk parameters

12.8. THE predict endpoint SHALL accept optional growth_stage parameter

12.9. THE predict endpoint SHALL validate uploaded file types are image/jpeg, image/png, or image/jpg

12.10. THE predict endpoint SHALL enforce maximum upload size of 10 MB per image

12.11. THE predict endpoint SHALL return HTTP 202 with request_id for async processing

12.12. WHERE query parameter wait is true, THE predict endpoint SHALL block up to configured timeout waiting for result

12.13. THE API SHALL provide endpoint GET /disease/result/{request_id} to retrieve prediction results

12.14. THE API SHALL provide endpoint GET /jobs/{request_id} to retrieve job status

12.15. THE API SHALL emit partial results with quality and visual verdict before full report when requested

12.16. THE API SHALL provide endpoint POST /variety/resolve to resolve variety from text or barcode

12.17. THE API SHALL provide endpoint GET /genotype/profile/{canonical_variety_id} returning metadata only without raw VCF data

12.18. THE API SHALL provide endpoint POST /expert/escalate to submit structured case reports

12.19. THE escalate endpoint SHALL require consent_given flag in request body

12.20. WHERE images_included is true in escalation request, THE escalate endpoint SHALL validate that consent_given is also true

12.21. THE API SHALL provide endpoint GET /models/health returning loaded models, VRAM usage, versions, and queue depth

12.22. THE API SHALL implement rate limiting per client IP address

12.23. THE API SHALL maintain a bounded async job queue

12.24. WHEN the job queue is full, THE API SHALL return HTTP 429 or 503 with retry-after header

12.25. THE API SHALL include request_id in all log entries for request tracing

12.26. THE API SHALL emit audit events for prediction requests, escalations, and errors


### Requirement 13: Response Schema and Safety Notices

**User Story:** As a farmer, I want prediction responses to clearly communicate confidence, evidence, and safety disclaimers, so that I can make informed decisions.

#### Acceptance Criteria

13.1. THE System SHALL return responses matching the PredictResponse Pydantic schema

13.2. THE response SHALL include request_id field

13.3. THE response SHALL include status field with values: success, recapture_required, insufficient_data, or escalated

13.4. THE response SHALL include crop field

13.5. THE response SHALL include variety object with canonical_name, source, and mapping_confidence

13.6. THE response SHALL include visual_result object with top_3 predictions array, quality_score, lesion_count, and unknown_ood_probability

13.7. THE response SHALL include phenotype_observations object when Qwen extraction succeeds

13.8. THE response SHALL include genomic_result object with used flag, confidence, source_type, is_placeholder flag, and disclaimer text

13.9. THE response SHALL include fusion_result object with disease, severity, confidence_level, calibrated_probability, uncertainty_score, model_agreement, and reasons array

13.10. THE response SHALL include recommended_action object with action_type and message

13.11. THE response SHALL include safety_notice field with text: "This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist"

13.12. THE response SHALL include audit object with model_versions, data_source_versions, timestamp, privacy_actions array, and degraded_modes array

13.13. THE response SHALL include prediction_set field reserved for future conformal prediction (always null in Phase 1)

13.14. THE fusion_result object SHALL include prediction_set field reserved for future conformal prediction (always null in Phase 1)

13.15. THE response SHALL surface is_placeholder flag when genomic data is placeholder data

13.16. THE System SHALL validate response completeness before returning to client

13.17. IF response is missing required fields confidence_level, evidence arrays, or safety_notice, THEN THE System SHALL block the response and return an error status

### Requirement 14: Database Schema and Migrations

**User Story:** As a database administrator, I want a clear schema with proper relationships and retention policies, so that data integrity is maintained and privacy requirements are met.

#### Acceptance Criteria

14.1. THE System SHALL use SQLAlchemy 2.0 ORM with Alembic migrations

14.2. THE System SHALL support SQLite for development and PostgreSQL for production using generic JSON column type with JSONB variant

14.3. THE System SHALL define tables: users, farms, crop_activities, prediction_requests, uploaded_images, image_quality_results, florence_detections, qwen_phenotypes, efficientnet_predictions, variety_aliases, variety_accession_map, genomic_profiles, genomic_feature_vectors, fusion_results, expert_escalations, model_versions, and audit_events

14.4. THE System SHALL define primary keys, foreign keys, and appropriate indexes on all tables

14.5. THE System SHALL include created_at and updated_at timestamp columns on all tables

14.6. THE System SHALL include retention_expires_at column on tables containing user data

14.7. THE System SHALL include data_source_version column where applicable for traceability

14.8. THE System SHALL store complex payloads in JSON columns

14.9. THE expert_escalations table SHALL be self-contained without foreign key dependencies on prediction_requests

14.10. THE expert_escalations table SHALL include case_summary, visual_artifacts_summary, phenotype_observations, fusion_result, and user_notes as JSON fields

14.11. THE expert_escalations table SHALL include images_included boolean field

14.12. THE expert_escalations table SHALL include optional image_urls field for temporary signed URLs

14.13. THE expert_escalations table SHALL include consent_given boolean field

14.14. THE expert_escalations table SHALL include status field with values: open, under_review, resolved, or archived

14.15. THE expert_escalations table SHALL include retention_expires_at with default 180 days from creation

14.16. THE System SHALL provide Alembic baseline migration creating all tables

14.17. THE System SHALL provide idempotent seed script to populate initial variety aliases, class definitions, and seed genomic profiles

### Requirement 15: Data Retention and Privacy

**User Story:** As a privacy officer, I want automated data deletion and privacy safeguards, so that the system complies with data protection principles.

#### Acceptance Criteria

15.1. THE Janitor SHALL run as a background scheduled task

15.2. THE Janitor SHALL delete uploaded images and crop files where retention_expires_at is before current time

15.3. THE Janitor SHALL delete prediction_requests rows where retention_expires_at is before current time AND no open expert escalation references the request

15.4. THE Janitor SHALL check for open escalations before deleting prediction requests

15.5. THE Janitor SHALL cascade delete related rows: uploaded_images, image_quality_results, florence_detections, qwen_phenotypes, efficientnet_predictions, and fusion_results when parent prediction_request is deleted

15.6. THE Janitor SHALL delete expert_escalations where retention_expires_at is before current time AND status is not open

15.7. THE Janitor SHALL never delete expert_escalations with status open regardless of retention date

15.8. THE Janitor SHALL never delete variety_aliases, variety_accession_map, genomic_profiles, genomic_feature_vectors, model_versions, or audit_events tables

15.9. THE System SHALL default prediction_requests retention to 24 hours

15.10. THE System SHALL default expert_escalations retention to 180 days configurable via ESCALATION_RETENTION_DAYS

15.11. THE System SHALL reduce location data to district or taluk level only, never storing precise GPS coordinates

15.12. THE System SHALL log privacy_actions taken including EXIF stripped and location reduced to district

### Requirement 16: Model Registry and Versioning

**User Story:** As a model governance lead, I want all model versions tracked and auditable, so that predictions can be traced to specific model artifacts.

#### Acceptance Criteria

16.1. THE System SHALL maintain a model_versions table recording name, version, framework, weights_path, weights_hash, device, quantization, license, and loaded_at

16.2. THE System SHALL load model configurations from configs directory

16.3. THE System SHALL validate model weight files exist at configured paths before loading

16.4. THE System SHALL compute and verify hash of model weight files

16.5. THE System SHALL record model versions in audit log for each prediction

16.6. THE System SHALL include model versions in response audit object

16.7. THE System SHALL load class definitions from configs/classes.yaml

16.8. THE System SHALL load fusion thresholds from configs/thresholds.yaml

16.9. THE System SHALL load variety seed data from configs/varieties_seed.yaml

16.10. THE System SHALL never hard-code class lists or thresholds in business logic code

### Requirement 17: Testing and Quality Assurance

**User Story:** As a quality assurance engineer, I want comprehensive test coverage including mock-based and hardware-based tests, so that the system can be validated on any machine.

#### Acceptance Criteria

17.1. THE System SHALL include unit tests for quality checks operating on synthetic images

17.2. THE System SHALL include unit tests for NMS algorithm

17.3. THE System SHALL include unit tests for Qwen JSON validation and repair logic

17.4. THE System SHALL include unit tests for visual classifier aggregation strategies

17.5. THE System SHALL include unit tests for variety fuzzy matching with known aliases

17.6. THE System SHALL include unit tests for genomic lookup including placeholder-disabled path

17.7. THE System SHALL include unit tests for fusion weight renormalization with missing modalities

17.8. THE System SHALL include unit tests for phenotype term family-to-class mapping without splitting probability

17.9. THE System SHALL include unit tests for epsilon clamping before logarithm operations

17.10. THE System SHALL include unit tests for visual guard override condition

17.11. THE System SHALL include unit tests for severity computation from lesion area ratio

17.12. THE System SHALL include one unit test per routing rule covering all status and confidence_level outcomes

17.13. THE System SHALL include unit tests for janitor expiry logic including open escalation checks

17.14. THE System SHALL include integration tests for mocked end-to-end /predict endpoint with status success

17.15. THE System SHALL include integration tests for recapture_required status when quality is below threshold

17.16. THE System SHALL include integration tests for insufficient_data status

17.17. THE System SHALL include integration tests for escalated status

17.18. THE System SHALL include integration tests for degraded mode when Qwen returns invalid JSON

17.19. THE System SHALL include integration tests for degraded mode when Florence detection fails

17.20. THE System SHALL include integration tests for degraded mode when genomic profile is missing

17.21. THE System SHALL include tests that all mocked services and entire pipeline execute successfully on CPU without GPU

17.22. THE System SHALL mark GPU-dependent tests with pytest.mark.gpu decorator

17.23. THE GPU-marked tests SHALL be skipped when CUDA is unavailable

17.24. THE System SHALL include GPU smoke tests validating Florence-2 loads and runs detection

17.25. THE System SHALL include GPU smoke tests validating Qwen loads and extracts valid JSON

17.26. THE System SHALL include GPU smoke tests validating EfficientNet loads and classifies

17.27. THE System SHALL include GPU smoke tests validating all real models fit within 6 GB VRAM budget

17.28. THE System SHALL include end-to-end latency test measuring total prediction time with real models

17.29. THE System SHALL include honesty grep test validating no forbidden phrases exist in codebase

17.30. THE honesty grep test SHALL be implemented in pure Python using pathlib and re modules

17.31. THE honesty grep test SHALL strip known allowed texts (safety notice and genomic disclaimer) before searching

17.32. THE honesty grep test SHALL search for forbidden patterns: "DNA of this plant", "sequenc(ed|ing) (this|the) plant", "diagnos(is|ed)" outside safety notice, and "pesticide.*[0-9]+.*(ml|gram|kg)"

17.33. THE honesty grep test SHALL scan .py, .yaml, .yml, and .md files under app/, configs/, and docs/ directories

17.34. THE System SHALL include integration test validating escalation requires consent_given flag

17.35. THE System SHALL include integration test validating escalation with images_included true requires consent

17.36. THE System SHALL assert exact status values in test assertions: success, recapture_required, insufficient_data, or escalated

17.37. THE test mocks SHALL use quality_score as float from 0.0 to 1.0

17.38. THE test mocks SHALL use min_pixels consistently as 256 times 28 times 28

### Requirement 18: Error Handling and Graceful Degradation

**User Story:** As a reliability engineer, I want the system to degrade gracefully when components fail, so that users receive partial results rather than complete failures.

#### Acceptance Criteria

18.1. THE System SHALL define a typed error hierarchy for all error conditions

18.2. THE System SHALL use consistent JSON error response bodies

18.3. THE System SHALL log all errors with request_id, timestamp, and stack trace

18.4. WHEN any optional pipeline stage fails, THE System SHALL log the error, mark that stage unavailable in the response, and continue execution

18.4.1. WHEN a component fails, THE System SHALL set component_state to DEGRADED in the audit log

18.5. WHEN Visual_Classifier fails on GPU, THE System SHALL retry inference on CPU

18.6. IF Visual_Classifier fails on CPU after GPU retry, THEN THE System SHALL return status insufficient_data with explanation

18.7. WHEN Florence-2 model fails to load, THE System SHALL fall back to classical computer vision localizer

18.8. WHEN Qwen model fails or returns invalid JSON after repair, THE System SHALL set qwen_phenotype_available to false and continue without phenotype evidence

18.9. WHEN genomic profile is not found, THE System SHALL continue fusion with genomic weight set to 0

18.10. WHEN environment service fails to fetch weather data, THE System SHALL continue fusion with environment weight set to 0

18.11. THE System SHALL populate degraded_modes array in audit object listing all fallback modes activated

18.12. THE System SHALL never crash the entire request due to a single component failure unless that component is the visual classifier

### Requirement 19: Configuration Management

**User Story:** As a deployment engineer, I want all configuration externalized and validated, so that behavior can be adjusted without code changes.

#### Acceptance Criteria

19.1. THE System SHALL load all configuration from environment variables and .env file using pydantic-settings

19.2. THE System SHALL validate all configuration values at startup using Pydantic validators

19.3. THE System SHALL fail fast at startup if required configuration is missing or invalid

19.4. THE System SHALL load APP_ENV setting controlling production vs development behavior

19.5. THE System SHALL load QUALITY_MIN threshold with default 0.50 marked as placeholder requiring calibration

19.6. THE System SHALL load DISAGREEMENT_THRESHOLD with default 0.15 marked as placeholder requiring calibration

19.7. THE System SHALL load OOD_THRESHOLD with default 0.30 marked as placeholder requiring calibration

19.8. THE System SHALL load VISUAL_UNCERTAIN_THRESHOLD with default 0.80

19.9. THE System SHALL load GENOMIC_VARIETY_CONF_PENALTY with default 0.5

19.10. THE System SHALL load GENOMIC_ALLOW_PLACEHOLDER with default false for production safety

19.11. THE System SHALL load QWEN_POLICY with default always and options: always, uncertain_only, or off

19.12. THE System SHALL load MODEL_RESIDENCY with options: on_demand or batch_stage

19.13. THE System SHALL load USE_REAL_FLORENCE configuration flag to control Lesion_Localizer service factory selection

19.14. THE System SHALL load USE_REAL_QWEN configuration flag to control Phenotype_Extractor service factory selection

19.15. THE System SHALL load USE_REAL_EFFICIENTNET configuration flag to control Visual_Classifier service factory selection

19.16. THE System SHALL load ESCALATION_RETENTION_DAYS with default 180 to control expert escalation retention period

19.17. THE System SHALL load NMS_IOU_THRESHOLD for lesion detection Non-Maximum Suppression

19.18. THE System SHALL load HF_HOME for Hugging Face model cache directory

19.19. THE System SHALL support HF_HUB_OFFLINE mode for air-gapped deployments

19.20. THE System SHALL load database connection URL from DATABASE_URL

19.21. THE System SHALL load upload size limits, rate limiting parameters, and queue size from configuration

19.22. THE System SHALL document in README and inline comments that DISAGREEMENT_THRESHOLD and OOD_THRESHOLD are placeholders requiring validation set calibration

### Requirement 20: Documentation and Setup

**User Story:** As a new developer, I want clear setup instructions for Windows with CUDA, so that I can run the system on my local machine.

#### Acceptance Criteria

20.1. THE README SHALL include prerequisites: Python 3.11, NVIDIA GPU with CUDA support, and Git

20.2. THE README SHALL include instructions to verify NVIDIA driver with nvidia-smi command

20.3. THE README SHALL include instructions to create Python virtual environment

20.4. THE README SHALL include PyTorch CUDA installation command obtained from pytorch.org with placeholder [VERIFIED_VERSION]

20.5. THE README SHALL include instructions to install bitsandbytes on Windows

20.6. THE README SHALL include instructions to copy .env.example to .env and configure HF_HOME

20.7. THE README SHALL include instructions to run scripts/download_models.py to fetch model weights

20.8. THE README SHALL include instructions to run Alembic migrations and seed script

20.9. THE README SHALL include instructions to start FastAPI server with uvicorn

20.10. THE README SHALL include instructions to run tests with pytest on CPU

20.11. THE README SHALL include instructions to run GPU smoke tests with pytest -m gpu

20.12. THE README SHALL include troubleshooting section for out-of-memory errors

20.13. THE README SHALL include troubleshooting section for Florence remote code and flash_attn issues

20.14. THE README SHALL include troubleshooting section for bitsandbytes Windows compatibility

20.15. THE README SHALL include instructions to run latency benchmarking script

20.16. THE README SHALL include section documenting verified dependency versions with placeholders filled after hardware testing: [VERIFIED_VERSION], [CUDA_VERSION], [VRAM_MEASURED]

20.17. THE README SHALL include reference to docs/setup-validation.md documenting tested configuration

20.18. THE docs/setup-validation.md SHALL document tested date, verified dependency versions including transformers and bitsandbytes, and measured peak VRAM usage

20.19. THE README SHALL never claim specific version numbers until verified on target hardware

20.20. THE System SHALL include .env.example file with all required environment variables and comments

20.21. THE System SHALL include separate requirements files: requirements.txt (runtime), requirements-dev.txt (testing and linting), and requirements-research.txt (DNABERT-2 and XGBoost, never imported by API)

---

## Definition of Done

Phase 1 Foundation is complete when:

1. All requirements are implemented and verified
2. Mocked end-to-end pipeline runs successfully on CPU (requirements 1-12, 17.21)
3. Real model adapters load and execute within 6GB VRAM budget on target hardware (requirements 2-4, 11, 17.24-17.27)
4. Honesty grep test passes with no forbidden phrases (requirement 17.29-17.33) - **test failure blocks overall completion**
5. All configuration files exist with documented placeholder thresholds (requirements 16.7-16.9, 19)
6. README includes setup instructions with version placeholders (requirement 20)
7. All unit and integration tests pass (requirement 17)
8. Degraded mode paths are tested for all optional components (requirements 18, 17.18-17.20)
9. Database migrations and seed scripts are idempotent (requirement 14.16-14.17)
10. docs/latency.md is populated with measured values from actual hardware
11. Expert portal endpoints (GET /expert/cases, POST /expert/cases/{id}/reply) are implemented and tested (requirements 12.27-12.34)
12. Expert case queue filtering by status works correctly
13. Expert reply audit trail preserves all feedback entries with timestamps
