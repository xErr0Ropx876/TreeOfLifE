# Kiro Spec Prompt 01 — Phase 1 Foundation (paste into a NEW SPEC session)

> Before pasting: `.kiro/steering/{product,tech,structure}.md` and `docs/research-notes.md` must already exist in the workspace.

---

## PROMPT (copy from here)

Create a spec called **phase1-foundation** for TREE OF LIFE.
Follow `.kiro/steering/*` strictly (honesty rules, hardware budget, service/mock pattern, structure). Use `#docs/research-notes.md` for background only.

**Process:** first write `requirements.md` in EARS style ("WHEN ... THE SYSTEM SHALL ..."), then STOP and wait for my approval. After approval write `design.md`, stop again, then `tasks.md`. Order tasks so a **fully mocked end-to-end `/predict` works by the midpoint**, and real model adapters come after.

### Goal
A FastAPI backend that turns 3–5 rice leaf photos (+ optional seed-packet photo, variety text, district, growth stage) into a structured decision-support response using two evidence branches (phenotype; variety-level genomic prior) plus an optional weak environment branch, fused with calibrated uncertainty and routed to high / moderate / low-escalate / insufficient-data / recapture outcomes.

### In scope (this spec)
Schemas, config, DB + migrations + seed, storage + janitor, quality gate, service interfaces + mocks, real adapters for Florence-2 / Qwen2.5-VL / EfficientNet (inference only, behind flags), variety resolver, genotype lookup, fusion + routing, orchestrator, GPU lock + queue, endpoints, tests, README (Windows CUDA).
### Out of scope
EfficientNet training code, DNABERT-2/XGBoost pipeline, React UI, learned fusion MLP, real genotype data ingestion, lab assays, treatment recommendations beyond the four allowed action types.

### Classes and registries
`classes.yaml`: healthy, rice_blast, rice_brown_spot, bacterial_leaf_blight, rice_leaf_scald, rice_tungro, unknown_or_ood. Disease-family map: blast, brown_spot, scald = fungal; BLB = bacterial; tungro = viral. Varieties (seed): IR64, BPT-5204 (Sona Masuri), Swarna (MTU-7029), MTU-1010, RNR-15048, Pusa Basmati 1121, with common aliases. All registry-driven.

### Functional requirements

**1. Image ingest + quality gate** (before any heavy model): min resolution, Laplacian-variance blur, exposure (under/over), perceptual-hash duplicate detection, leaf coverage (green/vegetation ratio via simple segmentation), EXIF strip, per-image and aggregate quality score 0–1. If aggregate quality < `QUALITY_MIN` -> status `recapture_required` with specific recapture instructions, and no GPU model is run.

**2. Lesion localisation + OCR** — interface `LesionLocalizer` with implementations: Florence-2-large (FP16; `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>`, `<OCR_WITH_REGION>`; boxes in original-image coordinates, NMS/IoU de-dup, top K=3–5, 10–20 % padded crops saved as temp artifacts), a **classical CV fallback** (colour-threshold lesion candidates on the leaf mask), and a mock. If no credible region: `lesion_localization_unavailable=true` and downstream uses the full image. Florence is unloaded and CUDA cache cleared after use. Seed-packet OCR is a separate pluggable `OcrEngine` (Florence default; interface allows EasyOCR/PaddleOCR later because packets may contain Kannada/Hindi).

**3. Phenotype extraction** — `PhenotypeExtractor` (Qwen2.5-VL-3B-Instruct 4-bit). Input: full image + up to K crops + boxes. Output validated against the Qwen schema below. One repair retry, then `qwen_phenotype_available=false`. Aggregate across regions (lesion count, mean/max area ratio, dominant colour/shape/margin, tissue response, family distribution, mean observation confidence). Qwen never outputs a disease name. Config `QWEN_POLICY = always | uncertain_only | off` (default `always`).

**4. Visual classification** — `VisualClassifier` (torchvision EfficientNet-B0, 224x224, FP16, 7 outputs). Runs on full image and crops; aggregates (full probs, mean-crop, max-crop, disagreement score); temperature scaling loaded from model artifact; entropy; OOD score (energy/max-logit) feeding `unknown_or_ood`. Weights loaded from registry path; if no trained checkpoint exists the real adapter raises a typed `ModelUnavailable` and the pipeline falls back per error policy.

**5. Variety resolution** — priority: seed-packet OCR (fuzzy match with `rapidfuzz` against aliases) > barcode/QR > controlled dropdown > expert > photo guess (weak, max confidence 0.40). OCR/barcode matches are *suggestions needing confirmation* unless confidence >= 0.85. Output per the variety schema; confidence bands: >= 0.85 high, 0.60–0.85 medium, < 0.60 low. Inconsistent crop/variety -> flag.

**6. Genotype lookup** — local store keyed by canonical variety ID returning a precomputed 5-disease susceptibility vector (+ optional 64/128-d vector), `source_database`, `source_version`, `is_placeholder`. Seed data are clearly fake deterministic fixtures marked `is_placeholder=true`, `source_database="local_curated"`, and are disabled when `APP_ENV=production` unless `GENOMIC_ALLOW_PLACEHOLDER=true`. Always attach the genomic disclaimer. Reserve `source_type="lab_allele_assay"` in enums, unimplemented.

**7. Environment (optional)** — district-level only; Open-Meteo fetch with on-disk cache; rule-based risk vector (placeholder rule table in `configs/`, marked pending agronomist validation). If unavailable: weight 0.

**8. Fusion** (pure functions in `services/fusion`):
- Default weights (config): visual 0.70, genomic 0.15, phenotype 0.10, environment 0.05.
- Classes: 7. Visual term = temperature-scaled visual logits. Genomic and environment terms apply to the 5 disease classes only: `term_c = logit(clip(s_c, eps, 1-eps)) - mean_over_diseases(logit(s))` (centred so a generally susceptible variety does not inflate every disease); healthy/unknown get 0. Phenotype term = centred log of the Qwen family probability mapped through the class→family map, scaled by mean Qwen confidence; 0 when Qwen unavailable.
- `fused_logits = Σ w_i' * term_i`, where missing modalities get weight 0 and the rest are renormalised; genomic weight multiplied by variety factor (1.0 if conf >= 0.85, configurable reduced factor for 0.60–0.85, 0 below 0.60 or if profile missing/placeholder-disabled); then softmax.
- Guard: if visual top prob >= 0.95 and quality >= 0.70, fused top class must equal visual top class (add reason code).
- Quality penalty on confidence (monotonic, configurable). `uncertainty_score` = bounded weighted sum of normalised entropy, full-vs-crop disagreement, family conflict (Qwen family vs class family), (1 - quality), OOD probability; document as a heuristic, not a calibrated probability.
- Every contributing factor appends a machine-readable `reason`.

**9. Routing**
- `high`: top fused prob >= 0.80 AND quality >= 0.70 AND entropy below threshold AND no severe modality contradiction AND no OOD trigger.
- `moderate`: 0.60–<0.80, or moderate quality/disagreement.
- `low` -> `escalated`: < 0.60, OR quality < 0.50, OR OOD prob above threshold, OR strong visual-vs-phenotype/genotype disagreement, OR crop/variety inconsistency, OR no usable lesion detection with low visual confidence.
- `insufficient_data`: crop not identified, images unusable, or no class above minimum probability.
- Actions: `monitor | recapture_images | consult_expert | treatment_advisory_pending_validation` (the last only as a stub message; no pesticide names/quantities).

**10. Error policy:** any optional stage failure -> log, mark unavailable in response, continue. Visual branch failure on GPU -> retry on CPU -> else `insufficient_data` with explanation. All errors use a typed error hierarchy and a consistent JSON error body.

### API (prefix `/api/v1`)
- `POST /disease/predict` (multipart: 3–5 leaf images, optional seed_packet image, crop, variety_text, district/taluk, growth_stage, optional env fields). Returns `202` + `request_id`; `?wait=true` blocks up to a timeout (dev). Bounded queue; `429/503` when full.
- `GET /disease/result/{request_id}` (also `/jobs/{id}` status; supports a fast partial result with quality + visual verdict before the full report).
- `POST /variety/resolve`, `GET /genotype/profile/{canonical_variety_id}` (metadata only, never raw VCF), `POST /expert/escalate` (structured case report for future MedHive), `GET /models/health` (loaded models, VRAM, versions, queue depth).
- Rate limiting, upload size/type limits, request IDs in logs, audit events.

### Response JSON (Pydantic-enforced)
```json
{
  "request_id": "str",
  "status": "success | recapture_required | insufficient_data | escalated",
  "crop": "rice | unknown",
  "variety": {"canonical_name": "str|null", "source": "seed_packet_ocr|barcode|farmer_dropdown|expert|visual_guess|unknown", "mapping_confidence": 0.0},
  "visual_result": {"top_3": [{"disease": "str", "probability": 0.0}], "quality_score": 0.0, "lesion_count": 0, "unknown_ood_probability": 0.0},
  "phenotype_observations": {},
  "genomic_result": {"used": false, "confidence": 0.0, "source_type": "variety_level_public_profile", "is_placeholder": false, "disclaimer": "str"},
  "fusion_result": {"disease": "str|unknown", "severity": "low|moderate|high|not_estimated", "confidence_level": "high|moderate|low", "calibrated_probability": 0.0, "uncertainty_score": 0.0, "model_agreement": 0.0, "reasons": ["str"]},
  "recommended_action": {"action_type": "monitor|recapture_images|consult_expert|treatment_advisory_pending_validation", "message": "str"},
  "safety_notice": "str",
  "audit": {"model_versions": {}, "data_source_versions": {}, "timestamp": "ISO-8601", "privacy_actions": ["EXIF stripped", "location reduced to district"], "degraded_modes": []}
}
```
Qwen output schema (strict, JSON only): `crop_guess`, `visual_quality{is_usable, issues[], quality_score}`, `lesion_observations[]{region_id, bbox[4], lesion_color, lesion_shape, lesion_margin, surface_texture, tissue_response(chlorotic|necrotic|water_soaked|powdery|wilted|unknown), estimated_region_affected_percent, distribution(isolated|scattered|clustered|streaked|diffuse|unknown), possible_cause_family(fungal|bacterial|viral|nutrient_deficiency|pest_damage|unknown), visible_symptoms[], observation_confidence}`, `whole_leaf_observations{overall_affected_percent, dominant_symptom_pattern, needs_more_images, requested_images[]}`, `limitations[]`.

### Database (SQLAlchemy 2.0, SQLite now / PostgreSQL-ready)
Tables: users, farms, crop_activities (stub), prediction_requests, uploaded_images, image_quality_results, florence_detections, qwen_phenotypes, efficientnet_predictions, variety_aliases, variety_accession_map, genomic_profiles, genomic_feature_vectors, fusion_results, expert_escalations, model_versions, audit_events. Every table: PK, FKs, `created_at`, `updated_at`; privacy flag and `retention_expires_at` where user data exists; `data_source_version` where relevant; JSON columns for payloads. Alembic baseline migration + idempotent seed script.

### Model registry
`model_versions` table + `configs`: name, version, framework, weights path/hash, device, quantisation, license, loaded_at. Responses and audit log reference versions.

### Tests (pytest; all pass with mocks and no GPU)
Unit: quality checks on synthetic images; NMS; Qwen JSON validation + repair path; aggregation; variety fuzzy matching and confidence bands; genotype lookup incl. placeholder-disabled path; fusion weight renormalisation, centring, visual-guard, missing-modality cases; routing table (one test per rule); janitor expiry. Integration: mocked end-to-end `/predict` for each status (success, recapture_required, insufficient_data, escalated); degraded-mode cases (Qwen invalid JSON, Florence none, genomic missing). `@pytest.mark.gpu` smoke tests for real adapters, auto-skipped without CUDA.

### README
Windows setup: Python 3.11 venv, NVIDIA driver check (`nvidia-smi`), PyTorch CUDA wheel command copied from pytorch.org, bitsandbytes on Windows, `.env` from `.env.example`, `scripts/download_models.py`, run server, run tests (CPU and GPU), troubleshooting (OOM, Florence remote code / flash_attn, bitsandbytes), latency benchmarking script.

### Definition of done
Mocked end-to-end pipeline + tests green on CPU; real adapters load/unload within 6 GB (documented VRAM readings); no honesty-rule violations (grep tests for forbidden phrases such as "DNA of this plant"); `.env.example`, requirements split files, README complete.

## END PROMPT

---

## Follow-up spec prompts (one new spec each, after 01 is green)

**02 visual-classifier-training** — Training pipeline for EfficientNet-B0 (ImageNet weights, new 7-way head, AdamW, mixed precision, class-weighted CE or focal loss, early stopping, the listed augmentations), grouped source-aware splits, `scripts/validate_dataset.py`, temperature scaling, OOD calibration, metrics report (accuracy, macro-F1, per-class recall, confusion matrix, ECE, Brier, latency), export checkpoint + registry entry. Untouched field test set kept separate; lab vs field reported separately.

**03 florence-quality-validation** — Evaluate Florence zero-shot lesion boxes vs classical fallback on 50+ hand-annotated images; decide default localiser from measured results; add annotation guideline doc.

**04 qwen-phenotype-hardening** — Measure latency/VRAM on the laptop, tune pixel caps and tokens, evaluate JSON validity rate, test constrained decoding, implement `uncertain_only` policy.

**05 genomics-offline** — Curated variety-susceptibility table builder (literature/extension ratings + marker presence), optional SNP/XGBoost baseline on public accessions with phenotype labels, optional DNABERT-2 embedding path (VCF + reference + alt allele -> context sequences), cached vectors by variety ID. Sample-size check required before any model is trained.

**06 evaluation-ablations** — The ten baselines/ablations, risk-coverage curves, optional split-conformal prediction sets, Cohen's kappa vs expert labels.

**07 react-dashboard** — Upload flow with recapture guidance, variety confirmation step, confidence display with disclaimers, expert-escalation view.
