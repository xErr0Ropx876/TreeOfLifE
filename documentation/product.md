---
inclusion: always
---

# TREE OF LIFE — Product Steering

## What this is
A local, open-source crop-disease **decision-support** backend for Indian smallholder farmers.
Phase 1: crop = rice, region = Karnataka, no paid APIs, runs on one Windows laptop (see tech.md).
Output is decision support, never a laboratory diagnosis.

## Evidence branches
1. **Phenotype** (strongest evidence of *active* disease): leaf photos -> quality gate -> lesion localisation -> EfficientNet-B0 classifier, plus Qwen2.5-VL visible-attribute extraction.
2. **Genotype (variety-level prior)**: known variety -> locally cached curated/public profile -> susceptibility prior. NOT DNA from the farmer's plant.
3. **Environment (optional, weak)**: district-level weather -> rule-based risk vector.

## Non-negotiable honesty rules
- NEVER claim DNA was obtained or inferred from an image. NEVER claim a plant was sequenced unless a laboratory genotype file exists (reserved source type `lab_allele_assay`, not implemented in Phase 1).
- Every genomic result carries the label: "Variety-level genomic susceptibility estimate." and the disclaimer: "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant."
- Seed/fixture genomic values are PLACEHOLDERS. They must carry `is_placeholder=true`, be surfaced in the API response, and be disabled when `APP_ENV=production` unless explicitly allowed.
- Florence-2 and Qwen2.5-VL are NEVER final disease classifiers. Qwen may emit only a disease-*family* hint (fungal / bacterial / viral / nutrient_deficiency / pest_damage / unknown).
- Genotype and environment can lower confidence or shift ranking slightly; they never override a very-high-confidence visual prediction (visual top prob >= 0.95 with quality >= 0.70 cannot be flipped by non-visual evidence).
- No external-model failure may crash the visual diagnosis. Every optional branch degrades gracefully and the response says what was unavailable.
- Do not state accuracy numbers anywhere in code, docs or responses. Targets are labelled "target"; measured numbers come only from `ml/eval` reports.

## Treatment safety
- No pesticide brand names. No pesticide quantities unless an agronomist-validated rule exists (none in Phase 1).
- Low confidence -> recapture images or consult expert. Prediction and treatment advice are separate fields; Phase 1 advice is `monitor | recapture_images | consult_expert | treatment_advisory_pending_validation`.
- Always include the safety notice: "This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist."

## Privacy
- Strip EXIF (including GPS) on ingest. Store location at district/taluk level only.
- Temporary images expire (default 24 h) and are deleted by a janitor job; every record has `retention_expires_at`.
- Design for India's DPDP Act principles: consent flag, purpose limitation, data minimisation, deletion on request.

## Scope tiers
- **Required MVP:** quality gate, EfficientNet classifier with calibration + OOD, Florence localiser (with classical fallback), Qwen phenotype JSON, variety resolver, curated genotype lookup, fusion + routing, FastAPI, SQLite, tests.
- **Optional:** rule-based environment branch, barcode/QR, conformal prediction sets, React dashboard.
- **Later research:** DNABERT-2 embeddings, learned fusion MLP, lab allele assays, new crops/diseases beyond the registry.

## Extensibility
Crops, diseases and varieties live in config/DB registries (`configs/classes.yaml`, `variety_*` tables). Never hard-code class lists in business logic.
