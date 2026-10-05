# TREE OF LIFE Phase 1 Foundation — Plan Revisions

This document contains the approved implementation plan with all corrections applied.

---

## Updated Answers to Section F

**[1] Qwen `uncertain_only` policy:**
Run Qwen if:
- Visual top prob < 0.80, OR
- Full/crop disagreement > threshold (config `DISAGREEMENT_THRESHOLD`, default 0.15), OR
- OOD prob > threshold (config `OOD_THRESHOLD`, default 0.30), OR
- No lesion localisation available

Default policy remains `always`.

**[2] Classical fallback:**
HSV thresholding (option a), but **only inside a green leaf mask**:
1. Extract green vegetation mask (HSV: H 35–85, S > 0.2, V > 0.2)
2. Within mask, find yellow/brown lesion regions (H 10–40, S > 0.3)
3. Morphological ops, contour detection, return bounding boxes
4. Thresholds in `configs/classical_lesion.yaml`
5. Document as "rough heuristic fallback; may produce false positives"

**[3] Conformal prediction sets:**
Reserve `prediction_set: list[str] | None` in `FusionResult` schema; always `null` in Phase 1.

---

## Updated Section D Resolutions

### D.1.1 Qwen family-to-class mapping (revised)
**Each class gets its family''s full probability, then centre** (do not split):
- If Qwen outputs `fungal: 0.6`, all fungal classes (`rice_blast`, `rice_brown_spot`, `rice_leaf_scald`) each get `0.6` before centring.
- Centring formula: `term_c = log(prob_c) - mean_over_diseases(log(prob))` where prob is the family prob assigned to that class.
- **Epsilon clamp before log:** `prob = clip(prob, eps, 1-eps)` where `eps=1e-8` to prevent `log(0)`.

### D.1.6 Medium-confidence variety matches (revised)
**Medium-confidence (0.60–0.85) variety matches ARE used**, with:
- Reduced genomic weight (multiply by `GENOMIC_VARIETY_CONF_PENALTY`, default 0.5)
- `requires_user_confirmation=true` in response
- Only < 0.60 → `variety=unknown`, genomic weight = 0

### D.1.10 Severity estimation (revised)
**Primary source: lesion area / leaf area ratio** from visual detection:
- Compute **union of lesion masks** (merge overlapping boxes, use pixel-wise union)
- Compute **leaf mask area** (green vegetation mask from quality gate, reused)
- `severity_ratio = lesion_union_area / leaf_mask_area`
- < 10% → `low`, 10–40% → `moderate`, > 40% → `high`
- Fallback: Qwen''s `overall_affected_percent` if detection unavailable
- `not_estimated` for `healthy` or `unknown_or_ood` classes
- Label as "approximate" in response field description

---

## Critical Fixes

### Fix A: Orchestrator Order (FINAL)
Correct execution order ensures crops exist before Qwen AND Florence runs OCR in same cycle:

1. Quality gate (save `leaf_mask_area` for severity)
2. **EfficientNet on full images** (early visual signal)
3. **Florence: localisation (lesion boxes) + OCR (seed packet) → unload Florence immediately**
4. Generate crops from boxes
5. **EfficientNet on crops** → aggregate with full
6. **Emit `partial_result`** (quality + visual aggregated result)
7. **Qwen per policy** (if `uncertain_only` triggers or policy `always`) → unload Qwen
8. Variety resolution (using OCR result from step 3)
9. Genomic lookup
10. Environment (optional)
11. Fusion + routing (compute severity from lesion union / leaf mask)
12. Build final response

### Fix B: Async + Blocking I/O
- All service Protocols gain `async def load()` and `async def unload()` methods (added in M4)
- Blocking model inference wrapped in `asyncio.to_thread()` or executor with `max_workers=1` (dedicated model thread)
- Never block the event loop with model calls

### Fix C: M13 EfficientNet Real Adapter
- Use `efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)` for backbone
- Replace final classifier: `model.classifier[1] = nn.Linear(1280, 7)`
- Load trained weights from `data/models/efficientnet_rice_disease.pth`
- **If no trained checkpoint exists: raise `ModelUnavailableError`** (never random init in real adapter)
- Tests may use a tiny fixture checkpoint saved to `tests/fixtures/efficientnet_test_only.pth`, marked test-only

### Fix D: Valid Status Values
**Valid `status` enum:** `success | recapture_required | insufficient_data | escalated`

**`moderate` is a `confidence_level`, not a status.** Fix all test assertions:
- Mock tests must assert exact statuses: `assert response.status == "success"`
- Routing logic: `moderate` confidence → `status="success"` with `confidence_level="moderate"`, unless other factors trigger escalation

### Fix E: Honesty Grep Test (FINAL)

```python
def test_no_forbidden_phrases_in_codebase():
    import re
    from pathlib import Path
    
    # Known allowed texts to strip before searching
    ALLOWED_TEXTS = [
        "This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist.",
        "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant.",
    ]
    
    forbidden_patterns = [
        (r"DNA of this plant", "Never claim DNA from the plant"),
        (r"sequenc(?:ed|ing) (?:this|the) plant", "Never claim sequencing unless lab assay"),
        (r"\bdiagnos(?:is|ed)\b", "Decision support, not diagnosis (outside safety notice)"),
        (r"pesticide.*\d+.*(?:ml|gram|kg)", "No pesticide quantities"),
    ]
    
    search_paths = [Path("app"), Path("configs"), Path("docs")]
    extensions = ["*.py", "*.yaml", "*.yml", "*.md"]
    
    for search_path in search_paths:
        for ext in extensions:
            for file_path in search_path.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception:
                    continue  # Skip binary or unreadable files
                
                # Strip allowed texts
                sanitized = content
                for allowed in ALLOWED_TEXTS:
                    sanitized = sanitized.replace(allowed, "")
                
                # Search for forbidden patterns
                for pattern, reason in forbidden_patterns:
                    matches = list(re.finditer(pattern, sanitized, re.IGNORECASE))
                    assert len(matches) == 0, \
                        f"Found forbidden pattern in {file_path}: {reason}\n" \
                        f"Matches: {[m.group() for m in matches]}"
```

### Fix F: Milestone Reordering
**Move M13 (variety+genomic) and M14 (environment) before GPU adapters (M9-M12):**
- New order: M1-M8 (mock pipeline), **M9 (variety+genomic)**, **M10 (environment)**, M11 (Florence), M12 (Qwen), M13 (EfficientNet), M14 (GPU lock+residency), M15 (smoke tests)

### Fix G: Version Claims (FINAL)
**No hardcoded version numbers in plan or docs until verified:**

**In README.md (after hardware verification):**
```markdown
## Verified Dependencies (on this hardware)
- Python: 3.11.x
- PyTorch: [VERIFIED_VERSION] with CUDA [CUDA_VERSION]
- transformers: [VERIFIED_VERSION] (native Florence-2 support confirmed)
- bitsandbytes: [VERIFIED_VERSION] (Windows wheels confirmed)
- CUDA Driver: [DRIVER_VERSION]

See `requirements.txt` for exact pinned versions after testing.
```

**In Milestone 15:**
- README template with placeholders `[VERIFIED_VERSION]`, `[VRAM_MEASURED]`
- Smoke test logs actual transformers/bitsandbytes versions, VRAM peaks
- Human operator fills placeholders after validation
- Document in `docs/setup-validation.md`: "Tested on [date] with transformers X.Y.Z, bitsandbytes A.B.C, peak VRAM [measured] GB"

### Fix H: Test Sketches
- `quality_score` is `float` (0.0–1.0), not a dict
- Use `min_pixels=256*28*28` consistently (= 200,704 pixels)
- `max_pixels=512*28*28` (= 401,408 pixels)

### Fix I: Janitor Scope (FINAL)

#### Escalation Design
**`expert_escalations` table:**
- Self-contained case-report snapshot (no FK to `prediction_requests`)
- JSON fields: `case_summary`, `visual_artifacts_summary`, `phenotype_observations`, `fusion_result`, `user_notes`
- `images_included: bool` (requires explicit user consent)
- Optional `image_urls` (temporary signed URLs, not file paths)
- `consent_given: bool` (user consented to share data with experts)
- `status: open | under_review | resolved | archived`
- `retention_expires_at` (default: 180 days from creation, configurable `ESCALATION_RETENTION_DAYS`)
- `created_at`, `updated_at`

#### Janitor Deletion Rules
**Janitor deletes:**
- Expired files (uploads, crops, temp artifacts) where `retention_expires_at < now()`
- `prediction_requests` rows where `retention_expires_at < now()` AND no open escalation references them (check before cascade)
- Related cascaded rows: `uploaded_images`, `image_quality_results`, `florence_detections`, `qwen_phenotypes`, `efficientnet_predictions`, `fusion_results`
- `expert_escalations` where `retention_expires_at < now()` AND `status != "open"` (closed/resolved escalations eligible for deletion)

**Janitor NEVER deletes:**
- `expert_escalations` with `status = "open"` (regardless of retention date)
- `variety_*`, `genomic_*`, `model_versions`, `audit_events` tables
- Any `prediction_request` referenced by an open escalation (add check before deletion)

**Privacy compliance:**
- Default retention: prediction requests 24h, escalations 180 days
- `POST /expert/escalate` requires `consent_given` flag; if `images_included=true`, user must explicitly consent
- Escalation includes only summary data by default; raw images deleted after prediction request expires unless copied to escalation with consent

---

## Additional Smaller Fixes

### 1. Phenotype Term Epsilon Clamp
**In `app/services/fusion/fuse.py`:**
```python
def compute_phenotype_term(qwen_output, class_to_family_map, num_classes, eps=1e-8):
    family_probs = qwen_output.family_distribution  # dict: {family: prob}
    term = np.zeros(num_classes)
    for class_idx, class_id in enumerate(class_ids):
        family = class_to_family_map.get(class_id)
        if family and family in family_probs:
            prob = np.clip(family_probs[family], eps, 1.0 - eps)  # Clamp before log
            term[class_idx] = np.log(prob)
    # Centre over disease classes only
    disease_mask = get_disease_mask(class_ids)
    if disease_mask.sum() > 0:
        term[disease_mask] -= term[disease_mask].mean()
    # Scale by mean Qwen confidence
    term *= qwen_output.mean_confidence
    return term
```

### 2. GpuLock in M7
**Milestone 7 (Orchestrator Mocked) now includes:**
- Create `app/pipeline/gpu_lock.py`: simple `asyncio.Lock` wrapper (no actual GPU calls in mocks)
- Orchestrator structure uses `async with gpu_lock: ...` around mocked heavy stages (no-op in mocks)
- Tests: `test_gpu_lock_serializes_mocked()` with two concurrent jobs (lock ordering deterministic)

**Milestone 14 (GPU Lock + Residency) then:**
- Harden GpuLock: add VRAM logging, track which model owns lock, timeout handling
- Implement residency policies (`on_demand` vs `batch_stage`)
- Add OOM detection + CPU fallback

### 3. Threshold Placeholders
**In `configs/fusion_config.yaml`:**
```yaml
# IMPORTANT: These thresholds are initial heuristics.
# Calibrate on validation data before production deployment.
DISAGREEMENT_THRESHOLD: 0.15  # placeholder; calibrate
OOD_THRESHOLD: 0.30  # placeholder; calibrate
VISUAL_UNCERTAIN_THRESHOLD: 0.80  # triggers uncertain_only Qwen policy
```

**In README and design doc:**
> "Thresholds in `thresholds.yaml` are starting points. Calibrate `DISAGREEMENT_THRESHOLD` and `OOD_THRESHOLD` on a held-out validation set before production use. See `ml/eval/calibrate_thresholds.py` (spec-06)."

### 4. Mocked Test Assertions (CORRECTED)
**Update M7 and M8 test sketches with exact assertions:**

```python
@pytest.mark.asyncio
async def test_orchestrator_success_all_mocks(mock_services):
    # Mock: good quality (0.80), visual top prob 0.85, no OOD
    request = PredictRequest(crop="rice", images=[...], district="Mandya")
    response = await predict(request, mock_services, settings)
    assert response.status == "success"
    assert response.fusion_result.confidence_level == "high"
    assert response.genomic_result.used == True
    assert len(response.audit.degraded_modes) == 0

@pytest.mark.asyncio
async def test_orchestrator_qwen_fails_continues(mock_services_qwen_broken):
    # Mock: Qwen raises exception, but visual confidence sufficient
    response = await predict(request, mock_services_qwen_broken, settings)
    assert response.status == "success"  # Visual alone sufficient
    assert "qwen_unavailable" in response.audit.degraded_modes
    assert response.phenotype_observations is None

@pytest.mark.asyncio
async def test_orchestrator_low_quality_recapture(mock_services):
    # Mock: all images blurry, quality score 0.35 < QUALITY_MIN
    request = PredictRequest(images=[blurry1, blurry2, blurry3])
    response = await predict(request, mock_services, settings)
    assert response.status == "recapture_required"
    assert response.visual_result.quality_score < settings.QUALITY_MIN

@pytest.mark.asyncio
async def test_orchestrator_ood_escalated(mock_services_ood):
    # Mock: visual OOD prob 0.65 > OOD_THRESHOLD
    response = await predict(request, mock_services_ood, settings)
    assert response.status == "escalated"
    assert response.recommended_action.action_type == "consult_expert"
```

### 5. Severity Calculation Implementation
**In `app/services/visual/aggregate.py`:**
```python
def compute_severity(detections: List[DetectionBox], leaf_mask_area: float, qwen_percent: float | None) -> str:
    """
    Compute approximate severity from lesion area relative to leaf area.
    
    Args:
        detections: List of lesion bounding boxes
        leaf_mask_area: Total leaf area in pixels (from quality gate)
        qwen_percent: Fallback percentage from Qwen if detections unavailable
    
    Returns:
        Severity level: "low" | "moderate" | "high" | "not_estimated"
    """
    if detections:
        # Create binary mask, set lesion regions to 1, compute union area
        lesion_union_area = compute_union_area(detections)
        ratio = lesion_union_area / leaf_mask_area if leaf_mask_area > 0 else 0
    elif qwen_percent is not None:
        ratio = qwen_percent / 100.0
    else:
        return "not_estimated"
    
    if ratio < 0.10:
        return "low"
    elif ratio < 0.40:
        return "moderate"
    else:
        return "high"
```

---

## Revised 15 Milestones

| # | Milestone | Key Deliverable | Test Proof |
|---|-----------|-----------------|------------|
| 1 | Config + Schemas | Typed settings, all Pydantic models, `prediction_set` reserved field, threshold placeholders with calibration notes | `test_settings_load`, `test_schemas_validate` |
| 2 | Database | SQLAlchemy models, Alembic, seed script, `expert_escalations` table with consent fields | `test_db_migration`, `test_seed_idempotent` |
| 3 | Storage + Janitor | EXIF strip, temp files, retention cleanup (checks open escalations, excludes 180-day escalations) | `test_strip_exif`, `test_janitor_deletes_expired`, `test_janitor_preserves_open_escalations` |
| 4 | Service Interfaces + Mocks | All Protocols with `load()`/`unload()`, deterministic mocks, async executor wrapper | `test_mock_*` (one per service) |
| 5 | Quality Gate | Resolution, blur, exposure, duplicate, leaf coverage, save `leaf_mask_area` for severity | `test_blur_detection`, `test_aggregate_quality_below_min_triggers_recapture` |
| 6 | Fusion + Routing | Pure functions: family→class (full prob, no split), epsilon clamp, visual guard, uncertainty, routing, severity from union/mask | `test_qwen_family_to_class_no_split`, `test_phenotype_term_epsilon_clamp`, `test_visual_guard_override`, `test_severity_from_area_ratio` |
| 7 | Orchestrator (Mocked) | End-to-end pipeline with basic GpuLock, corrected order (EfficientNet full → Florence localisation+OCR → crops → EfficientNet crops → partial → Qwen), exact status assertions, graceful degradation | `test_orchestrator_success_all_mocks`, `test_orchestrator_qwen_fails_continues`, `test_gpu_lock_serializes_mocked` |
| 8 | API + Queue | FastAPI endpoints, bounded queue, rate limiting, `partial_result` support, `POST /expert/escalate` with consent validation | `test_predict_integration_success`, `test_rate_limit`, `test_partial_result_available`, `test_escalate_requires_consent` |
| **🎯 Mock Pipeline Complete** | **Mocked /predict works end-to-end on CPU** | **All tests green without GPU** |
| 9 | Variety Resolution + Genomic Lookup | Fuzzy matching, medium-conf matches used with penalty + `requires_user_confirmation`, placeholder enforcement | `test_fuzzy_match_variety`, `test_medium_conf_variety_used_with_penalty`, `test_genomic_placeholder_disabled_production` |
| 10 | Environment (Optional) | Open-Meteo fetch, disk cache, rule-based risk | `test_weather_fetch`, `test_weather_cache`, `test_risk_rules` |
| 11 | Florence-2 + Classical Fallback | Real localiser with localisation AND OCR in same cycle (unload immediately), classical CV (green mask → HSV lesions) | `test_florence_loads_and_detects`, `test_florence_ocr_same_cycle`, `test_classical_fallback_green_mask` |
| 12 | Qwen2.5-VL + JSON Repair | Real phenotype extractor, `uncertain_only` policy with 4 triggers, validation, repair, epsilon clamp in fusion term | `test_qwen_extracts_valid_json`, `test_qwen_uncertain_only_policy`, `test_json_repair` |
| 13 | EfficientNet + Calibration + OOD | Real visual classifier (IMAGENET1K_V1 weights, ModelUnavailable if no trained checkpoint), temperature scaling, OOD score | `test_efficientnet_classifies`, `test_efficientnet_no_checkpoint_raises`, `test_temperature_scaling` |
| 14 | GPU Lock + Residency + Async Executor | Harden GpuLock (from M7 basic), add VRAM tracking, residency policies, OOM fallback, dedicated worker thread (max_workers=1) | `test_gpu_lock_serializes`, `test_oom_fallback_cpu`, `test_blocking_call_not_on_event_loop` |
| 15 | GPU Smoke + README | All real adapters on hardware, VRAM/latency measured (not guaranteed), Windows setup guide with `[VERIFIED_VERSION]` placeholders, pure-Python honesty grep test (strip allowed texts, scan .py/.yaml/.md) | `test_all_models_fit_in_vram`, `test_end_to_end_real_models_latency`, `test_no_forbidden_phrases_in_codebase` |

---

## Summary of All Changes

### Section F Answers
- Qwen `uncertain_only`: 4 trigger conditions (visual < 0.80 OR disagreement OR OOD OR no lesion)
- Classical fallback: green mask first, then HSV lesion detection inside mask
- Conformal: reserve `prediction_set` field now (always null in Phase 1)

### Section D Fixes
- D.1.1: family prob assigned to each class fully, not split; epsilon clamp before log
- D.1.6: medium-conf variety used with penalty + confirmation flag
- D.1.10: severity from lesion union area / leaf mask area (primary), Qwen fallback

### Critical Fixes Applied
- **Orchestrator order:** EfficientNet full → Florence (localisation + OCR) → crops → EfficientNet crops → partial → Qwen
- **Async:** Protocols have `load()`/`unload()`, blocking calls wrapped in executor (max_workers=1)
- **M13 EfficientNet:** use IMAGENET1K_V1 backbone, raise ModelUnavailable if no trained checkpoint
- **Status values:** `moderate` is confidence_level, not status; fixed all test assertions to exact values
- **Honesty grep:** pure Python pathlib + re, strip allowed texts before search, scan .py/.yaml/.md
- **Milestone order:** variety+genomic (M9), environment (M10) before GPU adapters (M11-M14)
- **Version claims:** `[VERIFIED_VERSION]` placeholders until tested on hardware
- **Test fixes:** quality_score is float, min_pixels=256×28×28 consistently, exact status assertions
- **Janitor:** checks open escalations before deleting requests, 180-day retention for escalations, consent-based image inclusion
- **GpuLock:** basic version in M7, hardened in M14
- **Thresholds:** documented as placeholders requiring calibration

---

## Definition of Done

**Phase 1 Foundation is complete when:**

1. All 15 milestones pass their tests
2. Mocked end-to-end pipeline runs on CPU (M1-M8)
3. Real model adapters load/unload within measured VRAM budget on RTX 4050 6GB (M11-M15)
4. No honesty-rule violations (grep test passes)
5. `.env.example`, split requirements files, README with verified versions complete
6. All degraded-mode paths tested (Qwen fails, genomic unavailable, Florence OOM → classical, etc.)
7. `docs/latency.md` populated with measured values from actual hardware
8. `docs/setup-validation.md` documents verified dependency versions

**Ready for Spec 02 (visual classifier training) after Phase 1 green.**
