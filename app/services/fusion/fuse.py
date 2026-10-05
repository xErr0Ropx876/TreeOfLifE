"""Mathematical evidence fusion for Tree of Life Phase 1."""
from typing import Dict, List, Tuple
import numpy as np
from app.schemas.fusion import FusionResult, SeverityLevel, ConfidenceLevel
from app.schemas.visual import VisualResult
from app.schemas.phenotype import PhenotypePayload
from app.schemas.genomic import GenomicResult
from app.schemas.environment import EnvironmentResult

DISEASE_CLASSES = [
    "rice_blast",
    "rice_brown_spot",
    "bacterial_leaf_blight",
    "rice_leaf_scald",
    "rice_tungro",
]

ALL_CLASSES = [
    "healthy",
    "rice_blast",
    "rice_brown_spot",
    "bacterial_leaf_blight",
    "rice_leaf_scald",
    "rice_tungro",
    "unknown_or_ood",
]

CLASS_TO_FAMILY = {
    "rice_blast": "fungal",
    "rice_brown_spot": "fungal",
    "bacterial_leaf_blight": "bacterial",
    "rice_leaf_scald": "fungal",
    "rice_tungro": "viral",
}


def logit(p: float, eps: float = 1e-8) -> float:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(np.log(p_clipped / (1.0 - p_clipped)))


def compute_genomic_terms(
    genomic_result: GenomicResult,
    eps: float = 1e-8
) -> np.ndarray:
    terms = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    if not genomic_result.used or not genomic_result.susceptibility_vector:
        return terms

    disease_logits = []
    for cls in DISEASE_CLASSES:
        s = genomic_result.susceptibility_vector.get(cls, 0.5)
        disease_logits.append(logit(s, eps=eps))

    mean_logit = np.mean(disease_logits)
    for idx, cls in enumerate(ALL_CLASSES):
        if cls in DISEASE_CLASSES:
            s = genomic_result.susceptibility_vector.get(cls, 0.5)
            terms[idx] = logit(s, eps=eps) - mean_logit
        else:
            terms[idx] = 0.0

    return terms


def compute_phenotype_terms(
    phenotype: PhenotypePayload,
    eps: float = 1e-8
) -> np.ndarray:
    terms = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    if not phenotype or not phenotype.family_distribution:
        return terms

    family_probs = phenotype.family_distribution
    disease_logs = []
    for cls in DISEASE_CLASSES:
        fam = CLASS_TO_FAMILY.get(cls)
        p = family_probs.get(fam, 0.1) if fam else 0.1
        p_clipped = np.clip(p, eps, 1.0 - eps)
        disease_logs.append(np.log(p_clipped))

    mean_log = np.mean(disease_logs)
    for idx, cls in enumerate(ALL_CLASSES):
        if cls in DISEASE_CLASSES:
            fam = CLASS_TO_FAMILY.get(cls)
            p = family_probs.get(fam, 0.1) if fam else 0.1
            p_clipped = np.clip(p, eps, 1.0 - eps)
            terms[idx] = (np.log(p_clipped) - mean_log) * phenotype.mean_observation_confidence
        else:
            terms[idx] = 0.0

    return terms


def compute_environment_terms(
    env: EnvironmentResult,
    eps: float = 1e-8
) -> np.ndarray:
    terms = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    if not env or not env.used or not env.risk_vector:
        return terms

    disease_risks = []
    for cls in DISEASE_CLASSES:
        r = env.risk_vector.get(cls, 0.5)
        disease_risks.append(logit(r, eps=eps))

    mean_risk = np.mean(disease_risks)
    for idx, cls in enumerate(ALL_CLASSES):
        if cls in DISEASE_CLASSES:
            r = env.risk_vector.get(cls, 0.5)
            terms[idx] = logit(r, eps=eps) - mean_risk
        else:
            terms[idx] = 0.0

    return terms


def fuse_evidence(
    visual: VisualResult,
    genomic: GenomicResult,
    phenotype: Optional[PhenotypePayload],
    environment: Optional[EnvironmentResult],
    variety_confidence: float = 1.0,
    base_weights: Dict[str, float] = None,
    eps: float = 1e-8
) -> Tuple[str, float, Dict[str, float], List[str]]:
    """Execute mathematical evidence fusion across modalities."""
    if base_weights is None:
        base_weights = {
            "visual": 0.70,
            "genomic": 0.15,
            "phenotype": 0.10,
            "environment": 0.05,
        }

    reasons: List[str] = []
    active_weights: Dict[str, float] = {}

    # 1. Visual term
    active_weights["visual"] = base_weights["visual"]
    visual_probs = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    for p in visual.top_3:
        if p.disease in ALL_CLASSES:
            idx = ALL_CLASSES.index(p.disease)
            visual_probs[idx] = p.probability
    
    # Fill remaining uniformly if sum < 1.0
    rem = max(0.0, 1.0 - np.sum(visual_probs))
    unassigned = [i for i, v in enumerate(visual_probs) if v == 0.0]
    if unassigned:
        visual_probs[unassigned] = rem / len(unassigned)

    visual_terms = np.array([logit(p, eps=eps) for p in visual_probs])

    # 2. Genomic term
    if genomic and genomic.used:
        conf_factor = 1.0 if variety_confidence >= 0.85 else (0.50 if variety_confidence >= 0.60 else 0.0)
        active_weights["genomic"] = base_weights["genomic"] * conf_factor
        genomic_terms = compute_genomic_terms(genomic, eps=eps)
        if conf_factor < 1.0 and conf_factor > 0.0:
            reasons.append("genomic_weight_reduced_due_to_medium_variety_confidence")
    else:
        active_weights["genomic"] = 0.0
        genomic_terms = np.zeros(len(ALL_CLASSES))

    # 3. Phenotype term
    if phenotype and phenotype.family_distribution:
        active_weights["phenotype"] = base_weights["phenotype"]
        phenotype_terms = compute_phenotype_terms(phenotype, eps=eps)
    else:
        active_weights["phenotype"] = 0.0
        phenotype_terms = np.zeros(len(ALL_CLASSES))

    # 4. Environment term
    if environment and environment.used:
        active_weights["environment"] = base_weights["environment"]
        env_terms = compute_environment_terms(environment, eps=eps)
    else:
        active_weights["environment"] = 0.0
        env_terms = np.zeros(len(ALL_CLASSES))

    # Renormalise active weights
    total_w = sum(active_weights.values())
    if total_w > 0:
        eff_weights = {k: v / total_w for k, v in active_weights.items()}
    else:
        eff_weights = {"visual": 1.0, "genomic": 0.0, "phenotype": 0.0, "environment": 0.0}

    # Linear fusion of terms
    fused_logits = (
        eff_weights["visual"] * visual_terms +
        eff_weights.get("genomic", 0.0) * genomic_terms +
        eff_weights.get("phenotype", 0.0) * phenotype_terms +
        eff_weights.get("environment", 0.0) * env_terms
    )

    # Softmax
    exp_logits = np.exp(fused_logits - np.max(fused_logits))
    fused_probs = exp_logits / np.sum(exp_logits)

    top_idx = int(np.argmax(fused_probs))
    top_disease = ALL_CLASSES[top_idx]
    top_prob = float(fused_probs[top_idx])

    # Visual Guard rule: if visual top prob >= 0.95 and quality >= 0.70, cannot be flipped
    visual_top_idx = int(np.argmax(visual_probs))
    visual_top_prob = float(visual_probs[visual_top_idx])
    visual_top_disease = ALL_CLASSES[visual_top_idx]

    if visual_top_prob >= 0.95 and visual.quality_score >= 0.70:
        if top_disease != visual_top_disease:
            top_disease = visual_top_disease
            top_prob = visual_top_prob
            reasons.append("visual_guard_override_applied")

    return top_disease, top_prob, eff_weights, reasons
