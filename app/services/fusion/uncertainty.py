"""Uncertainty quantification and model agreement for Tree of Life."""
import numpy as np
from app.schemas.visual import VisualResult
from app.schemas.phenotype import PhenotypePayload
from app.services.fusion.fuse import CLASS_TO_FAMILY


def compute_uncertainty_score(
    entropy: float,
    disagreement_score: float,
    quality_score: float,
    ood_prob: float,
    family_conflict: float = 0.0
) -> float:
    """Compute bounded heuristic uncertainty score in [0.0, 1.0]."""
    # Weighted combination of uncertainty indicators
    score = (
        0.30 * min(1.0, entropy / 2.0) +
        0.25 * min(1.0, disagreement_score) +
        0.20 * max(0.0, 1.0 - quality_score) +
        0.15 * min(1.0, ood_prob) +
        0.10 * min(1.0, family_conflict)
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_model_agreement(
    visual_top_disease: str,
    phenotype: PhenotypePayload
) -> float:
    """Compute agreement between visual classifier disease family and Qwen family."""
    if not phenotype or not phenotype.family_distribution:
        return 0.5  # Neutral when phenotype unavailable

    visual_family = CLASS_TO_FAMILY.get(visual_top_disease)
    if not visual_family:
        return 0.5

    return float(phenotype.family_distribution.get(visual_family, 0.0))
