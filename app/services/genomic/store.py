"""Genomic Susceptibility Store for Tree of Life Phase 1."""
from typing import Optional, Dict
from app.schemas.genomic import GenomicResult, GenomicSourceType, MANDATORY_GENOMIC_DISCLAIMER
from app.core.config import settings, load_yaml
from app.db.mongo import db_manager


class GenomicStore:
    """Retrieves variety-level genomic susceptibility priors."""
    def __init__(self, seed_file: str = "configs/varieties_seed.yaml"):
        # Local fallback cache from YAML
        yaml_data = load_yaml(seed_file).get("varieties", [])
        self._cache: Dict[str, Dict] = {v["canonical_id"]: v for v in yaml_data}

    def get_profile(self, canonical_variety_id: Optional[str]) -> GenomicResult:
        if not canonical_variety_id:
            return GenomicResult(
                used=False,
                confidence=0.0,
                canonical_variety_id=None,
                susceptibility_vector={},
                is_placeholder=True,
                disclaimer=MANDATORY_GENOMIC_DISCLAIMER
            )

        # 1. Try MongoDB lookup
        profile_data = None
        try:
            db = db_manager.get_sync_db()
            profile_data = db.genomic_profiles.find_one({"canonical_variety_id": canonical_variety_id})
        except Exception:
            profile_data = None

        # 2. Fallback to cached seed
        if not profile_data:
            profile_data = self._cache.get(canonical_variety_id)

        if not profile_data:
            return GenomicResult(
                used=False,
                confidence=0.0,
                canonical_variety_id=canonical_variety_id,
                susceptibility_vector={},
                is_placeholder=True,
                disclaimer=MANDATORY_GENOMIC_DISCLAIMER
            )

        is_placeholder = profile_data.get("is_placeholder", True)
        if settings.APP_ENV == "production" and is_placeholder and not settings.GENOMIC_ALLOW_PLACEHOLDER:
            # Placeholders forbidden in production
            return GenomicResult(
                used=False,
                confidence=0.0,
                canonical_variety_id=canonical_variety_id,
                susceptibility_vector={},
                is_placeholder=True,
                disclaimer=MANDATORY_GENOMIC_DISCLAIMER
            )

        susceptibility = profile_data.get("susceptibility", {})

        return GenomicResult(
            used=True,
            confidence=0.90,
            canonical_variety_id=canonical_variety_id,
            susceptibility_vector=susceptibility,
            source_type=GenomicSourceType.VARIETY_LEVEL_PUBLIC_PROFILE,
            source_database=profile_data.get("source_database", "local_curated"),
            source_version=profile_data.get("source_version", "2024.1"),
            is_placeholder=is_placeholder,
            disclaimer=MANDATORY_GENOMIC_DISCLAIMER
        )
