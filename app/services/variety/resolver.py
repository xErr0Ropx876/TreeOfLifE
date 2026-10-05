"""Rice variety resolution using RapidFuzz."""
from typing import Optional, List, Dict, Any
from rapidfuzz import fuzz, process
from app.schemas.variety import VarietyResolution, VarietySource
from app.core.config import load_yaml


class VarietyResolver:
    """Resolves rice varieties from seed packet OCR, barcodes, or text dropdown."""
    def __init__(self, seed_file: str = "configs/varieties_seed.yaml"):
        self.varieties_config = load_yaml(seed_file).get("varieties", [])
        self.alias_index: List[Dict[str, Any]] = []
        for v in self.varieties_config:
            canonical_name = v["canonical_name"]
            canonical_id = v["canonical_id"]
            for alias in v.get("aliases", []):
                self.alias_index.append({
                    "alias": alias.lower().strip(),
                    "canonical_id": canonical_id,
                    "canonical_name": canonical_name
                })

    def resolve(
        self,
        variety_text: Optional[str] = None,
        ocr_text: Optional[str] = None,
        barcode: Optional[str] = None
    ) -> VarietyResolution:
        # 1. Barcode check
        if barcode:
            for v in self.varieties_config:
                if barcode.strip() == v.get("canonical_id", ""):
                    return VarietyResolution(
                        canonical_name=v["canonical_name"],
                        canonical_id=v["canonical_id"],
                        matched_alias=barcode,
                        source=VarietySource.BARCODE,
                        mapping_confidence=0.98,
                        requires_user_confirmation=False
                    )

        # 2. Seed Packet OCR check
        query = (ocr_text or "").strip()
        source = VarietySource.SEED_PACKET_OCR
        if not query and variety_text:
            query = variety_text.strip()
            source = VarietySource.FARMER_DROPDOWN

        if not query:
            return VarietyResolution(
                canonical_name=None,
                canonical_id=None,
                matched_alias=None,
                source=VarietySource.UNKNOWN,
                mapping_confidence=0.0,
                requires_user_confirmation=False
            )

        # Fuzzy match against alias index
        all_aliases = [item["alias"] for item in self.alias_index]
        match = process.extractOne(
            query.lower(),
            all_aliases,
            scorer=fuzz.WRatio
        )

        if not match:
            return VarietyResolution(
                canonical_name=None,
                canonical_id=None,
                matched_alias=None,
                source=VarietySource.UNKNOWN,
                mapping_confidence=0.0,
                requires_user_confirmation=False
            )

        matched_alias_str, score, idx = match
        conf = float(score) / 100.0
        entry = self.alias_index[idx]

        if conf >= 0.85:
            return VarietyResolution(
                canonical_name=entry["canonical_name"],
                canonical_id=entry["canonical_id"],
                matched_alias=matched_alias_str,
                source=source,
                mapping_confidence=conf,
                requires_user_confirmation=False
            )
        elif conf >= 0.60:
            return VarietyResolution(
                canonical_name=entry["canonical_name"],
                canonical_id=entry["canonical_id"],
                matched_alias=matched_alias_str,
                source=source,
                mapping_confidence=conf,
                requires_user_confirmation=True
            )
        else:
            return VarietyResolution(
                canonical_name=None,
                canonical_id=None,
                matched_alias=None,
                source=VarietySource.UNKNOWN,
                mapping_confidence=conf,
                requires_user_confirmation=False
            )
