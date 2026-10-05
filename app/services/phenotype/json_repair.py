"""JSON repair and validation utility for Qwen2.5-VL outputs."""
import json
import re
from typing import Optional
from app.schemas.phenotype import PhenotypePayload


def repair_and_parse_phenotype_json(raw_text: str) -> Optional[PhenotypePayload]:
    """Strip markdown codeblocks and repair simple json syntax issues."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    # Strip markdown block
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # Basic regex repair for trailing commas
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    try:
        data = json.loads(cleaned)
        return PhenotypePayload.model_validate(data)
    except Exception:
        # One repair retry: find first { and last }
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                sub = cleaned[start:end+1]
                data = json.loads(sub)
                return PhenotypePayload.model_validate(data)
        except Exception:
            return None

    return None
