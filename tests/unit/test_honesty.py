"""Honesty Grep Test - Non-negotiable ethical claims verification."""
import re
from pathlib import Path

ALLOWED_TEXTS = [
    "This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist.",
    "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant.",
]

FORBIDDEN_PATTERNS = [
    (r"DNA of this plant", "Never claim DNA from the plant"),
    (r"sequenc(?:ed|ing) (?:this|the) plant", "Never claim sequencing unless lab assay"),
    (r"pesticide.*\d+.*(?:ml|gram|kg)", "No pesticide quantities"),
]


def test_no_forbidden_phrases_in_codebase():
    search_paths = [Path("app"), Path("configs"), Path("src")]
    extensions = ["*.py", "*.yaml", "*.yml", "*.ts", "*.tsx"]

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for ext in extensions:
            for file_path in search_path.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception:
                    continue

                sanitized = content
                for allowed in ALLOWED_TEXTS:
                    sanitized = sanitized.replace(allowed, "")

                for pattern, reason in FORBIDDEN_PATTERNS:
                    matches = list(re.finditer(pattern, sanitized, re.IGNORECASE))
                    assert len(matches) == 0, (
                        f"Found forbidden pattern in {file_path}: {reason}\n"
                        f"Matches: {[m.group() for m in matches]}"
                    )


if __name__ == "__main__":
    test_no_forbidden_phrases_in_codebase()
    print("test_no_forbidden_phrases_in_codebase PASSED!")
