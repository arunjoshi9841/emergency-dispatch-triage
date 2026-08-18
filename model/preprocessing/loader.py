"""Load dataset CSV and resolve transcript text."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from config.config import (
    DATASET_CSV,
    TRANSCRIPT_DIRS,
    DISPATCH_LABELS,
    MIN_CATEGORY_SAMPLES,
)

# Regex to strip speaker labels like "CALLER_1:", "DISPATCHER:", "SYSTEM:" etc.
_SPEAKER_RE = re.compile(r"^(?:CALLER(?:_\d+)?|DISPATCHER|SYSTEM)\s*:\s*", re.MULTILINE)


def _strip_speaker_labels(text: str) -> str:
    """Remove speaker label prefixes from transcript lines."""
    return _SPEAKER_RE.sub("", text)


def _resolve_transcript(file_name: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = d / file_name
        if p.is_file():
            return p
    return None


def compute_category_mapping(
    csv_path: Path = DATASET_CSV,
    min_samples: int = MIN_CATEGORY_SAMPLES,
) -> dict:
    """Scan dataset CSV and build the supported training label mapping.

    Categories with fewer than *min_samples* rows are excluded from the
    supervised label space rather than being merged into a catch-all class.
    The category ``Other`` is also excluded.

    Returns dict with keys:
        category_labels      - sorted list of supported label strings
        excluded_categories  - set of labels excluded from training
        category_to_id       - {label: int}
        id_to_category       - {int: label}
        num_categories       - total count
    """
    counts: Counter[str] = Counter()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["category"].strip()] += 1

    excluded_categories: set[str] = set()
    keep: set[str] = set()
    for cat, n in counts.items():
        if not cat or cat == "Other" or n < min_samples:
            excluded_categories.add(cat)
        else:
            keep.add(cat)

    category_labels = sorted(keep)
    category_to_id = {c: i for i, c in enumerate(category_labels)}
    id_to_category = {i: c for i, c in enumerate(category_labels)}

    print(f"[loader] {len(category_labels)} categories "
          f"({len(excluded_categories)} excluded, min_samples={min_samples})")

    return {
        "category_labels": category_labels,
        "excluded_categories": excluded_categories,
        "category_to_id": category_to_id,
        "id_to_category": id_to_category,
        "num_categories": len(category_labels),
    }


def load_dataset(
    csv_path: Path = DATASET_CSV,
    transcript_dirs: list[Path] | None = None,
    category_to_id: dict[str, int] | None = None,
    excluded_categories: set[str] | None = None,
) -> list[dict]:
    """Load the dataset CSV and attach transcript text to each record.

    If *category_to_id* / *excluded_categories* are not provided, they are
    computed automatically via :func:`compute_category_mapping`.

    Returns a list of dicts with keys:
        transcript_file_name, text, severity, category, category_id,
        dispatch_police, dispatch_emt, dispatch_fire
    """
    if transcript_dirs is None:
        transcript_dirs = list(TRANSCRIPT_DIRS)

    if category_to_id is None or excluded_categories is None:
        mapping = compute_category_mapping(csv_path)
        category_to_id = mapping["category_to_id"]
        excluded_categories = mapping["excluded_categories"]

    records: list[dict] = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["transcript_file_name"].strip()
            path = _resolve_transcript(fname, transcript_dirs)
            if path is None:
                skipped += 1
                continue

            text = path.read_text(encoding="utf-8").strip()
            if not text:
                skipped += 1
                continue

            # Strip speaker labels (CALLER_1:, DISPATCHER:, SYSTEM:)
            text = _strip_speaker_labels(text)

            category = row["category"].strip()
            if category in excluded_categories:
                skipped += 1
                continue

            if category not in category_to_id:
                skipped += 1
                continue

            records.append({
                "transcript_file_name": fname,
                "text": text,
                "severity": int(row["severity"]),
                "category": category,
                "category_id": category_to_id[category],
                "dispatch_police": row["hasDispatchedPolice"].strip() == "True",
                "dispatch_emt": row["hasDispatchedEMT"].strip() == "True",
                "dispatch_fire": row["hasDispatchedFire"].strip() == "True",
            })

    if skipped:
        print(f"[loader] Skipped {skipped} rows (missing transcript / invalid category)")
    print(f"[loader] Loaded {len(records)} records")
    return records
