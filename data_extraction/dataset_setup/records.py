from __future__ import annotations

import csv
from pathlib import Path

DATASET_COLUMNS = [
    "transcript_file_name",
    "severity",
    "category",
    "hasDispatchedPolice",
    "hasDispatchedEMT",
    "hasDispatchedFire",
    "apdId",
    "audioRecordingId",
]


def _ensure_csv_header(dataset_path: Path) -> None:
    """Create the CSV with headers if it doesn't exist yet."""
    if dataset_path.is_file() and dataset_path.stat().st_size > 0:
        return
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()


def add_record(
    dataset_path: Path,
    *,
    transcript_file: str,
    severity: int,
    category: str,
    dispatch_police: bool,
    dispatch_emt: bool,
    dispatch_fire: bool,
    apd_id: str | None = None,
    audio_recording_id: str | None = None,
) -> None:
    """Append one record to the dataset CSV.

    Exactly one of ``apd_id`` (synthetic/APD source) or
    ``audio_recording_id`` (real 911 recording source) should be provided.
    Deduplicates by ``transcript_file_name`` - skips if already present.
    """
    _ensure_csv_header(dataset_path)

    # Check for duplicate transcript_file_name
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("transcript_file_name") == transcript_file:
                return  # already recorded

    row = {
        "transcript_file_name": transcript_file,
        "severity": severity,
        "category": category,
        "hasDispatchedPolice": dispatch_police,
        "hasDispatchedEMT": dispatch_emt,
        "hasDispatchedFire": dispatch_fire,
        "apdId": apd_id or "",
        "audioRecordingId": audio_recording_id or "",
    }

    with open(dataset_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writerow(row)
