"""Sampling utilities for APD-grounded synthetic data generation.

Core flow:
1. ``compute_generation_plan()`` - proportional allocation with a floor,
   minus what already exists, capped by available unique APD records.
2. ``sample_category()`` - draw *n* unique APD rows for one category.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from apd_distribution_analysis.analysis import (
    FINAL_CATEGORY_COLUMN,
    _normalize_category,
)

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def compute_generation_plan(
    apd_df: pd.DataFrame,
    dataset_csv: Path,
    *,
    ceiling: int = 4000,
    min_per_category: int = 100,
) -> dict[str, dict[str, int]]:
    """Compute per-category generation targets and deficits.

    Steps:
        1. Count APD records per category (excl. "missing").
        2. Proportional allocation: ``round(cat_count / total * ceiling)``.
        3. Apply floor: ``max(proportional, min(min_per_category, apd_count))``.
        4. Subtract existing dataset counts -> deficit.
        5. Cap deficit by available (unused) unique APD Incident Numbers.

    Returns a dict keyed by category with sub-keys:
        ``apd_total``, ``proportional``, ``target``, ``existing``,
        ``available``, ``deficit``.
    """
    # --- APD counts & unique IDs per category ---
    apd_counts: dict[str, int] = {}
    apd_ids: dict[str, set[str]] = {}
    for _, row in apd_df.iterrows():
        cat = _normalize_category(row.get(FINAL_CATEGORY_COLUMN))
        if cat.lower() == "missing":
            continue
        apd_counts[cat] = apd_counts.get(cat, 0) + 1
        inc = str(row.get("Incident Number", "")).strip()
        if inc:
            apd_ids.setdefault(cat, set()).add(inc)

    apd_total = sum(apd_counts.values())
    if apd_total == 0:
        return {}

    # --- Existing dataset counts & used APD IDs ---
    existing_counts: dict[str, int] = {}
    used_apd_ids: set[str] = set()
    if dataset_csv.is_file() and dataset_csv.stat().st_size > 0:
        with open(dataset_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = row.get("category", "").strip()
                if cat:
                    existing_counts[cat] = existing_counts.get(cat, 0) + 1
                aid = row.get("apdId", "").strip()
                if aid:
                    used_apd_ids.add(aid)

    # --- Build plan ---
    plan: dict[str, dict[str, int]] = {}
    for cat in sorted(apd_counts):
        count = apd_counts[cat]
        proportional = round(count / apd_total * ceiling)
        effective_floor = min(min_per_category, count)
        target = max(proportional, effective_floor)
        existing = existing_counts.get(cat, 0)
        available = len(apd_ids.get(cat, set()) - used_apd_ids)
        deficit = max(0, target - existing)
        deficit = min(deficit, available)
        plan[cat] = {
            "apd_total": count,
            "proportional": proportional,
            "target": target,
            "existing": existing,
            "available": available,
            "deficit": deficit,
        }
    return plan


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def sample_category(
    apd_df: pd.DataFrame,
    category: str,
    n: int,
    *,
    exclude_ids: set[str],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample *n* unique APD rows for *category*, excluding already-used IDs.

    Returns a list of row dicts (never more than *n*, may be fewer if the
    pool is smaller).  No duplicate Incident Numbers will appear.
    """
    pool = apd_df[apd_df[FINAL_CATEGORY_COLUMN] == category].copy()

    # Filter out already-used Incident Numbers
    if "Incident Number" in pool.columns:
        pool = pool[~pool["Incident Number"].astype(str).isin(exclude_ids)]

    if pool.empty or n <= 0:
        return []

    actual_n = min(n, len(pool))
    sampled = pool.sample(n=actual_n, random_state=seed, replace=False)
    return sampled.to_dict(orient="records")
