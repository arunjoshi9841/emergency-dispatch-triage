from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

FINAL_CATEGORY_COLUMN = "Final Problem Category"
PRIORITY_COLUMN = "Priority Level"


def _normalize_text(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_category(value: Any) -> str:
    category = _normalize_text(value, fallback="missing")
    return category if category else "missing"


def _normalize_priority(value: Any) -> str:
    raw = _normalize_text(value, fallback="unknown")
    if raw.lower().startswith("priority"):
        return raw
    if raw.isdigit():
        return f"Priority {raw}"
    return raw


def priority_to_severity(value: Any) -> int:
    """Convert an APD priority value to a 0-3 severity int."""
    text = _normalize_priority(value).lower()
    if text.startswith("priority"):
        parts = text.split()
        if len(parts) >= 2 and parts[1].isdigit():
            n = int(parts[1])
            if 0 <= n <= 3:
                return n
    n = _parse_int(value, default=1)
    if n < 0:
        return 0
    if n > 3:
        return 3
    return n


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _hour_bucket(hour_value: Any) -> str:
    hour = _parse_int(hour_value, default=-1)
    if hour < 0 or hour > 23:
        return "unknown"
    if hour <= 5:
        return "0-5"
    if hour <= 11:
        return "6-11"
    if hour <= 17:
        return "12-17"
    return "18-23"


def _injury_bucket(value: Any) -> str:
    count = _parse_int(value, default=0)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    return "2+"


def find_categories(
    csv_path: Path,
    output_path: Path | None = None,
) -> set[str]:
    """Return distinct APD final categories (excluding 'missing') and optionally export them."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"APD CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if FINAL_CATEGORY_COLUMN not in df.columns:
        raise ValueError(f"Column '{FINAL_CATEGORY_COLUMN}' not found in APD dataset.")

    series = df[FINAL_CATEGORY_COLUMN].map(_normalize_category)
    categories = {value for value in series.unique() if value}

    if output_path is not None:
        output_path.write_text("\n".join(sorted(categories)), encoding="utf-8")

    return categories


def find_distribution(
    csv_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Compute category and severity totals from the APD dataset.

    Returns a dict with keys:
        - ``category_totals``: {category: count}
        - ``severity_totals``: {severity_int: count}
        - ``distribution_table``: crosstab DataFrame
        - ``apd_df``: the loaded and normalised DataFrame (for sampling)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"APD CSV file not found: {csv_path}")

    apd_df = pd.read_csv(csv_path)
    apd_df[FINAL_CATEGORY_COLUMN] = apd_df[FINAL_CATEGORY_COLUMN].map(_normalize_category)
    apd_df[PRIORITY_COLUMN] = apd_df[PRIORITY_COLUMN].map(_normalize_priority)

    # Distribution table (crosstab)
    ct = pd.crosstab(
        apd_df[FINAL_CATEGORY_COLUMN],
        apd_df[PRIORITY_COLUMN],
        margins=True,
        margins_name="Total",
    )
    total_row = ct.loc[["Total"]]
    ct_body = ct.drop("Total").sort_values("Total", ascending=False)
    ct_sorted = pd.concat([ct_body, total_row])
    ct_sorted["% of All Calls"] = (
        ct_sorted["Total"] / float(ct_sorted.loc["Total", "Total"]) * 100.0
    ).round(2)

    if output_path is not None:
        ct_sorted.to_csv(output_path)

    # Category totals (excluding 'missing')
    valid_categories = sorted(
        c for c in apd_df[FINAL_CATEGORY_COLUMN].unique() if c.lower() != "missing"
    )
    category_totals = {
        cat: int((apd_df[FINAL_CATEGORY_COLUMN] == cat).sum()) for cat in valid_categories
    }

    # Severity totals
    severity_series = apd_df[PRIORITY_COLUMN].map(priority_to_severity)
    severity_totals = {int(k): int(v) for k, v in severity_series.value_counts().items()}

    return {
        "category_totals": category_totals,
        "severity_totals": severity_totals,
        "distribution_table": ct_sorted,
        "apd_df": apd_df,
    }
