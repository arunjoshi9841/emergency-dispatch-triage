"""Stratified train / validation / test split."""

from __future__ import annotations

from collections import Counter

from sklearn.model_selection import train_test_split

from config.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED


def stratified_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train / val / test, stratified by category.

    Ratios are normalised so they sum to 1.0.
    Categories with only 1 sample go into the training set.
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Separate singletons (can't stratify on them)
    cat_counts = Counter(r["category"] for r in records)
    singletons = [r for r in records if cat_counts[r["category"]] < 2]
    splittable = [r for r in records if cat_counts[r["category"]] >= 2]

    labels = [r["category"] for r in splittable]

    val_test_ratio = val_ratio + test_ratio
    train_split, val_test_split = train_test_split(
        splittable,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    val_test_labels = [r["category"] for r in val_test_split]
    relative_test = test_ratio / val_test_ratio

    val_split, test_split = train_test_split(
        val_test_split,
        test_size=relative_test,
        stratify=val_test_labels,
        random_state=seed,
    )

    # Add singletons to training set
    train_split = train_split + singletons

    print(f"[splitter] Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    return train_split, val_split, test_split
