"""Compute class weights and per-task evaluation metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_class_weights(
    records: list[dict],
    key: str,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for a label field.

    Args:
        records: Training split records.
        key: e.g. ``"category_id"`` or ``"severity"``.
        num_classes: Total number of classes.

    Returns:
        Float tensor of shape ``(num_classes,)`` with balanced weights.
    """
    counts = Counter(r[key] for r in records)
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls_id in range(num_classes):
        c = counts.get(cls_id, 0)
        weights[cls_id] = total / (num_classes * c) if c > 0 else 1.0
    return weights


def compute_metrics_category(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "cat_accuracy": accuracy_score(labels, preds),
        "cat_f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "cat_f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "cat_precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "cat_recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def compute_metrics_severity(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "sev_accuracy": accuracy_score(labels, preds),
        "sev_f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "sev_f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def compute_metrics_dispatch(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Per-label binary metrics for the 3 dispatch outputs."""
    results: dict[str, float] = {}
    label_names = ["police", "emt", "fire"]
    for i, name in enumerate(label_names):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        results[f"disp_{name}_accuracy"] = accuracy_score(y_true, y_pred)
        results[f"disp_{name}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        results[f"disp_{name}_precision"] = precision_score(y_true, y_pred, zero_division=0)
        results[f"disp_{name}_recall"] = recall_score(y_true, y_pred, zero_division=0)
    return results


def compute_dispatch_pos_weight(records: list[dict]) -> torch.Tensor:
    """Compute pos_weight for each dispatch label (neg_count / pos_count).

    Used by BCEWithLogitsLoss to up-weight rare positive labels. Labels that
    are already majority-positive keep weight 1.0 instead of being downweighted.
    """
    n = len(records)
    keys = ["dispatch_police", "dispatch_emt", "dispatch_fire"]
    weights = []
    for key in keys:
        pos = sum(1 for r in records if r[key])
        neg = n - pos
        weights.append(max(1.0, neg / max(pos, 1)))
    return torch.tensor(weights, dtype=torch.float)
