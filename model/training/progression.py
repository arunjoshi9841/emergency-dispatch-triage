"""Named training helpers for the progression notebook."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config.config import (
    LOSS_WEIGHT_CATEGORY,
    LOSS_WEIGHT_DISPATCH,
    LOSS_WEIGHT_SEVERITY,
    MODEL_NAME,
)
from model.independent_multitask import IndependentHeadsMultiTaskModel
from model.multitask import DispatchMultiTaskModel
from model.multitask_simple import SimpleMultiTaskModel
from training.trainer import train_model


def _save_training_history(results: dict, history_path: Path | None) -> None:
    if history_path is None:
        return

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_history": results["train_history"],
                "val_history": results["val_history"],
                "best_epoch": results["best_epoch"],
                "best_val_score": float(results["best_val_score"]),
                "train_time_seconds": float(results["train_time_seconds"]),
            },
            f,
            indent=2,
        )


def train_simple_isolated_multihead_classifier(
    train_dataset,
    val_dataset,
    device: torch.device,
    num_categories: int,
    checkpoint_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
    loss_weight_category: float = LOSS_WEIGHT_CATEGORY,
    loss_weight_severity: float = LOSS_WEIGHT_SEVERITY,
    loss_weight_dispatch: float = LOSS_WEIGHT_DISPATCH,
) -> dict:
    """Train Model 1: shared encoder with plain linear heads."""
    model = SimpleMultiTaskModel(model_name, num_categories=num_categories)
    results = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        loss_weight_category=loss_weight_category,
        loss_weight_severity=loss_weight_severity,
        loss_weight_dispatch=loss_weight_dispatch,
        checkpoint_dir=checkpoint_dir,
    )
    _save_training_history(results, history_path)
    return results


def train_updated_isolated_multiclass_classifier(
    train_dataset,
    val_dataset,
    device: torch.device,
    num_categories: int,
    category_weights: torch.Tensor,
    severity_weights: torch.Tensor,
    dispatch_pos_weight: torch.Tensor,
    checkpoint_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
    loss_weight_category: float = LOSS_WEIGHT_CATEGORY,
    loss_weight_severity: float = LOSS_WEIGHT_SEVERITY,
    loss_weight_dispatch: float = LOSS_WEIGHT_DISPATCH,
) -> dict:
    """Train Model 2: shared encoder with independent MLP heads and weights."""
    model = IndependentHeadsMultiTaskModel(model_name, num_categories=num_categories)
    results = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        category_weights=category_weights,
        severity_weights=severity_weights,
        dispatch_pos_weight=dispatch_pos_weight,
        loss_weight_category=loss_weight_category,
        loss_weight_severity=loss_weight_severity,
        loss_weight_dispatch=loss_weight_dispatch,
        checkpoint_dir=checkpoint_dir,
    )
    _save_training_history(results, history_path)
    return results


def train_cascaded_multiclass_classifier(
    train_dataset,
    val_dataset,
    device: torch.device,
    num_categories: int,
    category_weights: torch.Tensor,
    severity_weights: torch.Tensor,
    dispatch_pos_weight: torch.Tensor,
    checkpoint_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
    loss_weight_category: float = LOSS_WEIGHT_CATEGORY,
    loss_weight_severity: float = LOSS_WEIGHT_SEVERITY,
    loss_weight_dispatch: float = LOSS_WEIGHT_DISPATCH,
) -> dict:
    """Train Model 3: category -> severity -> dispatch cascade."""
    model = DispatchMultiTaskModel(model_name, num_categories=num_categories)
    results = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        category_weights=category_weights,
        severity_weights=severity_weights,
        dispatch_pos_weight=dispatch_pos_weight,
        loss_weight_category=loss_weight_category,
        loss_weight_severity=loss_weight_severity,
        loss_weight_dispatch=loss_weight_dispatch,
        checkpoint_dir=checkpoint_dir,
    )
    _save_training_history(results, history_path)
    return results
