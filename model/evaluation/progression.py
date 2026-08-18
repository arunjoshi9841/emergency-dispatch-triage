"""Named checkpoint evaluation helpers for the progression notebook."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config.config import MODEL_NAME
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_confusion_matrices, plot_loss_curves
from model.independent_multitask import IndependentHeadsMultiTaskModel
from model.multitask import DispatchMultiTaskModel
from model.multitask_simple import SimpleMultiTaskModel


def _load_history(history_path: Path | None) -> dict:
    if history_path is None or not history_path.exists():
        return {
            "train_history": [],
            "val_history": [],
            "best_epoch": None,
            "best_val_score": None,
            "train_time_seconds": None,
        }

    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_saved_training_history(history: dict, plots_dir: Path) -> None:
    train_history = history.get("train_history", [])
    val_history = history.get("val_history", [])
    if train_history and val_history:
        plot_loss_curves(train_history, val_history, save_dir=plots_dir)
    else:
        print("No saved training history found; skipping training curves.")


def _load_state_dict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")


def _evaluate_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    test_dataset,
    device: torch.device,
    category_labels: list[str],
    results_dir: Path,
    plots_dir: Path,
    history_path: Path | None,
) -> dict:
    _load_state_dict(model, checkpoint_path, device)
    eval_results = evaluate_model(
        model=model,
        dataset=test_dataset,
        device=device,
        category_labels=category_labels,
        save_dir=results_dir,
    )

    history = _load_history(history_path)
    _plot_saved_training_history(history, plots_dir)
    plot_confusion_matrices(eval_results["predictions"], category_labels, save_dir=plots_dir)

    return {
        "model": model,
        "eval_results": eval_results,
        "history": history,
    }


def evaluate_simple_isolated_multihead_checkpoint(
    checkpoint_path: Path,
    test_dataset,
    device: torch.device,
    category_labels: list[str],
    num_categories: int,
    results_dir: Path,
    plots_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Load and evaluate Model 1: independent linear heads."""
    model = SimpleMultiTaskModel(model_name, num_categories=num_categories)
    return _evaluate_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        test_dataset=test_dataset,
        device=device,
        category_labels=category_labels,
        results_dir=results_dir,
        plots_dir=plots_dir,
        history_path=history_path,
    )


def evaluate_updated_isolated_multiclass_checkpoint(
    checkpoint_path: Path,
    val_dataset,
    test_dataset,
    device: torch.device,
    category_labels: list[str],
    num_categories: int,
    results_dir: Path,
    plots_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Load and evaluate Model 2: independent MLP heads."""
    del val_dataset
    model = IndependentHeadsMultiTaskModel(model_name, num_categories=num_categories)
    return _evaluate_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        test_dataset=test_dataset,
        device=device,
        category_labels=category_labels,
        results_dir=results_dir,
        plots_dir=plots_dir,
        history_path=history_path,
    )


def evaluate_cascaded_multiclass_checkpoint(
    checkpoint_path: Path,
    val_dataset,
    test_dataset,
    device: torch.device,
    category_labels: list[str],
    num_categories: int,
    results_dir: Path,
    plots_dir: Path,
    history_path: Path | None = None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Load and evaluate Model 3: cascaded multitask model."""
    del val_dataset
    model = DispatchMultiTaskModel(model_name, num_categories=num_categories)
    return _evaluate_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        test_dataset=test_dataset,
        device=device,
        category_labels=category_labels,
        results_dir=results_dir,
        plots_dir=plots_dir,
        history_path=history_path,
    )
