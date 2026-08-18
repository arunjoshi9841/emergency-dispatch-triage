"""Visualization helpers: confusion matrices, loss curves, class distribution."""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.config import SEVERITY_LABELS, DISPATCH_LABELS, PLOTS_DIR


def plot_confusion_matrices(
    predictions: dict,
    category_labels: list[str],
    save_dir: Path = PLOTS_DIR,
) -> None:
    """Plot and save confusion matrices for category and severity."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # -- Category confusion matrix --------------------------------------
    cat_preds = predictions["cat_preds"]
    cat_labels = predictions["cat_labels"]
    present_ids = sorted(set(cat_labels.tolist()) | set(cat_preds.tolist()))
    present_names = [category_labels[i] for i in present_ids]

    cm = confusion_matrix(cat_labels, cat_preds, labels=present_ids)
    fig, ax = plt.subplots(figsize=(max(14, len(present_names) * 0.5), max(12, len(present_names) * 0.4)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Category Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix_category.png", dpi=150)
    plt.show()

    # -- Severity confusion matrix --------------------------------------
    sev_preds = predictions["sev_preds"]
    sev_labels = predictions["sev_labels"]
    sev_names = [str(s) for s in SEVERITY_LABELS]

    cm_sev = confusion_matrix(sev_labels, sev_preds, labels=list(range(len(SEVERITY_LABELS))))
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_sev, annot=True, fmt="d", cmap="Oranges",
                xticklabels=sev_names, yticklabels=sev_names, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Severity Confusion Matrix")
    plt.tight_layout()
    fig2.savefig(save_dir / "confusion_matrix_severity.png", dpi=150)
    plt.show()


def plot_loss_curves(
    train_history: list[dict],
    val_history: list[dict],
    save_dir: Path = PLOTS_DIR,
) -> None:
    """Plot training and validation loss curves."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Total loss
    axes[0, 0].plot(epochs, [h["loss"] for h in train_history], label="Train")
    axes[0, 0].plot(epochs, [h["loss"] for h in val_history], label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")

    # Category loss
    axes[0, 1].plot(epochs, [h["category_loss"] for h in train_history], label="Train")
    axes[0, 1].plot(epochs, [h["category_loss"] for h in val_history], label="Val")
    axes[0, 1].set_title("Category Loss")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")

    # Severity loss
    axes[1, 0].plot(epochs, [h["severity_loss"] for h in train_history], label="Train")
    axes[1, 0].plot(epochs, [h["severity_loss"] for h in val_history], label="Val")
    axes[1, 0].set_title("Severity Loss")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")

    # Dispatch loss
    axes[1, 1].plot(epochs, [h["dispatch_loss"] for h in train_history], label="Train")
    axes[1, 1].plot(epochs, [h["dispatch_loss"] for h in val_history], label="Val")
    axes[1, 1].set_title("Dispatch Loss")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(save_dir / "loss_curves.png", dpi=150)
    plt.show()

    # Accuracy curves
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].plot(epochs, [h["cat_accuracy"] for h in train_history], label="Train")
    axes2[0].plot(epochs, [h["cat_accuracy"] for h in val_history], label="Val")
    axes2[0].set_title("Category Accuracy")
    axes2[0].legend()
    axes2[0].set_xlabel("Epoch")

    axes2[1].plot(epochs, [h["sev_accuracy"] for h in train_history], label="Train")
    axes2[1].plot(epochs, [h["sev_accuracy"] for h in val_history], label="Val")
    axes2[1].set_title("Severity Accuracy")
    axes2[1].legend()
    axes2[1].set_xlabel("Epoch")

    plt.tight_layout()
    fig2.savefig(save_dir / "accuracy_curves.png", dpi=150)
    plt.show()


def plot_class_distribution(
    records: list[dict],
    split_name: str = "Full Dataset",
    save_dir: Path = PLOTS_DIR,
) -> None:
    """Plot category and severity distribution bar charts."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Category distribution
    cat_counts = Counter(r["category"] for r in records)
    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
    names, counts = zip(*sorted_cats) if sorted_cats else ([], [])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(range(len(names)), counts, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Category Distribution - {split_name} (n={len(records)})")
    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(save_dir / f"category_distribution_{split_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    # Severity distribution
    sev_counts = Counter(r["severity"] for r in records)
    sev_names = [str(s) for s in sorted(sev_counts.keys())]
    sev_vals = [sev_counts[int(s)] for s in sev_names]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(sev_names, sev_vals, color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"][:len(sev_names)])
    ax2.set_xlabel("Severity Level")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Severity Distribution - {split_name}")
    for bar, val in zip(bars, sev_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha="center", fontsize=9)
    plt.tight_layout()
    fig2.savefig(save_dir / f"severity_distribution_{split_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
