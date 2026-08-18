"""Training loop for the multi-task dispatch model."""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR,
    LOSS_WEIGHT_CATEGORY,
    LOSS_WEIGHT_SEVERITY,
    LOSS_WEIGHT_DISPATCH,
)
from model.multitask import DispatchMultiTaskModel, compute_loss
from training.metrics import (
    compute_metrics_category,
    compute_metrics_severity,
    compute_metrics_dispatch,
)


def _run_epoch(
    model: DispatchMultiTaskModel,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler=None,
    category_weights: torch.Tensor | None = None,
    severity_weights: torch.Tensor | None = None,
    dispatch_pos_weight: torch.Tensor | None = None,
    loss_weight_category: float = LOSS_WEIGHT_CATEGORY,
    loss_weight_severity: float = LOSS_WEIGHT_SEVERITY,
    loss_weight_dispatch: float = LOSS_WEIGHT_DISPATCH,
) -> dict[str, float]:
    """Run one epoch of training (optimizer is not None) or evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_cat_loss = 0.0
    total_sev_loss = 0.0
    total_disp_loss = 0.0
    n_batches = 0

    all_cat_preds, all_cat_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    all_disp_preds, all_disp_labels = [], []

    cw = category_weights.to(device) if category_weights is not None else None
    sw = severity_weights.to(device) if severity_weights is not None else None
    dw = dispatch_pos_weight.to(device) if dispatch_pos_weight is not None else None

    ctx = nullcontext() if is_train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            cat_labels = batch["category_labels"].to(device)
            sev_labels = batch["severity_labels"].to(device)
            disp_labels = batch["dispatch_labels"].to(device)

            logits = model(
                input_ids, attention_mask, token_type_ids,
                category_labels=cat_labels if is_train else None,
                severity_labels=sev_labels if is_train else None,
            )

            losses = compute_loss(
                logits, cat_labels, sev_labels, disp_labels,
                category_weights=cw, severity_weights=sw,
                dispatch_pos_weight=dw,
                w_cat=loss_weight_category,
                w_sev=loss_weight_severity,
                w_disp=loss_weight_dispatch,
            )

            if is_train:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += losses["loss"].item()
            total_cat_loss += losses["category_loss"].item()
            total_sev_loss += losses["severity_loss"].item()
            total_disp_loss += losses["dispatch_loss"].item()
            n_batches += 1

            all_cat_preds.append(logits["category_logits"].argmax(dim=-1).cpu().numpy())
            all_cat_labels.append(cat_labels.cpu().numpy())
            all_sev_preds.append(logits["severity_logits"].argmax(dim=-1).cpu().numpy())
            all_sev_labels.append(sev_labels.cpu().numpy())
            all_disp_preds.append((logits["dispatch_logits"].sigmoid() > 0.5).int().cpu().numpy())
            all_disp_labels.append(disp_labels.int().cpu().numpy())

    cat_preds = np.concatenate(all_cat_preds)
    cat_labels_np = np.concatenate(all_cat_labels)
    sev_preds = np.concatenate(all_sev_preds)
    sev_labels_np = np.concatenate(all_sev_labels)
    disp_preds = np.concatenate(all_disp_preds)
    disp_labels_np = np.concatenate(all_disp_labels)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "category_loss": total_cat_loss / max(n_batches, 1),
        "severity_loss": total_sev_loss / max(n_batches, 1),
        "dispatch_loss": total_disp_loss / max(n_batches, 1),
    }
    metrics.update(compute_metrics_category(cat_preds, cat_labels_np))
    metrics.update(compute_metrics_severity(sev_preds, sev_labels_np))
    metrics.update(compute_metrics_dispatch(disp_preds, disp_labels_np))
    return metrics


def train_model(
    model: DispatchMultiTaskModel,
    train_dataset,
    val_dataset,
    device: torch.device,
    category_weights: torch.Tensor | None = None,
    severity_weights: torch.Tensor | None = None,
    dispatch_pos_weight: torch.Tensor | None = None,
    loss_weight_category: float = LOSS_WEIGHT_CATEGORY,
    loss_weight_severity: float = LOSS_WEIGHT_SEVERITY,
    loss_weight_dispatch: float = LOSS_WEIGHT_DISPATCH,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    """Full training loop with validation and early stopping.

    Returns:
        Dict with ``best_model_state``, ``train_history``, ``val_history``,
        ``best_epoch``, ``best_val_score``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.to(device)

    best_val_score = 0.0
    best_epoch = -1
    best_model_state = None
    patience_counter = 0
    train_history = []
    val_history = []
    train_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('='*60)

        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            category_weights=category_weights,
            severity_weights=severity_weights,
            dispatch_pos_weight=dispatch_pos_weight,
            loss_weight_category=loss_weight_category,
            loss_weight_severity=loss_weight_severity,
            loss_weight_dispatch=loss_weight_dispatch,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            category_weights=category_weights,
            severity_weights=severity_weights,
            dispatch_pos_weight=dispatch_pos_weight,
            loss_weight_category=loss_weight_category,
            loss_weight_severity=loss_weight_severity,
            loss_weight_dispatch=loss_weight_dispatch,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(f"  Train - loss: {train_metrics['loss']:.4f} | "
              f"cat_f1_macro: {train_metrics['cat_f1_macro']:.4f} | "
              f"sev_f1_macro: {train_metrics['sev_f1_macro']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f} | "
              f"cat_f1_macro: {val_metrics['cat_f1_macro']:.4f} | "
              f"sev_f1_macro: {val_metrics['sev_f1_macro']:.4f}")
        print(f"  Disp  - police_f1: {val_metrics['disp_police_f1']:.4f} | "
              f"emt_f1: {val_metrics['disp_emt_f1']:.4f} | "
              f"fire_f1: {val_metrics['disp_fire_f1']:.4f}")

        # Composite score: average macro F1 across all three tasks
        disp_avg_f1 = np.mean([val_metrics[f"disp_{n}_f1"] for n in ["police", "emt", "fire"]])
        val_score = (val_metrics["cat_f1_macro"] + val_metrics["sev_f1_macro"] + disp_avg_f1) / 3
        print(f"  Composite F1: {val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save(best_model_state, ckpt_path)
            print(f"  New best model saved (composite_f1={best_val_score:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time_seconds = time.perf_counter() - train_start
    print(f"Training time: {train_time_seconds / 60:.2f} minutes")

    return {
        "best_model_state": best_model_state,
        "train_history": train_history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "train_time_seconds": train_time_seconds,
    }
