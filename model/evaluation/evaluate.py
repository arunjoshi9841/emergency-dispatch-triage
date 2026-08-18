"""Evaluate model on a dataset split and produce classification reports."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from config.config import BATCH_SIZE, DISPATCH_LABELS, RESULTS_DIR, SEVERITY_LABELS
from model.multitask import DispatchMultiTaskModel


def evaluate_model(
    model: DispatchMultiTaskModel,
    dataset,
    device: torch.device,
    category_labels: list[str],
    batch_size: int = BATCH_SIZE,
    save_dir: Path = RESULTS_DIR,
) -> dict:
    """Run evaluation on category, severity, and dispatch tasks."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_cat_preds, all_cat_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    all_disp_preds, all_disp_labels = [], []
    inference_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)

            all_cat_preds.append(logits["category_logits"].argmax(dim=-1).cpu().numpy())
            all_cat_labels.append(batch["category_labels"].numpy())

            all_sev_preds.append(logits["severity_logits"].argmax(dim=-1).cpu().numpy())
            all_sev_labels.append(batch["severity_labels"].numpy())

            dispatch_probs = logits["dispatch_logits"].sigmoid().cpu().numpy()
            all_disp_preds.append((dispatch_probs > 0.5).astype(int))
            all_disp_labels.append(batch["dispatch_labels"].int().numpy())

    inference_time_seconds = time.perf_counter() - inference_start

    cat_preds = np.concatenate(all_cat_preds)
    cat_labels = np.concatenate(all_cat_labels)
    sev_preds = np.concatenate(all_sev_preds)
    sev_labels = np.concatenate(all_sev_labels)
    disp_preds = np.concatenate(all_disp_preds)
    disp_labels = np.concatenate(all_disp_labels)

    present_ids = sorted(set(cat_labels.tolist()) | set(cat_preds.tolist()))
    present_names = [category_labels[i] for i in present_ids]
    cat_report = classification_report(
        cat_labels,
        cat_preds,
        labels=present_ids,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    cat_report_str = classification_report(
        cat_labels,
        cat_preds,
        labels=present_ids,
        target_names=present_names,
        zero_division=0,
    )

    sev_names = [str(s) for s in SEVERITY_LABELS]
    sev_report = classification_report(
        sev_labels,
        sev_preds,
        labels=list(range(len(SEVERITY_LABELS))),
        target_names=sev_names,
        zero_division=0,
        output_dict=True,
    )
    sev_report_str = classification_report(
        sev_labels,
        sev_preds,
        labels=list(range(len(SEVERITY_LABELS))),
        target_names=sev_names,
        zero_division=0,
    )

    disp_report = {}
    disp_report_str_parts = []
    for i, name in enumerate(DISPATCH_LABELS):
        report_str = classification_report(
            disp_labels[:, i],
            disp_preds[:, i],
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
        )
        disp_report_str_parts.append(f"--- {name} ---\n{report_str}")
        disp_report[name] = classification_report(
            disp_labels[:, i],
            disp_preds[:, i],
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True,
        )
    disp_report_str = "\n".join(disp_report_str_parts)

    print("=" * 60)
    print("CATEGORY CLASSIFICATION REPORT")
    print("=" * 60)
    print(cat_report_str)
    print("\n" + "=" * 60)
    print("SEVERITY CLASSIFICATION REPORT")
    print("=" * 60)
    print(sev_report_str)
    print("\n" + "=" * 60)
    print("DISPATCH CLASSIFICATION REPORT")
    print("=" * 60)
    print(disp_report_str)
    print(f"\nInference time: {inference_time_seconds:.2f} seconds")

    with open(save_dir / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "category": cat_report,
                "severity": sev_report,
                "dispatch": disp_report,
                "inference_time_seconds": inference_time_seconds,
            },
            f,
            indent=2,
        )

    with open(save_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("CATEGORY\n" + cat_report_str + "\n\n")
        f.write("SEVERITY\n" + sev_report_str + "\n\n")
        f.write("DISPATCH\n" + disp_report_str + "\n")

    return {
        "category_report": cat_report,
        "severity_report": sev_report,
        "dispatch_report": disp_report,
        "inference_time_seconds": inference_time_seconds,
        "predictions": {
            "cat_preds": cat_preds,
            "cat_labels": cat_labels,
            "sev_preds": sev_preds,
            "sev_labels": sev_labels,
            "disp_preds": disp_preds,
            "disp_labels": disp_labels,
        },
    }
