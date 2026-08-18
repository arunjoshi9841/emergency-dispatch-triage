"""PyTorch Dataset for multi-task dispatch classification."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config.config import MODEL_NAME, MAX_LENGTH


class DispatchDataset(Dataset):
    """Tokenised dataset for the multi-task model."""

    def __init__(
        self,
        records: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        encoding = self.tokenizer(
            rec["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "category_labels": torch.tensor(rec["category_id"], dtype=torch.long),
            "severity_labels": torch.tensor(rec["severity"], dtype=torch.long),
            "dispatch_labels": torch.tensor(
                [
                    float(rec["dispatch_police"]),
                    float(rec["dispatch_emt"]),
                    float(rec["dispatch_fire"]),
                ],
                dtype=torch.float,
            ),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


def create_datasets(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    tokenizer: AutoTokenizer | None = None,
    max_length: int = MAX_LENGTH,
) -> tuple[DispatchDataset, DispatchDataset, DispatchDataset]:
    """Create train/val/test DispatchDataset instances."""
    if tokenizer is None:
        # DeBERTa-v3 uses SentencePiece; forcing the slow tokenizer avoids
        # fast-tokenizer conversion fallbacks that can fail in some envs.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    return (
        DispatchDataset(train_records, tokenizer, max_length),
        DispatchDataset(val_records, tokenizer, max_length),
        DispatchDataset(test_records, tokenizer, max_length),
    )
