"""Simple shared-encoder multitask model with linear heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from config.config import (
    MODEL_NAME,
    NUM_SEVERITY,
    NUM_DISPATCH,
)


class SimpleMultiTaskModel(nn.Module):
    """Baseline multitask model.

    Uses one shared encoder and three independent linear heads:
      - category (multiclass)
      - severity (multiclass)
      - dispatch (multilabel logits)

    ``category_labels`` and ``severity_labels`` are accepted for compatibility
    with the existing training loop API but are not used.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_categories: int = 38,
        num_severity: int = NUM_SEVERITY,
        num_dispatch: int = NUM_DISPATCH,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden, num_categories)
        self.severity_head = nn.Linear(hidden, num_severity)
        self.dispatch_head = nn.Linear(hidden, num_dispatch)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
        severity_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del category_labels, severity_labels  # Unused; kept for trainer compatibility.

        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        return {
            "category_logits": self.category_head(cls_output),
            "severity_logits": self.severity_head(cls_output),
            "dispatch_logits": self.dispatch_head(cls_output),
        }
