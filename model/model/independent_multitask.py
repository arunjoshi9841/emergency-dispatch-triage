"""Shared-encoder multitask model with three independent task heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from config.config import (
    MODEL_NAME,
    NUM_SEVERITY,
    NUM_DISPATCH,
    MLP_HIDDEN,
)


class _MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IndependentHeadsMultiTaskModel(nn.Module):
    """Shared encoder with category/severity/dispatch heads independent of each other.

    The forward signature accepts ``category_labels`` and ``severity_labels`` for
    compatibility with existing training loops, but this model does not use them.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_categories: int = 38,
        num_severity: int = NUM_SEVERITY,
        num_dispatch: int = NUM_DISPATCH,
        mlp_hidden: int = MLP_HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.category_head = _MLPHead(hidden, mlp_hidden, num_categories, dropout)
        self.severity_head = _MLPHead(hidden, mlp_hidden, num_severity, dropout)
        self.dispatch_head = _MLPHead(hidden, mlp_hidden, num_dispatch, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
        severity_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del category_labels, severity_labels  # Unused; kept for API compatibility.

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
