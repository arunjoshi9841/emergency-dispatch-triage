"""Cascaded multi-task DeBERTa model: shared encoder -> category -> severity -> dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from config.config import (
    MODEL_NAME,
    NUM_SEVERITY,
    NUM_DISPATCH,
    CAT_EMB_DIM,
    SEV_EMB_DIM,
    MLP_HIDDEN,
    LOSS_WEIGHT_CATEGORY,
    LOSS_WEIGHT_SEVERITY,
    LOSS_WEIGHT_DISPATCH,
    CATEGORY_LABEL_SMOOTHING,
    SEVERITY_LABEL_SMOOTHING,
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


class DispatchMultiTaskModel(nn.Module):
    """Cascaded DeBERTa model.

    Pipeline:
        1. transcript -> encoder -> [CLS] -> category MLP
        2. [CLS] + category_embedding -> severity MLP
        3. [CLS] + category_embedding + severity_embedding -> dispatch MLP

    During training, ground-truth labels feed the embeddings (teacher forcing).
    During inference, downstream heads receive probability-weighted embeddings
    instead of hard argmax labels to reduce cascade error amplification.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_categories: int = 38,
        num_severity: int = NUM_SEVERITY,
        num_dispatch: int = NUM_DISPATCH,
        cat_emb_dim: int = CAT_EMB_DIM,
        sev_emb_dim: int = SEV_EMB_DIM,
        mlp_hidden: int = MLP_HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Stage 1: category head (CLS only)
        self.category_head = _MLPHead(hidden, mlp_hidden, num_categories, dropout)

        # Learned embeddings for cascade
        self.cat_embedding = nn.Embedding(num_categories, cat_emb_dim)
        self.sev_embedding = nn.Embedding(num_severity, sev_emb_dim)

        # Stage 2: severity head (CLS + category embedding)
        self.severity_head = _MLPHead(hidden + cat_emb_dim, mlp_hidden, num_severity, dropout)

        # Stage 3: dispatch head (CLS + category embedding + severity embedding)
        self.dispatch_head = _MLPHead(hidden + cat_emb_dim + sev_emb_dim, mlp_hidden, num_dispatch, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
        severity_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            category_labels: Ground-truth category ids for teacher forcing (training).
            severity_labels: Ground-truth severity ids for teacher forcing (training).
            If None, uses argmax of predictions (inference).
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        # Stage 1: category
        category_logits = self.category_head(cls_output)

        # Cascade: category embedding
        if category_labels is not None:
            cat_emb = self.cat_embedding(category_labels)
        else:
            cat_probs = category_logits.softmax(dim=-1)
            cat_emb = cat_probs @ self.cat_embedding.weight

        # Stage 2: severity
        sev_input = torch.cat([cls_output, cat_emb], dim=-1)
        severity_logits = self.severity_head(sev_input)

        # Cascade: severity embedding
        if severity_labels is not None:
            sev_emb = self.sev_embedding(severity_labels)
        else:
            sev_probs = severity_logits.softmax(dim=-1)
            sev_emb = sev_probs @ self.sev_embedding.weight

        # Stage 3: dispatch
        disp_input = torch.cat([cls_output, cat_emb, sev_emb], dim=-1)
        dispatch_logits = self.dispatch_head(disp_input)

        return {
            "category_logits": category_logits,
            "severity_logits": severity_logits,
            "dispatch_logits": dispatch_logits,
        }


def compute_loss(
    logits: dict[str, torch.Tensor],
    category_labels: torch.Tensor,
    severity_labels: torch.Tensor,
    dispatch_labels: torch.Tensor,
    category_weights: torch.Tensor | None = None,
    severity_weights: torch.Tensor | None = None,
    dispatch_pos_weight: torch.Tensor | None = None,
    w_cat: float = LOSS_WEIGHT_CATEGORY,
    w_sev: float = LOSS_WEIGHT_SEVERITY,
    w_disp: float = LOSS_WEIGHT_DISPATCH,
    category_label_smoothing: float = CATEGORY_LABEL_SMOOTHING,
    severity_label_smoothing: float = SEVERITY_LABEL_SMOOTHING,
) -> dict[str, torch.Tensor]:
    """Compute weighted multi-task loss."""
    cat_loss_fn = nn.CrossEntropyLoss(
        weight=category_weights,
        label_smoothing=category_label_smoothing,
    )
    sev_loss_fn = nn.CrossEntropyLoss(
        weight=severity_weights,
        label_smoothing=severity_label_smoothing,
    )
    disp_loss_fn = nn.BCEWithLogitsLoss(pos_weight=dispatch_pos_weight)

    cat_loss = cat_loss_fn(logits["category_logits"], category_labels)
    sev_loss = sev_loss_fn(logits["severity_logits"], severity_labels)
    disp_loss = disp_loss_fn(logits["dispatch_logits"], dispatch_labels)

    total = w_cat * cat_loss + w_sev * sev_loss + w_disp * disp_loss

    return {
        "loss": total,
        "category_loss": cat_loss,
        "severity_loss": sev_loss,
        "dispatch_loss": disp_loss,
    }
