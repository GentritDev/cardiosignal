
"""Evaluation metrics for multi-label ECG classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a NumPy array or PyTorch tensor to NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)

def _safe_nanmean(values: list[float]) -> float:
    """Calculate a mean while ignoring undefined metric values."""
    valid_values = [value for value in values if np.isfinite(value)]

    if not valid_values:
        return float("nan")

    return float(np.mean(valid_values))

def compute_multilabel_metrics(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    class_names: tuple[str, ...] | list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate metrics for multi-label classification.

    Args:
        logits: Raw model outputs with shape (samples, classes).
        targets: Binary ground-truth labels with the same shape.
        class_names: Names of the diagnostic classes.
        threshold: Probability threshold used for predictions.

    Returns:
        Dictionary containing aggregate and per-class metrics.
    """
    logits_array = _to_numpy(logits).astype(np.float64)
    targets_array = _to_numpy(targets)

    if logits_array.ndim != 2:
        raise ValueError("logits must have shape (samples, classes).")

    if targets_array.ndim != 2:
        raise ValueError("targets must have shape (samples, classes).")

    if logits_array.shape != targets_array.shape:
        raise ValueError("logits and targets must have identical shapes.")

    if logits_array.shape[0] == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    if logits_array.shape[1] != len(class_names):
        raise ValueError("Number of class names must match output dimensions.")

    if not np.isfinite(logits_array).all():
        raise ValueError("logits contain non-finite values.")

    if not np.isfinite(targets_array).all():
        raise ValueError("targets contain non-finite values.")

    if not np.isin(targets_array, [0, 1]).all():
        raise ValueError("targets must contain only binary values (0 or 1).")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")

    targets_array = targets_array.astype(np.int32)

    # Convert raw logits into probabilities.
    clipped_logits = np.clip(logits_array, -50.0, 50.0)
    probabilities = 1.0 / (1.0 + np.exp(-clipped_logits))

    predictions = (probabilities >= threshold).astype(np.int32)

    metrics: dict[str, Any] = {
        "f1_macro": float(
            f1_score(
                targets_array,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_micro": float(
            f1_score(
                targets_array,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
    }

    per_class_f1: list[float] = []
    per_class_auroc: list[float] = []
    per_class_auprc: list[float] = []

    for index, class_name in enumerate(class_names):
        class_targets = targets_array[:, index]
        class_predictions = predictions[:, index]
        class_probabilities = probabilities[:, index]

        class_f1 = float(
            f1_score(
                class_targets,
                class_predictions,
                zero_division=0,
            )
        )

        positive_count = int(class_targets.sum())
        negative_count = int(len(class_targets) - positive_count)

        if positive_count > 0 and negative_count > 0:
            class_auroc = float(
                roc_auc_score(class_targets, class_probabilities)
            )
        else:
            class_auroc = float("nan")

        if positive_count > 0:
            class_auprc = float(
                average_precision_score(class_targets, class_probabilities)
            )
        else:
            class_auprc = float("nan")

        metrics[f"f1_{class_name}"] = class_f1
        metrics[f"auroc_{class_name}"] = class_auroc
        metrics[f"auprc_{class_name}"] = class_auprc

        metrics[f"positives_{class_name}"] = positive_count
        metrics[f"negatives_{class_name}"] = negative_count

        per_class_f1.append(class_f1)
        per_class_auroc.append(class_auroc)
        per_class_auprc.append(class_auprc)

    metrics["auroc_macro"] = _safe_nanmean(per_class_auroc)
    metrics["auprc_macro"] = _safe_nanmean(per_class_auprc)

    return metrics