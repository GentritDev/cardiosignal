
"""Training and evaluation engine for CardioSignal."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cardiosignal.training.metrics import compute_multilabel_metrics


def _validate_engine_arguments(
    class_names: tuple[str, ...] | list[str],
    threshold: float,
    max_batches: int | None,
    gradient_clip_norm: float | None,
) -> None:
    """Validate shared training and evaluation arguments."""
    if not class_names:
        raise ValueError("class_names cannot be empty.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1.")

    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...] | list[str],
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None

    if len(dataloader) == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")

    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        if "signal" not in batch or "labels" not in batch:
            raise KeyError("Each batch must contain 'signal' and 'labels'.")

        signals = batch["signal"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = batch["labels"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if signals.ndim != 3:
            raise ValueError(
                f"Expected signals with 3 dimensions, got {signals.ndim}."
            )

        if targets.ndim != 2:
            raise ValueError(
                f"Expected targets with 2 dimensions, got {targets.ndim}."
            )

        if signals.shape[0] != targets.shape[0]:
            raise ValueError("Signals and targets have different batch sizes.")

        if targets.shape[1] != len(class_names):
            raise ValueError(
                "Target dimensions do not match the number of class names."
            )

        if not torch.isfinite(signals).all():
            raise ValueError("Input signals contain non-finite values.")

        if not torch.isfinite(targets).all():
            raise ValueError("Targets contain non-finite values.")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(signals)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"Model output shape {tuple(logits.shape)} does not "
                    f"match target shape {tuple(targets.shape)}."
                )

            loss = loss_fn(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected in batch {batch_index}."
                )

            if is_training:
                loss.backward()

                if gradient_clip_norm is not None:
                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                optimizer.step()

        batch_size = signals.shape[0]

        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        total_batches += 1

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("No batches were processed during this epoch.")

    epoch_logits = torch.cat(all_logits, dim=0)
    epoch_targets = torch.cat(all_targets, dim=0)

    metrics = compute_multilabel_metrics(
        logits=epoch_logits,
        targets=epoch_targets,
        class_names=class_names,
        threshold=threshold,
    )

    results: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "num_samples": total_samples,
        "num_batches": total_batches,
        **metrics,
    }

    return results


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: tuple[str, ...] | list[str],
    threshold: float = 0.5,
    gradient_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Train the model for one epoch."""
    _validate_engine_arguments(
        class_names=class_names,
        threshold=threshold,
        max_batches=max_batches,
        gradient_clip_norm=gradient_clip_norm,
    )

    return _run_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        class_names=class_names,
        threshold=threshold,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        max_batches=max_batches,
    )


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...] | list[str],
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate the model without calculating gradients."""
    _validate_engine_arguments(
        class_names=class_names,
        threshold=threshold,
        max_batches=max_batches,
        gradient_clip_norm=None,
    )

    return _run_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        class_names=class_names,
        threshold=threshold,
        optimizer=None,
        gradient_clip_norm=None,
        max_batches=max_batches,
    )
