
"""Main training entry point for CardioSignal."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from torch import nn

from cardiosignal.data.loader import create_dataloaders
from cardiosignal.models.cnn1d import ECG1DCNN
from cardiosignal.models.resnet1d import ECGResNet1D
from cardiosignal.training.engine import (
    evaluate_one_epoch,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGGER = logging.getLogger("cardiosignal.training")


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_config: str) -> torch.device:
    """Resolve the requested training device."""
    if device_config == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_config)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return device


def build_model(
    model_config: dict[str, Any],
    model_name: str,
) -> nn.Module:
    """Create a model from the model configuration."""
    config = model_config["model"]

    architecture_config = config["architecture"]

    if model_name not in architecture_config:
        raise ValueError(
            f"Unknown model architecture: {model_name}"
        )

    model_kwargs = architecture_config[model_name]

    common_arguments = {
        "in_channels": int(config["input_channels"]),
        "num_classes": int(config["num_classes"]),
        "dropout": float(model_kwargs["dropout"]),
    }

    if model_name == "cnn1d":
        model = ECG1DCNN(
            **common_arguments,
            channels=tuple(model_kwargs["channels"]),
        )

    elif model_name == "resnet1d":
        model = ECGResNet1D(
            **common_arguments,
            channels=tuple(model_kwargs["channels"]),
            blocks_per_stage=tuple(
                model_kwargs["blocks_per_stage"]
            ),
        )

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}"
        )

    return model


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_f1: float,
    model_name: str,
    class_names: list[str],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    """Save a checkpoint atomically."""
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "class_names": class_names,
        "model_config": model_config,
        "training_config": training_config,
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def log_finite_metrics(
    metrics: dict[str, Any],
    step: int,
    prefix: str,
) -> None:
    """Log finite numeric metrics to MLflow."""
    values = {}

    for name, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_value = float(value)

            if np.isfinite(numeric_value):
                values[f"{prefix}_{name}"] = numeric_value

    if values:
        mlflow.log_metrics(values, step=step)


def run_training(
    model_name: str,
    epochs: int | None = None,
    max_batches: int | None = None,
    batch_size: int | None = None,
    enable_mlflow: bool = True,
    run_name: str | None = None,
) -> Path:
    """Execute the complete model training workflow."""
    data_config = load_yaml(
        PROJECT_ROOT / "configs" / "data.yaml"
    )
    model_config = load_yaml(
        PROJECT_ROOT / "configs" / "model.yaml"
    )
    training_file_config = load_yaml(
        PROJECT_ROOT / "configs" / "training.yaml"
    )

    training_config = training_file_config["training"]

    seed = int(training_config["seed"])

    reproducibility = training_config.get(
        "reproducibility", {}
    )
    deterministic = bool(
        reproducibility.get("deterministic", True)
    )

    set_seed(seed, deterministic)

    device = resolve_device(
        str(training_config.get("device", "auto"))
    )

    actual_epochs = (
        epochs
        if epochs is not None
        else int(training_config["epochs"])
    )

    actual_batch_size = (
        batch_size
        if batch_size is not None
        else int(training_config["batch_size"])
    )

    if actual_epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if actual_batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    model_definition = model_config["model"]
    class_names = list(model_definition["class_names"])

    if len(class_names) != int(model_definition["num_classes"]):
        raise ValueError(
            "num_classes does not match class_names."
        )

    if class_names != list(data_config["labels"]["classes"]):
        raise ValueError(
            "Class names in model.yaml and data.yaml do not match."
        )

    LOGGER.info("Preparing DataLoaders...")

    dataloaders = create_dataloaders(
        project_root=PROJECT_ROOT,
        batch_size=actual_batch_size,
        num_workers=int(training_config["num_workers"]),
    )

    LOGGER.info(
        "Dataset sizes | train=%s | validation=%s | test=%s",
        f"{len(dataloaders['train'].dataset):,}",
        f"{len(dataloaders['validation'].dataset):,}",
        f"{len(dataloaders['test'].dataset):,}",
    )

    model = build_model(
        model_config=model_config,
        model_name=model_name,
    ).to(device)

    loss_config = training_config["loss"]

    if loss_config["name"] != "BCEWithLogitsLoss":
        raise ValueError(
            f"Unsupported loss function: {loss_config['name']}"
        )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer_config = training_config["optimizer"]

    if optimizer_config["name"] != "AdamW":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_config['name']}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    scheduler_config = training_config["scheduler"]

    if scheduler_config["name"] != "ReduceLROnPlateau":
        raise ValueError(
            "Only ReduceLROnPlateau is currently supported."
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_lr"]),
    )

    threshold = float(
        training_config["evaluation"]["threshold"]
    )

    gradient_clip_norm = float(
        training_config.get("gradient_clip_norm", 1.0)
    )

    early_stopping_config = training_config["early_stopping"]
    early_stopping_enabled = bool(
        early_stopping_config["enabled"]
    )
    early_stopping_patience = int(
        early_stopping_config["patience"]
    )
    min_delta = float(
        early_stopping_config["min_delta"]
    )

    if early_stopping_patience < 1:
        raise ValueError(
            "Early stopping patience must be at least 1."
        )

    checkpoint_config = training_config["checkpoint"]
    checkpoint_directory = (
        PROJECT_ROOT / checkpoint_config["directory"]
    )

    # Keep smoke-test checkpoints separate from full training checkpoints.
    checkpoint_suffix = (
        "_smoke_test_best.pt"
        if max_batches is not None
        else "_best.pt"
    )

    checkpoint_path = (
        checkpoint_directory
        / f"{model_name}{checkpoint_suffix}"
    )

    tracking_config = training_config.get(
        "experiment_tracking", {}
    )

    use_mlflow = (
        enable_mlflow
        and bool(tracking_config.get("enabled", False))
    )

    if use_mlflow:
        tracking_uri = tracking_config["tracking_uri"]

        if tracking_uri.startswith("."):
            tracking_uri = str(
                (PROJECT_ROOT / tracking_uri).resolve()
            )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(
            tracking_config["experiment_name"]
        )

        context = mlflow.start_run(
            run_name=run_name or model_name
        )
    else:
        context = nullcontext()

    best_val_f1 = float("-inf")
    epochs_without_improvement = 0

    LOGGER.info("=" * 65)
    LOGGER.info("CARDIOSIGNAL TRAINING")
    LOGGER.info("=" * 65)
    LOGGER.info("Model: %s", model_name)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Epochs: %s", actual_epochs)
    LOGGER.info("Batch size: %s", actual_batch_size)
    LOGGER.info("Smoke-test max batches: %s", max_batches)
    LOGGER.info("MLflow enabled: %s", use_mlflow)
    LOGGER.info("=" * 65)

    with context:
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "epochs": actual_epochs,
                    "batch_size": actual_batch_size,
                    "learning_rate": float(
                        optimizer_config["learning_rate"]
                    ),
                    "weight_decay": float(
                        optimizer_config["weight_decay"]
                    ),
                    "loss": loss_config["name"],
                    "optimizer": optimizer_config["name"],
                    "device": str(device),
                    "seed": seed,
                    "threshold": threshold,
                    "max_batches": (
                        max_batches
                        if max_batches is not None
                        else "full"
                    ),
                }
            )

            mlflow.log_dict(
                model_config,
                "configuration/model.yaml.json",
            )
            mlflow.log_dict(
                training_file_config,
                "configuration/training.yaml.json",
            )

        for epoch in range(1, actual_epochs + 1):
            LOGGER.info("")
            LOGGER.info("Epoch %d/%d", epoch, actual_epochs)

            train_results = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                class_names=class_names,
                threshold=threshold,
                gradient_clip_norm=gradient_clip_norm,
                max_batches=max_batches,
            )

            validation_results = evaluate_one_epoch(
                model=model,
                dataloader=dataloaders["validation"],
                loss_fn=loss_fn,
                device=device,
                class_names=class_names,
                threshold=threshold,
                max_batches=max_batches,
            )

            val_macro_f1 = float(
                validation_results["f1_macro"]
            )

            scheduler.step(val_macro_f1)

            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            LOGGER.info(
                "Train | loss=%.5f | macro_f1=%.5f | micro_f1=%.5f",
                train_results["loss"],
                train_results["f1_macro"],
                train_results["f1_micro"],
            )

            LOGGER.info(
                "Valid | loss=%.5f | macro_f1=%.5f | micro_f1=%.5f",
                validation_results["loss"],
                validation_results["f1_macro"],
                validation_results["f1_micro"],
            )

            LOGGER.info("Learning rate: %.8f", current_lr)

            if use_mlflow:
                log_finite_metrics(
                    train_results,
                    step=epoch,
                    prefix="train",
                )
                log_finite_metrics(
                    validation_results,
                    step=epoch,
                    prefix="val",
                )
                mlflow.log_metric(
                    "learning_rate",
                    current_lr,
                    step=epoch,
                )

            improved = (
                np.isfinite(val_macro_f1)
                and val_macro_f1 > best_val_f1 + min_delta
            )

            if improved:
                best_val_f1 = val_macro_f1
                epochs_without_improvement = 0

                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_f1=best_val_f1,
                    model_name=model_name,
                    class_names=class_names,
                    model_config=model_config,
                    training_config=training_file_config,
                )

                LOGGER.info(
                    "New best model saved: %s",
                    checkpoint_path,
                )

                if use_mlflow:
                    mlflow.log_metric(
                        "best_val_macro_f1",
                        best_val_f1,
                        step=epoch,
                    )
            else:
                epochs_without_improvement += 1

                LOGGER.info(
                    "No improvement: %d/%d",
                    epochs_without_improvement,
                    early_stopping_patience,
                )

            if (
                early_stopping_enabled
                and epochs_without_improvement
                >= early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping triggered at epoch %d.",
                    epoch,
                )
                break

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Training completed without producing a valid checkpoint. "
            "Check validation metrics and training logs."
        )

    LOGGER.info("")
    LOGGER.info("=" * 65)
    LOGGER.info("TRAINING COMPLETED")
    LOGGER.info("Best validation macro F1: %.5f", best_val_f1)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("=" * 65)

    return checkpoint_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CardioSignal ECG classification model."
    )

    parser.add_argument(
        "--model",
        choices=["cnn1d", "resnet1d"],
        default=None,
        help="Model architecture. Defaults to model.yaml.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured number of epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for smoke testing.",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name.",
    )

    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    setup_logging()
    args = parse_args()

    model_config = load_yaml(
        PROJECT_ROOT / "configs" / "model.yaml"
    )

    model_name = (
        args.model
        or model_config["model"]["architecture"]["default"]
    )

    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be at least 1.")

    run_training(
        model_name=model_name,
        epochs=args.epochs,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        enable_mlflow=not args.no_mlflow,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
