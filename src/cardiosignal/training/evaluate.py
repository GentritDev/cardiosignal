
"""Evaluation entry point for CardioSignal."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cardiosignal.data.loader import create_dataloaders
from cardiosignal.training.engine import evaluate_one_epoch
from cardiosignal.training.train import build_model, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGGER = logging.getLogger("cardiosignal.evaluation")


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def to_json_compatible(value: Any) -> Any:
    """Convert metric values into JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())

    if isinstance(value, np.generic):
        return to_json_compatible(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def get_class_metric(
    metrics: dict[str, Any],
    metric_name: str,
    class_name: str,
) -> Any:
    """Read a per-class metric from supported metric layouts."""
    possible_keys = (
        f"{metric_name}_{class_name}",
        f"{class_name}_{metric_name}",
    )

    for key in possible_keys:
        if key in metrics:
            return metrics[key]

    nested_keys = (
        f"{metric_name}_per_class",
        f"per_class_{metric_name}",
    )

    for nested_key in nested_keys:
        nested = metrics.get(nested_key)

        if isinstance(nested, dict) and class_name in nested:
            return nested[class_name]

    return None


def write_json_report(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    """Write an evaluation report atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_json_compatible(report),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_path.replace(output_path)


def write_per_class_csv(
    output_path: Path,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    """Write per-class evaluation metrics to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "class_name",
        "f1",
        "auroc",
        "auprc",
        "positives",
        "negatives",
    ]

    rows = []

    for class_name in class_names:
        rows.append(
            {
                "class_name": class_name,
                "f1": get_class_metric(
                    metrics, "f1", class_name
                ),
                "auroc": get_class_metric(
                    metrics, "auroc", class_name
                ),
                "auprc": get_class_metric(
                    metrics, "auprc", class_name
                ),
                "positives": get_class_metric(
                    metrics, "positives", class_name
                ),
                "negatives": get_class_metric(
                    metrics, "negatives", class_name
                ),
            }
        )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(output_path)


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str = "test",
    batch_size: int | None = None,
    max_batches: int | None = None,
    threshold: float | None = None,
    device_override: str | None = None,
    output_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Evaluate a saved checkpoint and export its metrics."""
    if split not in {"validation", "test"}:
        raise ValueError(
            "split must be either 'validation' or 'test'."
        )

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1.")

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
    model_definition = model_config["model"]

    device = resolve_device(
        device_override
        or str(training_config.get("device", "auto"))
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must contain a dictionary."
        )

    required_keys = {
        "model_name",
        "model_state_dict",
        "class_names",
    }

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model_name = str(checkpoint["model_name"])
    class_names = list(model_definition["class_names"])

    if model_name not in {"cnn1d", "resnet1d"}:
        raise ValueError(
            f"Unsupported model in checkpoint: {model_name}"
        )

    checkpoint_classes = list(checkpoint["class_names"])

    if checkpoint_classes != class_names:
        raise ValueError(
            "Checkpoint class names do not match model.yaml."
        )

    if class_names != list(data_config["labels"]["classes"]):
        raise ValueError(
            "Class names in data.yaml and model.yaml do not match."
        )

    if len(class_names) != int(model_definition["num_classes"]):
        raise ValueError(
            "num_classes does not match class_names."
        )

    model = build_model(
        model_config=model_config,
        model_name=model_name,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    LOGGER.info("Model loaded successfully.")
    LOGGER.info("Model architecture: %s", model_name)
    LOGGER.info("Device: %s", device)

    actual_batch_size = (
        batch_size
        if batch_size is not None
        else int(training_config["batch_size"])
    )

    if actual_batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    actual_threshold = (
        threshold
        if threshold is not None
        else float(training_config["evaluation"]["threshold"])
    )

    if not 0.0 < actual_threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    LOGGER.info("Preparing DataLoaders...")

    dataloaders = create_dataloaders(
        project_root=PROJECT_ROOT,
        batch_size=actual_batch_size,
        num_workers=int(training_config["num_workers"]),
    )

    dataloader = dataloaders[split]

    LOGGER.info(
        "Evaluation split: %s | samples: %s",
        split,
        f"{len(dataloader.dataset):,}",
    )

    if split == "test" and max_batches is not None:
        LOGGER.warning(
            "This is a partial test-set evaluation. "
            "Do not report it as a final test result."
        )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    LOGGER.info("Starting evaluation...")

    results = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        class_names=class_names,
        threshold=actual_threshold,
        max_batches=max_batches,
    )

    LOGGER.info("Evaluation completed.")
    LOGGER.info("Samples evaluated: %s", results["num_samples"])
    LOGGER.info("Batches evaluated: %s", results["num_batches"])
    LOGGER.info("Loss: %.5f", results["loss"])

    for metric_name in (
        "f1_macro",
        "f1_micro",
        "auroc_macro",
        "auprc_macro",
    ):
        value = results.get(metric_name)

        if isinstance(value, (int, float, np.number)):
            if np.isfinite(float(value)):
                LOGGER.info(
                    "%s: %.5f",
                    metric_name,
                    float(value),
                )
            else:
                LOGGER.info("%s: unavailable", metric_name)

    for class_name in class_names:
        f1 = get_class_metric(results, "f1", class_name)
        auroc = get_class_metric(results, "auroc", class_name)
        auprc = get_class_metric(results, "auprc", class_name)

        LOGGER.info(
            "%s | F1=%s | AUROC=%s | AUPRC=%s",
            class_name,
            f"{float(f1):.4f}" if f1 is not None else "N/A",
            f"{float(auroc):.4f}" if auroc is not None else "N/A",
            f"{float(auprc):.4f}" if auprc is not None else "N/A",
        )

    output_directory = (
        output_directory
        or PROJECT_ROOT / "artifacts" / "evaluation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "threshold": actual_threshold,
        "device": str(device),
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": results,
    }

    report_stem = f"{checkpoint_path.stem}_{split}"

    json_path = output_directory / f"{report_stem}_metrics.json"
    csv_path = output_directory / f"{report_stem}_per_class.csv"

    write_json_report(
        output_path=json_path,
        report=report,
    )

    write_per_class_csv(
        output_path=csv_path,
        metrics=results,
        class_names=class_names,
    )

    LOGGER.info("JSON report saved: %s", json_path)
    LOGGER.info("Per-class CSV saved: %s", csv_path)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CardioSignal model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path relative to the project root or absolute.",
    )

    parser.add_argument(
        "--model",
        choices=["cnn1d", "resnet1d"],
        default=None,
        help="Model name when using the default checkpoint path.",
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Dataset split to evaluate. Defaults to test.",
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
        help="Limit batches for a quick evaluation smoke test.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the configured classification threshold.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the configured device, e.g. cpu or cuda.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation reports.",
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

    if args.checkpoint:
        checkpoint_path = resolve_project_path(args.checkpoint)
    else:
        checkpoint_path = resolve_project_path(
            Path("artifacts")
            / "checkpoints"
            / f"{model_name}_best.pt"
        )

    output_directory = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "artifacts" / "evaluation"
    )

    evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        threshold=args.threshold,
        device_override=args.device,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()
