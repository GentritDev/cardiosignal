"""Reusable DataLoader construction for the CardioSignal project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cardiosignal.data.dataset import (
    PTBXLDataset,
    load_ptbxl_label_matrix,
    load_ptbxl_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")

    return config


def load_normalization_statistics(
    statistics_path: Path,
    expected_leads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate training-set normalization statistics."""
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Normalization statistics not found: {statistics_path}. "
            "Run scripts/prepare_data.py first."
        )

    with statistics_path.open("r", encoding="utf-8") as file:
        statistics = json.load(file)

    if "mean" not in statistics or "std" not in statistics:
        raise ValueError(
            "Normalization artifact must contain 'mean' and 'std'."
        )

    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)

    expected_shape = (expected_leads,)

    if mean.shape != expected_shape:
        raise ValueError(
            f"Invalid normalization mean shape: {mean.shape}; "
            f"expected {expected_shape}."
        )

    if std.shape != expected_shape:
        raise ValueError(
            f"Invalid normalization std shape: {std.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain non-finite values.")

    if np.any(std <= 0):
        raise ValueError("All normalization standard deviations must be positive.")

    return mean, std


def create_dataloaders(
    project_root: Path | str | None = None,
    config_path: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Normalization statistics are loaded from the existing artifact.
    They are not recomputed here.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    yaml_path = (
        Path(config_path).resolve()
        if config_path
        else root / "configs" / "data.yaml"
    )

    config = load_yaml_config(yaml_path)

    dataset_config = config["dataset"]
    labels_config = config["labels"]
    splits_config = config["splits"]

    dataset_root = root / dataset_config["root"]

    metadata = load_ptbxl_metadata(
        dataset_root=dataset_root,
        metadata_filename=dataset_config["metadata_file"],
    )

    statements_path = dataset_root / dataset_config["statements_file"]

    label_matrix = load_ptbxl_label_matrix(
        metadata=metadata,
        statements_path=statements_path,
    )

    expected_leads = int(dataset_config["num_leads"])
    expected_samples = int(dataset_config["num_samples"])
    sampling_rate = int(dataset_config["sampling_rate"])
    lead_names = dataset_config["lead_names"]

    if len(lead_names) != expected_leads:
        raise ValueError(
            "Configured lead_names length does not match num_leads."
        )

    statistics_relative_path = config["preprocessing"]["statistics"][
        "save_path"
    ]
    statistics_path = root / statistics_relative_path

    normalization_mean, normalization_std = load_normalization_statistics(
        statistics_path=statistics_path,
        expected_leads=expected_leads,
    )

    exclude_unlabeled = bool(labels_config["exclude_unlabeled"])

    split_folds = {
        "train": splits_config["train"],
        "validation": splits_config["validation"],
        "test": splits_config["test"],
    }

    datasets: dict[str, PTBXLDataset] = {}

    for split_name, folds in split_folds.items():
        datasets[split_name] = PTBXLDataset(
            dataset_root=dataset_root,
            metadata=metadata,
            label_matrix=label_matrix,
            folds=folds,
            sampling_rate=sampling_rate,
            expected_samples=expected_samples,
            expected_leads=expected_leads,
            lead_names=lead_names,
            exclude_unlabeled=exclude_unlabeled,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        if len(datasets[split_name]) == 0:
            raise ValueError(f"The {split_name} dataset is empty.")

    seed = int(config["reproducibility"]["seed"])

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataloaders: dict[str, DataLoader] = {}

    for split_name, dataset in datasets.items():
        generator = torch.Generator()
        generator.manual_seed(seed)

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=generator,
            persistent_workers=(num_workers > 0),
        )

    return dataloaders