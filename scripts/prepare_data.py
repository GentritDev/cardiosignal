
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cardiosignal.data.dataset import (
    PTBXLDataset,
    load_ptbxl_label_matrix,
    load_ptbxl_metadata,
)
from cardiosignal.data.preprocessing import compute_channel_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 32
NUM_WORKERS = 0  # Safe default for Windows.


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "data.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml configuration.")

    return config


def verify_patient_disjointness(
    metadata: pd.DataFrame,
    fold_column: str,
    patient_column: str,
    splits: dict,
) -> None:
    """Ensure that no patient appears in multiple splits."""
    patient_sets = {}

    for split_name, folds in splits.items():
        fold_values = pd.Series(
            pd.to_numeric(metadata[fold_column], errors="coerce"),
            index=metadata.index,
        )

        selected = fold_values.isin(folds)

        patients = set(
            metadata.loc[selected, patient_column].dropna()
        )

        patient_sets[split_name] = patients

    split_names = list(patient_sets)

    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            overlap = patient_sets[first] & patient_sets[second]

            if overlap:
                raise ValueError(
                    f"Patient leakage detected between {first} and "
                    f"{second}: {len(overlap)} shared patients."
                )

            print(f"{first} vs {second}: no shared patients")


def iter_training_signals(train_dataset: PTBXLDataset):
    """
    Yield raw training ECG signals in (time, leads) layout.

    Signals are loaded one at a time to keep memory usage low.
    """
    total = len(train_dataset)

    for index in range(total):
        sample = train_dataset[index]

        # Dataset returns (leads, time).
        # Statistics function expects (time, leads).
        signal = sample["signal"].numpy().T

        if (index + 1) % 500 == 0 or index + 1 == total:
            print(
                f"Processed training ECGs: {index + 1}/{total}",
                flush=True,
            )

        yield signal


def save_normalization_statistics(
    output_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict,
    training_records: int,
) -> None:
    """Save normalization statistics as a reproducible JSON artifact."""
    dataset_config = config["dataset"]
    preprocessing_config = config["preprocessing"]

    payload = {
        "schema_version": 1,
        "dataset": dataset_config["name"],
        "dataset_version": dataset_config["version"],
        "method": preprocessing_config["normalization"]["method"],
        "sampling_rate": dataset_config["sampling_rate"],
        "duration_seconds": dataset_config["duration_seconds"],
        "num_samples": dataset_config["num_samples"],
        "num_leads": dataset_config["num_leads"],
        "lead_names": dataset_config["lead_names"],
        "training_folds": config["splits"]["train"],
        "training_records": training_records,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(payload, indent=4),
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


def load_normalization_statistics(path: Path):
    """Load and validate saved normalization statistics."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalization statistics not found: {path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)

    return payload, mean, std


def build_dataset(
    dataset_root: Path,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    folds: list[int],
    config: dict,
    mean: np.ndarray,
    std: np.ndarray,
) -> PTBXLDataset:
    dataset_config = config["dataset"]
    label_config = config["labels"]

    return PTBXLDataset(
        dataset_root=dataset_root,
        metadata=metadata,
        label_matrix=labels,
        folds=folds,
        sampling_rate=dataset_config["sampling_rate"],
        expected_samples=dataset_config["num_samples"],
        expected_leads=dataset_config["num_leads"],
        lead_names=dataset_config["lead_names"],
        record_path_column="filename_hr",
        exclude_unlabeled=label_config["exclude_unlabeled"],
        normalization_mean=mean,
        normalization_std=std,
    )


def main() -> None:
    print("========== CARDIOSIGNAL DATA PREPARATION ==========")

    config = load_config()

    dataset_config = config["dataset"]
    split_config = config["splits"]
    preprocessing_config = config["preprocessing"]

    dataset_root = PROJECT_ROOT / dataset_config["root"]

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    print("\n[1/7] Loading metadata and diagnostic labels...")

    metadata = load_ptbxl_metadata(
        dataset_root,
        dataset_config["metadata_file"],
    )

    labels = load_ptbxl_label_matrix(
        metadata,
        dataset_root / dataset_config["statements_file"],
    )

    print(f"Metadata records: {len(metadata):,}")
    print(f"Unique patients: {metadata['patient_id'].nunique():,}")

    print("\n[2/7] Verifying official patient-level splits...")

    fold_column = split_config["fold_column"]
    patient_column = split_config["patient_id_column"]

    splits = {
        "train": split_config["train"],
        "validation": split_config["validation"],
        "test": split_config["test"],
    }

    verify_patient_disjointness(
        metadata,
        fold_column,
        patient_column,
        splits,
    )

    print("\n[3/7] Creating raw training dataset...")

    train_dataset = PTBXLDataset(
        dataset_root=dataset_root,
        metadata=metadata,
        label_matrix=labels,
        folds=splits["train"],
        sampling_rate=dataset_config["sampling_rate"],
        expected_samples=dataset_config["num_samples"],
        expected_leads=dataset_config["num_leads"],
        lead_names=dataset_config["lead_names"],
        record_path_column="filename_hr",
        exclude_unlabeled=config["labels"]["exclude_unlabeled"],
    )

    print(f"Training ECGs: {len(train_dataset):,}")
    print(
        "Training records before label filtering: "
        f"{train_dataset.num_records_before_label_filter:,}"
    )
    print(
        f"Unlabeled training records excluded: "
        f"{train_dataset.num_unlabeled_excluded:,}"
    )

    print("\n[4/7] Computing training-only normalization statistics...")
    print("This step reads every training ECG from disk.")

    normalization_config = preprocessing_config["normalization"]

    if normalization_config["method"] != "train_set_per_lead_zscore":
        raise ValueError(
            "Unsupported normalization method: "
            f"{normalization_config['method']}"
        )

    mean, std = compute_channel_statistics(
        iter_training_signals(train_dataset),
        expected_samples=dataset_config["num_samples"],
        expected_leads=dataset_config["num_leads"],
        epsilon=normalization_config["epsilon"],
    )

    print("\nPer-lead training statistics:")

    for lead, channel_mean, channel_std in zip(
        dataset_config["lead_names"],
        mean,
        std,
    ):
        print(
            f"{lead:>3}: mean={channel_mean: .6f}, "
            f"std={channel_std: .6f}"
        )

    print("\n[5/7] Saving normalization statistics...")

    statistics_path = (
        PROJECT_ROOT
        / preprocessing_config["statistics"]["save_path"]
    )

    save_normalization_statistics(
        statistics_path,
        mean,
        std,
        config,
        len(train_dataset),
    )

    print(f"Saved: {statistics_path}")

    _, saved_mean, saved_std = load_normalization_statistics(
        statistics_path
    )

    if not np.allclose(mean, saved_mean):
        raise ValueError("Saved means do not match computed means.")

    if not np.allclose(std, saved_std):
        raise ValueError("Saved standard deviations do not match.")

    print("Saved statistics verified successfully.")

    print("\n[6/7] Building normalized datasets...")

    train_dataset = build_dataset(
        dataset_root,
        metadata,
        labels,
        splits["train"],
        config,
        saved_mean,
        saved_std,
    )

    validation_dataset = build_dataset(
        dataset_root,
        metadata,
        labels,
        splits["validation"],
        config,
        saved_mean,
        saved_std,
    )

    test_dataset = build_dataset(
        dataset_root,
        metadata,
        labels,
        splits["test"],
        config,
        saved_mean,
        saved_std,
    )

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    for name, dataset in datasets.items():
        print(
            f"{name:>10}: {len(dataset):,} ECGs | "
            f"excluded unlabeled: {dataset.num_unlabeled_excluded:,}"
        )

        positive_counts = dataset.labels.sum(axis=0)

        for class_name, count in zip(
            config["labels"]["classes"],
            positive_counts,
        ):
            print(f"    {class_name}: {int(count):,}")

    print("\n[7/7] Creating and testing DataLoaders...")

    seed = config.get("reproducibility", {}).get("seed", 42)
    generator = torch.Generator().manual_seed(seed)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    for name, loader in loaders.items():
        batch = next(iter(loader))

        signals = batch["signal"]
        targets = batch["labels"]

        expected_signal_shape = (
            min(BATCH_SIZE, len(datasets[name])),
            dataset_config["num_leads"],
            dataset_config["num_samples"],
        )

        expected_label_shape = (
            min(BATCH_SIZE, len(datasets[name])),
            len(config["labels"]["classes"]),
        )

        if tuple(signals.shape) != expected_signal_shape:
            raise ValueError(
                f"Unexpected {name} signal batch shape: "
                f"{tuple(signals.shape)}"
            )

        if tuple(targets.shape) != expected_label_shape:
            raise ValueError(
                f"Unexpected {name} label batch shape: "
                f"{tuple(targets.shape)}"
            )

        if not torch.isfinite(signals).all():
            raise ValueError(f"{name} batch contains non-finite signals.")

        print(
            f"{name:>10}: signals={tuple(signals.shape)}, "
            f"labels={tuple(targets.shape)}, "
            f"dtype={signals.dtype}"
        )

    print("\n========== DATA PREPARATION COMPLETED ==========")
    print(f"Normalization artifact: {statistics_path}")
    print("All datasets and DataLoaders passed validation.")


if __name__ == "__main__":
    main()
