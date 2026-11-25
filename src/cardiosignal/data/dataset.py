
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from cardiosignal.data.labels import (
    DIAGNOSTIC_CLASSES,
    build_label_matrix,
    load_diagnostic_mapping,
)
from cardiosignal.data.preprocessing import (
    normalize_ecg_signal,
    validate_ecg_signal,
)


def load_ptbxl_metadata(
    dataset_root: str | Path,
    metadata_filename: str = "ptbxl_database.csv",
) -> pd.DataFrame:
    """Load and validate the PTB-XL metadata file."""
    root = Path(dataset_root)
    metadata_path = root / metadata_filename

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PTB-XL metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "filename_hr",
        "filename_lr",
        "scp_codes",
    }

    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata.empty:
        raise ValueError("PTB-XL metadata is empty.")

    if not metadata.index.is_unique:
        raise ValueError("Metadata index must be unique.")

    if metadata["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs found in metadata.")

    return metadata


def load_ptbxl_label_matrix(
    metadata: pd.DataFrame,
    statements_path: str | Path,
) -> pd.DataFrame:
    """Build the five diagnostic superclass labels for every ECG."""
    path = Path(statements_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"SCP statements file not found: {path}"
        )

    mapping = load_diagnostic_mapping(path)
    labels = build_label_matrix(metadata, mapping)

    expected_columns = list(DIAGNOSTIC_CLASSES)

    if list(labels.columns) != expected_columns:
        raise ValueError(
            f"Unexpected label columns. Expected {expected_columns}, "
            f"got {list(labels.columns)}."
        )

    return labels


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG recordings.

    Each item contains:
        signal:   float32 tensor with shape (12, 5000)
        labels:   float32 tensor with shape (5,)
        record_id: ECG record identifier

    WFDB signals are loaded lazily in __getitem__, rather than
    loading the entire dataset into RAM.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        metadata: pd.DataFrame,
        label_matrix: pd.DataFrame,
        folds: Sequence[int],
        sampling_rate: int = 500,
        expected_samples: int = 5000,
        expected_leads: int = 12,
        lead_names: Sequence[str] | None = None,
        record_path_column: str = "filename_hr",
        exclude_unlabeled: bool = True,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root).resolve()
        self.sampling_rate = sampling_rate
        self.expected_samples = expected_samples
        self.expected_leads = expected_leads
        self.lead_names = (
            tuple(lead_names) if lead_names is not None else None
        )
        self.record_path_column = record_path_column

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.dataset_root}"
            )

        if not folds:
            raise ValueError("At least one stratified fold is required.")

        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        if expected_samples <= 0 or expected_leads <= 0:
            raise ValueError("Expected samples and leads must be positive.")

        if record_path_column not in metadata.columns:
            raise ValueError(
                f"Metadata is missing record path column: "
                f"{record_path_column}"
            )

        required_metadata = {"ecg_id", "strat_fold"}
        missing = required_metadata.difference(metadata.columns)

        if missing:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing)}"
            )

        if not metadata.index.is_unique:
            raise ValueError("Metadata index must be unique.")

        if not label_matrix.index.is_unique:
            raise ValueError("Label matrix index must be unique.")

        missing_label_rows = metadata.index.difference(label_matrix.index)

        if len(missing_label_rows) > 0:
            raise ValueError(
                f"Label matrix is missing {len(missing_label_rows)} "
                "metadata records."
            )

        missing_label_columns = set(DIAGNOSTIC_CLASSES).difference(
            label_matrix.columns
        )

        if missing_label_columns:
            raise ValueError(
                f"Label matrix is missing classes: "
                f"{sorted(missing_label_columns)}"
            )

        # Align labels explicitly with metadata index and class order.
        aligned_labels = label_matrix.loc[
            metadata.index, list(DIAGNOSTIC_CLASSES)
        ]

        label_values = aligned_labels.to_numpy(dtype=np.float32)

        if not np.isfinite(label_values).all():
            raise ValueError("Label matrix contains non-finite values.")

        if not np.isin(label_values, [0.0, 1.0]).all():
            raise ValueError("Diagnostic labels must contain only 0 and 1.")

        fold_values = pd.to_numeric(
            metadata["strat_fold"],
            errors="coerce",
        )

        if fold_values.isna().any():
            raise ValueError("Metadata contains invalid strat_fold values.")

        selected = fold_values.isin(set(folds))

        self.num_records_before_label_filter = int(selected.sum())

        if exclude_unlabeled:
            has_labels = aligned_labels.sum(axis=1) > 0
            selected &= has_labels

        selected_indices = metadata.index[selected]

        self.metadata = metadata.loc[selected_indices].reset_index(drop=True)

        self.labels = aligned_labels.loc[
            selected_indices
        ].to_numpy(dtype=np.float32, copy=True)

        self.num_records = len(self.metadata)
        self.num_unlabeled_excluded = (
            self.num_records_before_label_filter - self.num_records
        )

        if self.num_records == 0:
            raise ValueError(
                f"No ECG records found for folds {list(folds)}."
            )

        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError(
                "Provide both normalization_mean and "
                "normalization_std, or neither."
            )

        self.normalization_mean = (
            np.asarray(normalization_mean, dtype=np.float64)
            if normalization_mean is not None
            else None
        )
        self.normalization_std = (
            np.asarray(normalization_std, dtype=np.float64)
            if normalization_std is not None
            else None
        )

        if self.normalization_mean is not None:
            if self.normalization_mean.shape != (expected_leads,):
                raise ValueError("Invalid normalization mean shape.")

            if self.normalization_std.shape != (expected_leads,):
                raise ValueError("Invalid normalization standard deviation shape.")

            if not np.isfinite(self.normalization_mean).all():
                raise ValueError("Normalization mean contains non-finite values.")

            if (
                not np.isfinite(self.normalization_std).all()
                or np.any(self.normalization_std <= 0)
            ):
                raise ValueError(
                    "Normalization standard deviations must be finite "
                    "and positive."
                )

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0 or index >= self.num_records:
            raise IndexError(f"Dataset index out of range: {index}")

        row = self.metadata.iloc[index]
        relative_path = Path(str(row[self.record_path_column]))

        if relative_path.is_absolute():
            raise ValueError(
                f"Expected a relative ECG path, got: {relative_path}"
            )

        record_path = (self.dataset_root / relative_path).resolve()

        # Prevent metadata paths from escaping the dataset directory.
        if not record_path.is_relative_to(self.dataset_root):
            raise ValueError(
                f"ECG path escapes dataset root: {relative_path}"
            )

        record = wfdb.rdrecord(str(record_path))

        if not np.isclose(record.fs, self.sampling_rate):
            raise ValueError(
                f"Unexpected sampling rate for ECG {row['ecg_id']}: "
                f"expected {self.sampling_rate}, got {record.fs}."
            )

        if self.lead_names is not None:
            actual_leads = tuple(record.sig_name)

            if actual_leads != self.lead_names:
                raise ValueError(
                    f"Unexpected lead order for ECG {row['ecg_id']}: "
                    f"expected {self.lead_names}, got {actual_leads}."
                )

        signal = validate_ecg_signal(
            record.p_signal,
            expected_samples=self.expected_samples,
            expected_leads=self.expected_leads,
        )

        if self.normalization_mean is not None:
            signal = normalize_ecg_signal(
                signal,
                mean=self.normalization_mean,
                std=self.normalization_std,
                expected_samples=self.expected_samples,
                expected_leads=self.expected_leads,
            )
        else:
            # Convert (time, leads) to (leads, time) for Conv1d.
            signal = np.ascontiguousarray(signal.T, dtype=np.float32)

        # Explicit copies ensure tensors own writable, contiguous memory.
        signal_tensor = torch.from_numpy(
            np.array(signal, dtype=np.float32, order="C", copy=True)
        )

        label_tensor = torch.from_numpy(
            self.labels[index].copy()
        )

        return {
            "signal": signal_tensor,
            "labels": label_tensor,
            "record_id": str(row["ecg_id"]),
        }
