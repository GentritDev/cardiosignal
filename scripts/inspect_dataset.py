
"""Inspect PTB-XL metadata, diagnostic labels, and official splits."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from cardiosignal.data.labels import (
    build_label_matrix,
    load_diagnostic_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "ptb-xl-1.0.3"

METADATA_PATH = DATASET_ROOT / "ptbxl_database.csv"
STATEMENTS_PATH = DATASET_ROOT / "scp_statements.csv"


def main() -> None:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"PTB-XL metadata not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "scp_codes",
    }

    missing = required_columns.difference(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing)}"
        )

    diagnostic_mapping = load_diagnostic_mapping(
        STATEMENTS_PATH
    )

    labels = build_label_matrix(
        metadata,
        diagnostic_mapping,
    )

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Total ECG records: {len(metadata):,}")
    print(
        "Unique patients: "
        f"{metadata['patient_id'].nunique():,}"
    )

    print("\n========== DIAGNOSTIC LABEL COUNTS ==========")

    for class_name in labels.columns:
        count = int(labels[class_name].sum())
        percentage = count / len(labels) * 100

        print(
            f"{class_name:>5}: {count:>6,} "
            f"({percentage:6.2f}%)"
        )

    label_counts = labels.sum(axis=1)

    print("\n========== LABEL DISTRIBUTION ==========")
    print(
        "Records with at least one diagnostic label: "
        f"{int(label_counts.gt(0).sum()):,}"
    )
    print(
        "Records without a diagnostic label: "
        f"{int(label_counts.eq(0).sum()):,}"
    )
    print(
        "Records with multiple diagnostic labels: "
        f"{int(label_counts.gt(1).sum()):,}"
    )

    print("\nNumber of diagnostic labels per record:")
    print(label_counts.value_counts().sort_index())

    print("\n========== OFFICIAL SPLITS ==========")

    metadata = metadata.copy()
    metadata["strat_fold"] = pd.to_numeric(
        metadata["strat_fold"],
        errors="raise",
    ).astype(int)

    if not metadata["strat_fold"].between(1, 10).all():
        raise ValueError("Unexpected strat_fold values found.")

    split_frames = {
        "train": metadata[metadata["strat_fold"].between(1, 8)],
        "validation": metadata[metadata["strat_fold"].eq(9)],
        "test": metadata[metadata["strat_fold"].eq(10)],
    }

    for split_name, split_df in split_frames.items():
        print(
            f"{split_name:>10}: "
            f"{len(split_df):>6,} records | "
            f"{split_df['patient_id'].nunique():>6,} patients"
        )

    print("\n========== PATIENT OVERLAP CHECK ==========")

    patient_sets = {
        name: set(split_df["patient_id"].dropna())
        for name, split_df in split_frames.items()
    }

    overlap_found = False

    for first, second in combinations(patient_sets, 2):
        overlap = patient_sets[first].intersection(
            patient_sets[second]
        )

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} shared patients"
        )

        if overlap:
            overlap_found = True

    if overlap_found:
        raise RuntimeError(
            "Patient leakage detected between dataset splits."
        )

    print("\nDataset inspection completed successfully.")


if __name__ == "__main__":
    main()
