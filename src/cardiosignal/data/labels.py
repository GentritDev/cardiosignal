
"""Utilities for mapping PTB-XL SCP statements to diagnostic superclasses."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

DIAGNOSTIC_CLASSES: tuple[str, ...] = (
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
)

REQUIRED_STATEMENT_COLUMNS = {
    "diagnostic",
    "diagnostic_class",
}


def load_diagnostic_mapping(
    statements_path: str | Path,
) -> dict[str, str]:
    """Load mappings from diagnostic SCP codes to diagnostic superclasses.

    Only SCP statements marked as diagnostic are included.
    """
    statements_path = Path(statements_path)

    if not statements_path.is_file():
        raise FileNotFoundError(
            f"SCP statements file not found: {statements_path}"
        )

    statements = pd.read_csv(statements_path, index_col=0)

    missing_columns = REQUIRED_STATEMENT_COLUMNS.difference(
        statements.columns
    )

    if missing_columns:
        raise ValueError(
            f"SCP statements file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    diagnostic_flags = pd.to_numeric(
        statements["diagnostic"],
        errors="coerce",
    )

    diagnostic_classes = (
        statements["diagnostic_class"]
        .astype("string")
        .str.strip()
    )

    mask = (
        diagnostic_flags.eq(1)
        & diagnostic_classes.isin(DIAGNOSTIC_CLASSES)
    )

    filtered = statements.loc[mask, ["diagnostic_class"]]

    mapping = {
        str(code): str(row["diagnostic_class"])
        for code, row in filtered.iterrows()
    }

    mapped_classes = set(mapping.values())
    missing_classes = set(DIAGNOSTIC_CLASSES).difference(
        mapped_classes
    )

    if missing_classes:
        raise ValueError(
            "The SCP mapping does not contain all expected "
            f"diagnostic classes: {sorted(missing_classes)}"
        )

    return mapping


def parse_scp_codes(value: Any) -> dict[str, float]:
    """Parse an SCP-code dictionary from PTB-XL metadata.

    The likelihood values are retained as metadata but are not
    interpreted as model probabilities.
    """
    if isinstance(value, Mapping):
        parsed = dict(value)

    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid SCP-code dictionary: {value!r}"
            ) from exc

    elif value is None or (
        isinstance(value, float) and math.isnan(value)
    ):
        return {}

    else:
        raise TypeError(
            "SCP codes must be a dictionary or a string "
            f"containing a dictionary. Received: {type(value).__name__}"
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "Parsed SCP codes must be a dictionary."
        )

    normalized: dict[str, float] = {}

    for code, likelihood in parsed.items():
        if not isinstance(code, str):
            raise ValueError(
                f"SCP code must be a string. Received: {code!r}"
            )

        try:
            normalized[code] = float(likelihood)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid likelihood for SCP code {code!r}: "
                f"{likelihood!r}"
            ) from exc

    return normalized


def aggregate_diagnostic_labels(
    scp_codes: Any,
    diagnostic_mapping: Mapping[str, str],
) -> tuple[str, ...]:
    """Convert SCP codes into diagnostic superclass labels.

    One ECG may contain multiple diagnostic superclasses.
    """
    parsed_codes = parse_scp_codes(scp_codes)

    found_classes = {
        diagnostic_mapping[code]
        for code in parsed_codes
        if code in diagnostic_mapping
    }

    # Preserve a consistent order across all records.
    return tuple(
        class_name
        for class_name in DIAGNOSTIC_CLASSES
        if class_name in found_classes
    )


def build_label_matrix(
    metadata: pd.DataFrame,
    diagnostic_mapping: Mapping[str, str],
    scp_codes_column: str = "scp_codes",
) -> pd.DataFrame:
    """Build a multi-hot diagnostic target matrix for ECG records."""
    if scp_codes_column not in metadata.columns:
        raise ValueError(
            f"Metadata is missing column: {scp_codes_column!r}"
        )

    labels_per_record = [
        set(
            aggregate_diagnostic_labels(
                value,
                diagnostic_mapping,
            )
        )
        for value in metadata[scp_codes_column]
    ]

    label_matrix = pd.DataFrame(
        {
            class_name: [
                int(class_name in record_labels)
                for record_labels in labels_per_record
            ]
            for class_name in DIAGNOSTIC_CLASSES
        },
        index=metadata.index,
        dtype="uint8",
    )

    return label_matrix
