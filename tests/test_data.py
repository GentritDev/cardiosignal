
import pandas as pd
import pytest

from cardiosignal.data.labels import (
    DIAGNOSTIC_CLASSES,
    aggregate_diagnostic_labels,
    build_label_matrix,
    load_diagnostic_mapping,
    parse_scp_codes,
)


@pytest.fixture
def diagnostic_mapping() -> dict[str, str]:
    return {
        "NORM": "NORM",
        "IMI": "MI",
        "ASMI": "MI",
        "STTC_CODE": "STTC",
        "RBBB": "CD",
        "LVH": "HYP",
    }


def test_parse_scp_codes() -> None:
    result = parse_scp_codes(
        "{'NORM': 100.0, 'IMI': 50.0}"
    )

    assert result == {
        "NORM": 100.0,
        "IMI": 50.0,
    }


def test_parse_scp_codes_invalid_input() -> None:
    with pytest.raises(ValueError):
        parse_scp_codes("{'NORM': 100")


def test_aggregate_diagnostic_labels(
    diagnostic_mapping: dict[str, str],
) -> None:
    result = aggregate_diagnostic_labels(
        "{'IMI': 100.0, 'RBBB': 100.0}",
        diagnostic_mapping,
    )

    assert result == ("MI", "CD")


def test_unknown_scp_code_is_ignored(
    diagnostic_mapping: dict[str, str],
) -> None:
    result = aggregate_diagnostic_labels(
        "{'UNKNOWN_CODE': 100.0}",
        diagnostic_mapping,
    )

    assert result == ()


def test_build_label_matrix(
    diagnostic_mapping: dict[str, str],
) -> None:
    metadata = pd.DataFrame(
        {
            "scp_codes": [
                "{'NORM': 100.0}",
                "{'IMI': 100.0, 'RBBB': 100.0}",
                "{'LVH': 100.0}",
                "{'UNKNOWN_CODE': 100.0}",
            ]
        }
    )

    result = build_label_matrix(
        metadata,
        diagnostic_mapping,
    )

    assert list(result.columns) == list(DIAGNOSTIC_CLASSES)

    assert result.values.tolist() == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]

    assert result.shape == (4, 5)


def test_load_diagnostic_mapping(tmp_path) -> None:
    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 1, 1, 0],
            "diagnostic_class": [
                "NORM",
                "MI",
                "STTC",
                "CD",
                "HYP",
                None,
            ],
        },
        index=[
            "NORM_CODE",
            "MI_CODE",
            "STTC_CODE",
            "CD_CODE",
            "HYP_CODE",
            "RHYTHM_CODE",
        ],
    )

    path = tmp_path / "scp_statements.csv"
    statements.to_csv(path)

    mapping = load_diagnostic_mapping(path)

    assert len(mapping) == 5
    assert mapping["MI_CODE"] == "MI"
    assert "RHYTHM_CODE" not in mapping
