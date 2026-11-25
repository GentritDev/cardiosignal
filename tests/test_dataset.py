
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cardiosignal.data.dataset import (
    PTBXLDataset,
    load_ptbxl_metadata,
)
from cardiosignal.data.labels import DIAGNOSTIC_CLASSES


def make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1001, 1002, 1003],
            "patient_id": [101, 102, 103],
            "strat_fold": [1, 9, 1],
            "filename_hr": [
                "records500/00000/00001_hr",
                "records500/00000/00002_hr",
                "records500/00000/00003_hr",
            ],
        },
        index=[10, 11, 12],
    )


def make_labels() -> pd.DataFrame:
    values = np.zeros((3, len(DIAGNOSTIC_CLASSES)), dtype=np.uint8)

    # First ECG: NORM
    values[0, DIAGNOSTIC_CLASSES.index("NORM")] = 1

    # Second ECG: MI
    values[1, DIAGNOSTIC_CLASSES.index("MI")] = 1

    # Third ECG: deliberately unlabeled

    return pd.DataFrame(
        values,
        columns=DIAGNOSTIC_CLASSES,
        index=[10, 11, 12],
    )


def test_dataset_filters_by_fold_and_excludes_unlabeled(tmp_path):
    dataset = PTBXLDataset(
        dataset_root=tmp_path,
        metadata=make_metadata(),
        label_matrix=make_labels(),
        folds=[1],
        expected_samples=4,
        expected_leads=2,
        exclude_unlabeled=True,
    )

    assert len(dataset) == 1
    assert dataset.num_records_before_label_filter == 2
    assert dataset.num_unlabeled_excluded == 1
    assert dataset.metadata.iloc[0]["ecg_id"] == 1001


def test_dataset_can_keep_unlabeled_records(tmp_path):
    dataset = PTBXLDataset(
        dataset_root=tmp_path,
        metadata=make_metadata(),
        label_matrix=make_labels(),
        folds=[1],
        expected_samples=4,
        expected_leads=2,
        exclude_unlabeled=False,
    )

    assert len(dataset) == 2
    assert dataset.num_unlabeled_excluded == 0


def test_dataset_rejects_empty_folds(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        PTBXLDataset(
            dataset_root=tmp_path,
            metadata=make_metadata(),
            label_matrix=make_labels(),
            folds=[],
            expected_samples=4,
            expected_leads=2,
        )


def test_dataset_loads_ecg_as_pytorch_tensors(
    tmp_path,
    monkeypatch,
):
    metadata = make_metadata()
    labels = make_labels()

    def fake_rdrecord(record_path):
        assert record_path.endswith("00001_hr")

        return SimpleNamespace(
            p_signal=np.ones((4, 2), dtype=np.float64),
            fs=500,
            sig_name=["I", "II"],
        )

    monkeypatch.setattr(
        "cardiosignal.data.dataset.wfdb.rdrecord",
        fake_rdrecord,
    )

    dataset = PTBXLDataset(
        dataset_root=tmp_path,
        metadata=metadata,
        label_matrix=labels,
        folds=[1],
        sampling_rate=500,
        expected_samples=4,
        expected_leads=2,
        lead_names=["I", "II"],
        exclude_unlabeled=True,
    )

    sample = dataset[0]

    assert isinstance(sample["signal"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)

    assert sample["signal"].shape == (2, 4)
    assert sample["labels"].shape == (5,)

    assert sample["signal"].dtype == torch.float32
    assert sample["labels"].dtype == torch.float32

    assert sample["record_id"] == "1001"

    np.testing.assert_allclose(
        sample["signal"].numpy(),
        np.ones((2, 4)),
    )


def test_dataset_rejects_wrong_sampling_rate(
    tmp_path,
    monkeypatch,
):
    def fake_rdrecord(record_path):
        return SimpleNamespace(
            p_signal=np.ones((4, 2)),
            fs=100,
            sig_name=["I", "II"],
        )

    monkeypatch.setattr(
        "cardiosignal.data.dataset.wfdb.rdrecord",
        fake_rdrecord,
    )

    dataset = PTBXLDataset(
        dataset_root=tmp_path,
        metadata=make_metadata(),
        label_matrix=make_labels(),
        folds=[1],
        sampling_rate=500,
        expected_samples=4,
        expected_leads=2,
        exclude_unlabeled=True,
    )

    with pytest.raises(ValueError, match="Unexpected sampling rate"):
        dataset[0]


def test_load_ptbxl_metadata(tmp_path):
    metadata = pd.DataFrame(
        {
            "ecg_id": [1],
            "patient_id": [101],
            "strat_fold": [1],
            "filename_hr": ["records500/00000/00001_hr"],
            "filename_lr": ["records100/00000/00001_lr"],
            "scp_codes": ["{'NORM': 100.0}"],
        }
    )

    metadata.to_csv(tmp_path / "ptbxl_database.csv", index=False)

    result = load_ptbxl_metadata(tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["ecg_id"] == 1
    assert result.iloc[0]["strat_fold"] == 1


def test_load_ptbxl_metadata_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ptbxl_metadata(tmp_path)
