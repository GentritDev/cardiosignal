
import numpy as np
import pytest
import torch

from cardiosignal.training.metrics import compute_multilabel_metrics

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")

def test_metrics_returns_expected_keys() -> None:
    targets = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
        ]
    )

    logits = np.array(
        [
            [3, -3, 2, -2, -2],
            [-3, 3, -2, 2, -2],
            [3, -3, -2, -2, 3],
            [-3, 3, 2, -2, -2],
        ],
        dtype=np.float32,
    )

    metrics = compute_multilabel_metrics(
        logits,
        targets,
        CLASS_NAMES,
    )

    assert "f1_macro" in metrics
    assert "f1_micro" in metrics
    assert "auroc_macro" in metrics
    assert "auprc_macro" in metrics

    for class_name in CLASS_NAMES:
        assert f"f1_{class_name}" in metrics
        assert f"auroc_{class_name}" in metrics
        assert f"auprc_{class_name}" in metrics

def test_perfect_predictions_produce_perfect_f1() -> None:
    targets = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
        ]
    )

    logits = np.where(targets == 1, 5.0, -5.0)

    metrics = compute_multilabel_metrics(
        logits,
        targets,
        CLASS_NAMES,
    )

    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["f1_micro"] == pytest.approx(1.0)
    assert metrics["auroc_macro"] == pytest.approx(1.0)
    assert metrics["auprc_macro"] == pytest.approx(1.0)

def test_metrics_accept_pytorch_tensors() -> None:
    targets = torch.tensor(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor(
        [
            [3, -3, 3, -3, -3],
            [-3, 3, -3, 3, 3],
        ],
        dtype=torch.float32,
    )

    metrics = compute_multilabel_metrics(
        logits,
        targets,
        CLASS_NAMES,
    )

    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["f1_micro"] == pytest.approx(1.0)
def test_metrics_handle_classes_without_positive_samples() -> None:
    targets = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
        ]
    )

    logits = np.zeros((3, 5))

    metrics = compute_multilabel_metrics(
        logits,
        targets,
        CLASS_NAMES,
    )

    assert np.isnan(metrics["auroc_STTC"])
    assert np.isnan(metrics["auprc_STTC"])
    assert np.isfinite(metrics["f1_STTC"])
    assert np.isfinite(metrics["auroc_macro"])

def test_metrics_reject_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="logits must have shape"):
        compute_multilabel_metrics(
            np.zeros(5),
            np.zeros(5),
            CLASS_NAMES,
        )

def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        compute_multilabel_metrics(
            np.zeros((4, 5)),
            np.zeros((4, 4)),
            CLASS_NAMES,
        )

def test_metrics_reject_non_binary_targets() -> None:
    targets = np.zeros((4, 5))
    targets[0, 0] = 2

    with pytest.raises(ValueError, match="binary"):
        compute_multilabel_metrics(
            np.zeros((4, 5)),
            targets,
            CLASS_NAMES,
        )

def test_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_multilabel_metrics(
            np.zeros((4, 5)),
            np.zeros((4, 5)),
            CLASS_NAMES,
            threshold=1.0,
        )