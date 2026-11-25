
import numpy as np
import pytest
import importlib

preprocessing = importlib.import_module("cardiosignal.data.preprocessing")
compute_channel_statistics = preprocessing.compute_channel_statistics
normalize_ecg_signal = preprocessing.normalize_ecg_signal
validate_ecg_signal = preprocessing.validate_ecg_signal


def test_validate_ecg_signal_returns_float32():
    signal = np.ones((100, 3), dtype=np.float64)

    result = validate_ecg_signal(
        signal,
        expected_samples=100,
        expected_leads=3,
    )

    assert result.shape == (100, 3)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]


def test_validate_ecg_signal_rejects_wrong_dimensions():
    signal = np.ones(100)

    with pytest.raises(ValueError, match="2D"):
        validate_ecg_signal(
            signal,
            expected_samples=100,
            expected_leads=3,
        )


def test_validate_ecg_signal_rejects_wrong_shape():
    signal = np.ones((100, 2))

    with pytest.raises(ValueError, match="Expected ECG shape"):
        validate_ecg_signal(
            signal,
            expected_samples=100,
            expected_leads=3,
        )


def test_validate_ecg_signal_rejects_non_finite_values():
    signal = np.ones((100, 3))
    signal[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_ecg_signal(
            signal,
            expected_samples=100,
            expected_leads=3,
        )


def test_compute_channel_statistics():
    signal_1 = np.full((100, 3), 2.0)
    signal_2 = np.full((100, 3), 4.0)

    mean, std = compute_channel_statistics(
        [signal_1, signal_2],
        expected_samples=100,
        expected_leads=3,
    )

    np.testing.assert_allclose(mean, [3.0, 3.0, 3.0])
    np.testing.assert_allclose(std, [1.0, 1.0, 1.0])


def test_compute_channel_statistics_rejects_empty_input():
    with pytest.raises(ValueError, match="empty dataset"):
        compute_channel_statistics(
            [],
            expected_samples=100,
            expected_leads=3,
        )


def test_normalize_ecg_signal():
    signal = np.full((100, 3), 5.0)

    mean = np.array([3.0, 3.0, 3.0])
    std = np.array([2.0, 2.0, 2.0])

    result = normalize_ecg_signal(
        signal,
        mean,
        std,
        expected_samples=100,
        expected_leads=3,
    )

    assert result.shape == (3, 100)
    assert result.dtype == np.float32

    np.testing.assert_allclose(result, 1.0)


def test_normalize_ecg_signal_rejects_invalid_statistics():
    signal = np.ones((100, 3))

    with pytest.raises(ValueError, match="must be positive"):
        normalize_ecg_signal(
            signal,
            mean=np.zeros(3),
            std=np.array([1.0, 0.0, 1.0]),
            expected_samples=100,
            expected_leads=3,
        )


def test_normalize_ecg_signal_rejects_wrong_mean_shape():
    signal = np.ones((100, 3))

    with pytest.raises(ValueError, match="mean shape"):
        normalize_ecg_signal(
            signal,
            mean=np.zeros(2),
            std=np.ones(3),
            expected_samples=100,
            expected_leads=3,
        )
