
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
StatsArray = npt.NDArray[np.float64]


def validate_ecg_signal(
    signal: npt.ArrayLike,
    expected_samples: int = 5000,
    expected_leads: int = 12,
) -> FloatArray:
    """
    Validate an ECG recording.

    Expected input shape: (time_samples, leads).
    Returns a float32 NumPy array.
    """
    array = np.asarray(signal, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D ECG signal, got {array.ndim} dimensions."
        )

    expected_shape = (expected_samples, expected_leads)

    if array.shape != expected_shape:
        raise ValueError(
            f"Expected ECG shape {expected_shape}, got {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError("ECG signal contains NaN or infinite values.")

    return np.ascontiguousarray(array)


def compute_channel_statistics(
    signals: Iterable[npt.ArrayLike],
    expected_samples: int = 5000,
    expected_leads: int = 12,
    epsilon: float = 1e-8,
) -> tuple[StatsArray, StatsArray]:
    """
    Compute per-lead mean and standard deviation using streaming
    statistics.

    Statistics are calculated across all time samples and recordings.
    The function does not load the complete dataset into memory.

    IMPORTANT:
    Pass training recordings only to avoid data leakage.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    count = 0
    mean = np.zeros(expected_leads, dtype=np.float64)
    m2 = np.zeros(expected_leads, dtype=np.float64)

    for signal in signals:
        array = validate_ecg_signal(
            signal,
            expected_samples=expected_samples,
            expected_leads=expected_leads,
        ).astype(np.float64, copy=False)

        batch_count = array.shape[0]
        batch_mean = array.mean(axis=0)
        batch_m2 = np.square(array - batch_mean).sum(axis=0)

        if count == 0:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue

        total_count = count + batch_count
        delta = batch_mean - mean

        m2 += batch_m2 + np.square(delta) * (
            count * batch_count / total_count
        )
        mean += delta * (batch_count / total_count)
        count = total_count

    if count == 0:
        raise ValueError("Cannot compute statistics from an empty dataset.")

    variance = np.maximum(m2 / count, 0.0)
    std = np.sqrt(variance)

    # Protect against division by zero for constant channels.
    std = np.maximum(std, epsilon)

    return mean, std


def normalize_ecg_signal(
    signal: npt.ArrayLike,
    mean: npt.ArrayLike,
    std: npt.ArrayLike,
    expected_samples: int = 5000,
    expected_leads: int = 12,
    epsilon: float = 1e-8,
) -> FloatArray:
    """
    Normalize an ECG using training-set per-lead statistics.

    Input shape:  (time_samples, leads)
    Output shape: (leads, time_samples)

    The output layout is suitable for PyTorch Conv1d.
    """
    array = validate_ecg_signal(
        signal,
        expected_samples=expected_samples,
        expected_leads=expected_leads,
    )

    channel_mean = np.asarray(mean, dtype=np.float64)
    channel_std = np.asarray(std, dtype=np.float64)

    if channel_mean.shape != (expected_leads,):
        raise ValueError(
            f"Expected mean shape ({expected_leads},), "
            f"got {channel_mean.shape}."
        )

    if channel_std.shape != (expected_leads,):
        raise ValueError(
            f"Expected std shape ({expected_leads},), "
            f"got {channel_std.shape}."
        )

    if not np.isfinite(channel_mean).all():
        raise ValueError("Mean contains NaN or infinite values.")

    if not np.isfinite(channel_std).all():
        raise ValueError("Standard deviation contains NaN or infinite values.")

    if np.any(channel_std <= 0):
        raise ValueError("All standard deviations must be positive.")

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    safe_std = np.maximum(channel_std, epsilon)

    normalized = (
        array.astype(np.float64, copy=False) - channel_mean
    ) / safe_std

    # PyTorch Conv1d expects (channels, sequence_length).
    return np.ascontiguousarray(normalized.T, dtype=np.float32)
