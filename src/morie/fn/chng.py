# morie.fn — function file (hadesllm/morie)
"""CUSUM changepoint detection."""

import numpy as np

from ._containers import DescriptiveResult


def changepoint_detect(y: np.ndarray, threshold: float | None = None) -> DescriptiveResult:
    """
    Detect changepoints using CUSUM (cumulative sum) method.

    Computes the CUSUM statistic and identifies the point of maximum
    deviation from the overall mean, indicating a potential changepoint.

    :param y: (n,) time series.
    :param threshold: Detection threshold (default: 2 * std).
    :return: DescriptiveResult with changepoint index and CUSUM values.
    :raises ValueError: If series too short.

    References
    ----------
    Page ES (1954). Continuous inspection schemes.
    Biometrika, 41(1-2), 100-115.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n < 4:
        raise ValueError("Need at least 4 observations.")
    mean_y = y.mean()
    cusum = np.cumsum(y - mean_y)
    if threshold is None:
        threshold = 2 * y.std()
    cp_idx = int(np.argmax(np.abs(cusum)))
    cp_value = float(cusum[cp_idx])
    detected = abs(cp_value) > threshold
    changepoints = []
    if detected:
        changepoints.append(cp_idx)
    return DescriptiveResult(
        name="changepoint_detect",
        value=float(cp_idx),
        extra={
            "cusum": cusum,
            "changepoint_index": cp_idx,
            "cusum_value": cp_value,
            "threshold": float(threshold),
            "detected": detected,
            "changepoints": changepoints,
            "n": n,
        },
    )


chng = changepoint_detect


def cheatsheet() -> str:
    return "changepoint_detect({}) -> CUSUM changepoint detection."
