# moirais.fn — function file (hadesllm/moirais)
"""Moving average smoother for time series."""

import numpy as np


def ma_(
    series: np.ndarray,
    window: int = 5,
    center: bool = True,
) -> np.ndarray:
    """
    Simple moving average smoother.

    Computes a rolling arithmetic mean over a sliding window.

    :param series: 1-D array of time series values.
    :param window: Width of the moving-average window. Must be >= 1.
        Default 5.
    :param center: If True, center the window so that the smoothed value
        at index *t* uses observations symmetrically around *t*.
        Boundary values are NaN. Default True.
    :return: 1-D ndarray of smoothed values (same length as *series*).
        Positions where the full window is unavailable contain NaN.
    :raises ValueError: If window < 1 or series is empty.

    References
    ----------
    Shumway, R. H. & Stoffer, D. S. (2017). Time Series Analysis and
    Its Applications (4th ed.). Springer.
    """
    y = np.asarray(series, dtype=float)
    if len(y) == 0:
        raise ValueError("Series must not be empty.")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if window == 1:
        return y.copy()

    n = len(y)
    result = np.full(n, np.nan)
    cumsum = np.concatenate([[0.0], np.cumsum(y)])

    if center:
        half = window // 2
        for i in range(n):
            lo = i - half
            hi = i + (window - half)
            if lo >= 0 and hi <= n:
                result[i] = (cumsum[hi] - cumsum[lo]) / window
    else:
        for i in range(window - 1, n):
            lo = i - window + 1
            result[i] = (cumsum[i + 1] - cumsum[lo]) / window

    return result


def cheatsheet() -> str:
    return "ma_({}) -> Moving average smoother for time series."
