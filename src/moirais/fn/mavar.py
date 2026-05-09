# moirais.fn — function file (hadesllm/moirais)
"""Moving average smoother (simple, weighted, exponential)."""

import numpy as np

from ._containers import DescriptiveResult


def moving_average(
    y: np.ndarray,
    window: int = 5,
    method: str = "simple",
    weights: np.ndarray | None = None,
) -> DescriptiveResult:
    """
    Moving average smoother.

    :param y: 1-D time series.
    :param window: Window size. Default 5.
    :param method: 'simple', 'weighted', or 'exponential'. Default 'simple'.
    :param weights: Custom weights for 'weighted' method (length = window).
    :return: DescriptiveResult with smoothed series.
    :raises ValueError: If window > n or invalid method.

    References
    ----------
    Hamilton J.D. (1994). Time Series Analysis. Princeton.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if window < 1 or window > n:
        raise ValueError(f"window must be in [1, {n}], got {window}.")
    if method == "simple":
        smoothed = np.convolve(y, np.ones(window) / window, mode="same")
    elif method == "weighted":
        if weights is None:
            w = np.arange(1, window + 1, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != window:
                raise ValueError(f"weights length must equal window ({window}).")
        w = w / w.sum()
        smoothed = np.convolve(y, w[::-1], mode="same")
    elif method == "exponential":
        alpha = 2.0 / (window + 1)
        smoothed = np.zeros(n)
        smoothed[0] = y[0]
        for t in range(1, n):
            smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t - 1]
    else:
        raise ValueError(f"method must be 'simple', 'weighted', or 'exponential', got '{method}'.")
    return DescriptiveResult(
        name="moving_average",
        value=float(smoothed[-1]),
        extra={
            "smoothed": smoothed,
            "window": window,
            "method": method,
            "n": n,
        },
    )


mavar = moving_average


def cheatsheet() -> str:
    return "moving_average({}) -> Moving average smoother (simple/weighted/exponential)."
