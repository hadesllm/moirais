# morie.fn -- function file (hadesllm/morie)
"""CUSUM change-point detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Study the past if you would define the future. -- Confucius"


def cusum_detect(x, threshold: float = 5.0, drift: float = 0.5, **kwargs) -> DescriptiveResult:
    """Detect change points via cumulative sum (CUSUM) algorithm.

    Parameters
    ----------
    x : array-like
        Input signal.
    threshold : float
        Decision threshold for declaring a change.
    drift : float
        Allowance (slack) parameter.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    mu = np.mean(x)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    alarms = []
    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i - 1] + (x[i] - mu) - drift)
        s_neg[i] = max(0, s_neg[i - 1] - (x[i] - mu) - drift)
        if s_pos[i] > threshold or s_neg[i] > threshold:
            alarms.append(i)
            s_pos[i] = 0
            s_neg[i] = 0
    return DescriptiveResult(
        name="cusum_detect",
        value=float(len(alarms)),
        extra={
            "alarms": np.array(alarms, dtype=int),
            "s_pos": s_pos,
            "s_neg": s_neg,
            "threshold": threshold,
            "drift": drift,
        },
    )


cusdt = cusum_detect


def cheatsheet() -> str:
    return "cusum_detect({}) -> CUSUM change-point detection."
