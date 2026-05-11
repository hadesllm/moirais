# morie.fn — function file (hadesllm/morie)
"""Sequential change-point detection (CUSUM/EWMA)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience there is no such thing as luck."


def sequential_detect(x, threshold=3.0, method="cusum", **kwargs) -> DescriptiveResult:
    """Sequential change-point detection.

    Parameters
    ----------
    x : array-like
        Input signal.
    threshold : float
        Detection threshold. Default 3.0.
    method : str
        ``"cusum"`` or ``"ewma"``. Default ``"cusum"``.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        sigma = 1.0

    alarms = []

    if method == "cusum":
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        for i in range(1, n):
            z = (x[i] - mu) / sigma
            s_pos[i] = max(0, s_pos[i - 1] + z - 0.5)
            s_neg[i] = max(0, s_neg[i - 1] - z - 0.5)
            if s_pos[i] > threshold or s_neg[i] > threshold:
                alarms.append(i)
        stat = s_pos
    else:
        lam = kwargs.get("lambda_", 0.2)
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = lam * (x[i] - mu) / sigma + (1 - lam) * z[i - 1]
        limit = threshold * np.sqrt(lam / (2 - lam))
        alarms = list(np.where(np.abs(z) > limit)[0])
        stat = z

    return DescriptiveResult(
        name="sequential_detect",
        value=float(len(alarms)),
        extra={
            "alarms": np.array(alarms, dtype=int),
            "statistic": stat,
            "method": method,
            "threshold": threshold,
        },
    )


seqdt = sequential_detect


def cheatsheet() -> str:
    return "sequential_detect({}) -> Sequential change-point detection (CUSUM/EWMA)."
