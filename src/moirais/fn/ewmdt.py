# moirais.fn — function file (hadesllm/moirais)
"""EWMA control chart detector."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful, the mind of a child is."


def ewma_detect(x, lambda_: float = 0.2, L: float = 3.0, **kwargs) -> DescriptiveResult:
    """Detect out-of-control points via EWMA control chart.

    Parameters
    ----------
    x : array-like
        Input signal.
    lambda_ : float
        Smoothing parameter (0 < lambda_ <= 1).
    L : float
        Control limit width in standard deviations.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    mu = np.mean(x)
    sigma = np.std(x, ddof=1) if n > 1 else 1.0
    z = np.zeros(n)
    z[0] = mu
    alarms = []
    for i in range(1, n):
        z[i] = lambda_ * x[i] + (1 - lambda_) * z[i - 1]
        factor = np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * i)))
        ucl = mu + L * sigma * factor
        lcl = mu - L * sigma * factor
        if z[i] > ucl or z[i] < lcl:
            alarms.append(i)
    return DescriptiveResult(
        name="ewma_detect",
        value=float(len(alarms)),
        extra={
            "alarms": np.array(alarms, dtype=int),
            "ewma": z,
            "lambda": lambda_,
            "L": L,
        },
    )


ewmdt = ewma_detect


def cheatsheet() -> str:
    return "ewma_detect({}) -> EWMA control chart detector."
