# moirais.fn — function file (hadesllm/moirais)
"""Power law scaling. 'Me Grimlock strongest!' -- Grimlock"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def power_law_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float | None = None,
) -> DescriptiveResult:
    """Fit a power law y = a * x^b via log-linear regression.

    Applicable to allometric scaling (Kleiber's law, metabolic scaling),
    city sizes (Zipf), earthquake magnitudes, etc.

    Parameters
    ----------
    x, y : array-like
        Positive-valued data.
    x_min : float, optional
        Minimum x value to include (for truncated power laws).

    Returns
    -------
    DescriptiveResult
        With ``value`` = dict(a, b) and ``extra`` containing R-squared.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    mask = (x > 0) & (y > 0)
    if x_min is not None:
        mask &= x >= x_min
    x_f, y_f = x[mask], y[mask]
    if len(x_f) < 3:
        raise ValueError("Need at least 3 valid positive data points")

    log_x = np.log(x_f)
    log_y = np.log(y_f)
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    y_hat = a * x_f**b
    ss_res = np.sum((y_f - y_hat) ** 2)
    ss_tot = np.sum((y_f - y_f.mean()) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-30)

    return DescriptiveResult(
        name="power_law_fit",
        value={"a": float(a), "b": float(b)},
        extra={"r_squared": float(r_squared), "n": len(x_f), "x_range": (float(x_f.min()), float(x_f.max()))},
    )


grimk = power_law_fit


def cheatsheet() -> str:
    return "power_law_fit({}) -> Power law scaling. 'Me Grimlock strongest!' -- Grimlock"
