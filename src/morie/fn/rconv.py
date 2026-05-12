# morie.fn -- function file (hadesllm/morie)
"""Convergence rate estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Big results require big ambitions. -- Heraclitus"


def convergence_rate(errors, **kwargs) -> DescriptiveResult:
    """Estimate convergence rate from an error sequence.

    Fits a log-linear model to successive error values to estimate
    the exponential convergence rate.

    Parameters
    ----------
    errors : array-like
        Sequence of error values (must be positive).

    Returns
    -------
    DescriptiveResult
    """
    e = np.asarray(errors, dtype=float)
    if len(e) < 2:
        raise ValueError("Need at least 2 error values.")
    if np.any(e <= 0):
        raise ValueError("All errors must be positive.")
    log_e = np.log(e)
    t = np.arange(len(e), dtype=float)
    coeffs = np.polyfit(t, log_e, 1)
    rate = -coeffs[0]
    return DescriptiveResult(
        name="convergence_rate",
        value=float(rate),
        extra={
            "rate": float(rate),
            "intercept": float(coeffs[1]),
            "n_points": len(e),
            "converging": bool(rate > 0),
        },
    )


rconv = convergence_rate


def cheatsheet() -> str:
    return "convergence_rate({}) -> Convergence rate estimation."
