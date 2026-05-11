"""Trapezoidal rule integration."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. — Socrates"


def trapezoidal_integrate(x, dx=1.0, **kwargs) -> DescriptiveResult:
    """Integrate signal *x* using the trapezoidal rule.

    Parameters
    ----------
    x : array-like
        Function values at equally spaced points.
    dx : float
        Spacing between points (default 1.0).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("Need at least 2 points.")
    integral = float(np.trapezoid(x, dx=dx))
    return DescriptiveResult(
        name="trapezoidal_integrate",
        value=integral,
        extra={
            "integral": integral,
            "dx": dx,
            "n_points": len(x),
        },
    )


trprl = trapezoidal_integrate


def cheatsheet() -> str:
    return "trapezoidal_integrate({}) -> Trapezoidal rule integration."
