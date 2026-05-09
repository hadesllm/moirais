"""Simpson's rule integration."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def simpson_integrate(x, dx=1.0, **kwargs) -> DescriptiveResult:
    """Integrate signal *x* using Simpson's 1/3 rule.

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
    from scipy.integrate import simpson

    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        raise ValueError("Need at least 3 points for Simpson's rule.")
    integral = float(simpson(x, dx=dx))
    return DescriptiveResult(
        name="simpson_integrate",
        value=integral,
        extra={
            "integral": integral,
            "dx": dx,
            "n_points": len(x),
        },
    )


smpsn = simpson_integrate


def cheatsheet() -> str:
    return "simpson_integrate({}) -> Simpson's rule integration."
