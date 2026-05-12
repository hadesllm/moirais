# morie.fn — function file (hadesllm/morie)
"""Maximum LMS step size."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You must unlearn what you have learned."


def max_step_size(x, order: int = 16, **kwargs) -> DescriptiveResult:
    r"""Compute the maximum stable LMS step size.

    .. math::

        \\mu_{\\max} = \\frac{2}{M \\cdot P_x}

    where *M* is the filter order and *Px* is the input signal power.

    Parameters
    ----------
    x : array-like
        Input signal.
    order : int
        Filter order (number of taps).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    Px = float(np.mean(x**2))
    if Px <= 0:
        raise ValueError("Signal power is zero; step size is undefined.")
    if order <= 0:
        raise ValueError("Filter order must be positive.")
    mu_max = 2.0 / (order * Px)
    return DescriptiveResult(
        name="max_step_size",
        value=mu_max,
        extra={"mu_max": mu_max, "order": order, "Px": Px},
    )


mstep = max_step_size


def cheatsheet() -> str:
    return "max_step_size({}) -> Maximum LMS step size."
