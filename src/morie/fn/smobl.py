"""Hjorth mobility parameter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Already know you that which you need."


def mobility(x, **kwargs) -> DescriptiveResult:
    """Compute the Hjorth Mobility parameter.

    .. math::

        \\text{Mobility} = \\sqrt{\\frac{\\text{var}(x')}{\\text{var}(x)}}

    where :math:`x'` is the first derivative (np.diff).

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    var_x = float(np.var(x, ddof=0))
    var_dx = float(np.var(dx, ddof=0))
    mob = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
    return DescriptiveResult(
        name="mobility",
        value=mob,
        extra={"mobility": mob, "var_x": var_x, "var_dx": var_dx, "n": len(x)},
    )


smobl = mobility


def cheatsheet() -> str:
    return "mobility({}) -> Hjorth mobility parameter."
