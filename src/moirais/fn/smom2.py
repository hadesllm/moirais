"""Central moment."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "He who fights with monsters should be careful lest he thereby become a monster. — Friedrich Nietzsche"


def central_moment(x, k=2, **kwargs) -> DescriptiveResult:
    """Compute the *k*-th central moment of signal *x*.

    .. math::

        \\mu_k = \\frac{1}{N} \\sum_{n=0}^{N-1} (x(n) - \\bar{x})^k

    Parameters
    ----------
    x : array-like
        Input signal.
    k : int
        Moment order (default 2).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    mu_k = float(np.mean((x - np.mean(x)) ** k))
    return DescriptiveResult(
        name="central_moment",
        value=mu_k,
        extra={"moment_order": k, "central_moment": mu_k, "n": len(x)},
    )


smom2 = central_moment


def cheatsheet() -> str:
    return "central_moment({}) -> Central moment."
