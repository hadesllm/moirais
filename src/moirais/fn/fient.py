# moirais.fn — function file (hadesllm/moirais)
"""Fisher information."""

import numpy as np

from ._containers import ESRes


def fisher_information(x, bins: int = 50, **kwargs) -> ESRes:
    """
    Estimate Fisher information from data via histogram density estimation.

    .. math::

        I_F = \\int \\frac{[p'(x)]^2}{p(x)} dx

    Approximated using finite differences on the histogram density.

    :param x: array-like data.
    :param bins: Number of histogram bins.
    :return: ESRes with Fisher information estimate.

    References
    ----------
    Fisher RA (1925). Theory of statistical estimation.
    Mathematical Proceedings of the Cambridge Philosophical Society,
    22(5), 700-725.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 3:
        raise ValueError("Need at least 3 observations.")
    counts, edges = np.histogram(x, bins=bins, density=True)
    dx = edges[1] - edges[0]
    p = counts.astype(np.float64)
    dp = np.gradient(p, dx)
    mask = p > 0
    fi = float(np.sum((dp[mask] ** 2) / p[mask]) * dx)
    return ESRes(
        measure="fisher_information",
        estimate=fi,
        n=len(x),
        extra={"n_bins": bins},
    )


fient = fisher_information


def cheatsheet() -> str:
    return "fisher_information(x) -> Fisher information estimate."
