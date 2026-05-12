"""Survival function from life table."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def survival_function(
    qx: list[float] | np.ndarray,
    radix: int = 100000,
) -> ESRes:
    r"""Compute the survival function S(x) from a life table qx column.

    .. math::

        S(x) = \\prod_{i=0}^{x-1} (1 - q_i)

    Parameters
    ----------
    qx : array-like of float
        Age-specific probabilities of death.
    radix : int, default 100000
        Starting cohort size.

    Returns
    -------
    ESRes
        estimate is e0 (life expectancy at birth).
        extra contains S(x) array and lx.

    References
    ----------
    Preston, S. H. et al. (2001). *Demography: Measuring and Modeling
    Population Processes*. Blackwell, Ch. 3.
    """
    q = np.asarray(qx, dtype=float)
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("qx values must be in [0, 1]")

    k = len(q)
    lx = np.zeros(k + 1)
    lx[0] = radix

    for i in range(k):
        lx[i + 1] = lx[i] * (1 - q[i])

    sx = lx / radix
    e0 = float(np.sum(lx[1:]) + 0.5 * lx[0]) / radix

    return ESRes(
        measure="survival_function",
        estimate=e0,
        n=k,
        extra={"S": sx.tolist(), "lx": lx.tolist()},
    )


srvfn = survival_function


def cheatsheet() -> str:
    return "survival_function({}) -> Survival function S(x) from life table qx."
