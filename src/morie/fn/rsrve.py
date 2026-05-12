# morie.fn -- function file (hadesllm/morie)
"""Chain-ladder loss reserving."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def chain_ladder(triangle: np.ndarray) -> DescriptiveResult:
    r"""Chain-ladder method for loss reserving.

    Estimates ultimate claims from a cumulative loss development
    triangle using age-to-age (link) factors:

    .. math::

        f_j = \\frac{\\sum_i C_{i,j+1}}{\\sum_i C_{i,j}}

    The IBNR (incurred but not reported) reserve is the difference
    between projected ultimate and current cumulative paid.

    Parameters
    ----------
    triangle : ndarray
        Upper-left cumulative loss triangle (n x n).  Entry (i, j)
        is the cumulative claims for accident year *i* at development
        period *j*.  Lower-right entries should be 0 or NaN (to be
        projected).

    Returns
    -------
    DescriptiveResult
        ``value`` is the total IBNR reserve.  ``extra`` has
        ``ultimate`` (projected ultimates per year), ``ibnr``
        (per-year reserves), ``link_factors``, ``completed``
        (filled triangle).

    Raises
    ------
    ValueError
        If triangle is not square or has invalid values.

    References
    ----------
    Mack, T. (1993). Distribution-free calculation of the standard
    error of chain ladder reserve estimates. *ASTIN Bulletin*,
    23(2), 213--225.

    England, P. D., & Verrall, R. J. (2002). Stochastic claims
    reserving in general insurance. *British Actuarial Journal*,
    8(3), 443--518.
    """
    T = np.asarray(triangle, dtype=np.float64)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("triangle must be a square matrix.")

    n = T.shape[0]
    completed = T.copy()

    completed[np.isnan(completed)] = 0.0

    link_factors = np.ones(n - 1)
    for j in range(n - 1):
        num = 0.0
        den = 0.0
        for i in range(n - 1 - j):
            if completed[i, j] > 0 and completed[i, j + 1] > 0:
                num += completed[i, j + 1]
                den += completed[i, j]
        link_factors[j] = num / den if den > 0 else 1.0

    for i in range(1, n):
        last_known = n - i - 1
        for j in range(last_known + 1, n):
            completed[i, j] = completed[i, j - 1] * link_factors[j - 1]

    ultimate = completed[:, -1].copy()
    current = np.array([completed[i, n - i - 1] if n - i - 1 >= 0 else 0.0 for i in range(n)])
    ibnr = ultimate - current
    total_ibnr = float(np.sum(ibnr))

    return DescriptiveResult(
        name="ChainLadder",
        value=total_ibnr,
        extra={
            "ultimate": ultimate,
            "ibnr": ibnr,
            "link_factors": link_factors,
            "completed": completed,
            "n_years": n,
        },
    )


rsrve = chain_ladder


def cheatsheet() -> str:
    return "chain_ladder({}) -> Chain-ladder loss reserving."
