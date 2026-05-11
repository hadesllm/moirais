"""Somers' D asymmetric measure of association."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def somers_d(
    x,
    y,
) -> ESRes:
    """Somers' D for ordinal association (asymmetric).

    D(Y|X) = (concordant - discordant) / (concordant + discordant + tied_on_X).

    Parameters
    ----------
    x, y : array-like
        Ordinal variables.

    Returns
    -------
    ESRes
    """
    from scipy.stats import somersd

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    result = somersd(x, y)
    return ESRes(
        measure="somers_d",
        estimate=float(result.statistic),
        n=n,
        extra={"p_value": float(result.pvalue)},
    )


somrd = somers_d


def cheatsheet() -> str:
    return "somers_d(x, y) -> Somers' D ordinal association."
