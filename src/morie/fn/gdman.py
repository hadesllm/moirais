# morie.fn — function file (hadesllm/morie)
"""Goodman-Kruskal gamma."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def goodman_kruskal_gamma(
    x,
    y,
) -> ESRes:
    """Goodman-Kruskal gamma for ordinal association.

    gamma = (concordant - discordant) / (concordant + discordant).

    Parameters
    ----------
    x, y : array-like
        Ordinal variables.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = concordant + discordant
    gamma = (concordant - discordant) / denom if denom > 0 else 0.0

    se = 0.0
    if denom > 0 and n > 2:
        se = 2.0 * np.sqrt(concordant * discordant) / (denom * np.sqrt(n))
    z = gamma / se if se > 1e-12 else 0.0
    from scipy.stats import norm

    pval = 2.0 * norm.sf(abs(z)) if se > 1e-12 else 1.0

    return ESRes(
        measure="goodman_kruskal_gamma",
        estimate=float(gamma),
        se=float(se),
        n=n,
        extra={
            "concordant": concordant,
            "discordant": discordant,
            "z": float(z),
            "p_value": float(pval),
        },
    )


gdman = goodman_kruskal_gamma


def cheatsheet() -> str:
    return "goodman_kruskal_gamma(x, y) -> Goodman-Kruskal gamma."
