# moirais.fn — function file (hadesllm/moirais)
"""Polychoric correlation (two-step approximation)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def polychoric_corr(x_ordinal: np.ndarray, y_ordinal: np.ndarray) -> ESRes:
    """Polychoric correlation via Pearson on normal-scores proxy.

    Uses normal-quantile transformation of the ordinal marginals
    as a fast two-step approximation.

    Parameters
    ----------
    x_ordinal, y_ordinal : array-like (integer-coded ordinal)

    Returns
    -------
    ESRes
    """
    x = np.asarray(x_ordinal, dtype=float).ravel()
    y = np.asarray(y_ordinal, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        raise ValueError("Need >= 5 paired observations.")

    def _normal_scores(v):
        ranks = sp_stats.rankdata(v, method="average")
        return sp_stats.norm.ppf(ranks / (len(v) + 1))

    zx = _normal_scores(x)
    zy = _normal_scores(y)
    r_poly = float(np.corrcoef(zx, zy)[0, 1])
    se = np.sqrt((1 - r_poly**2) / (n - 2)) if n > 2 else 0.0

    return ESRes(
        measure="polychoric_r",
        estimate=r_poly,
        se=float(se),
        n=n,
        extra={"method": "normal_scores_approximation"},
    )


polyc = polychoric_corr


def cheatsheet() -> str:
    return "polychoric_corr({}) -> Polychoric correlation (two-step approximation)."
