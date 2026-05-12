# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Lin's concordance correlation coefficient."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def concordance_corr(x: np.ndarray, y: np.ndarray) -> ESRes:
    """Lin's concordance correlation coefficient (CCC).

    Measures agreement between two continuous measurements.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        raise ValueError("Need >= 3 paired observations.")

    mx, my = x.mean(), y.mean()
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.sum((x - mx) * (y - my)) / (n - 1)

    denom = sx2 + sy2 + (mx - my) ** 2
    ccc = 2 * sxy / denom if denom > 0 else 0.0

    z = np.arctanh(ccc)
    se_z = np.sqrt(1 / (n - 3)) if n > 3 else 0.0
    z_crit = sp_stats.norm.ppf(0.975)
    ci_lo = np.tanh(z - z_crit * se_z)
    ci_hi = np.tanh(z + z_crit * se_z)

    return ESRes(
        measure="concordance_corr",
        estimate=float(ccc),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        se=float(se_z),
        n=n,
        extra={"pearson_r": float(sxy / np.sqrt(sx2 * sy2)) if sx2 * sy2 > 0 else 0.0},
    )


ccc = concordance_corr


def cheatsheet() -> str:
    return "concordance_corr({}) -> Lin's concordance correlation coefficient."
