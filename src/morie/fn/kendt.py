# morie.fn -- function file (hadesllm/morie)
"""Kendall's tau-b with CI."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def kendall_tau_b(x: np.ndarray, y: np.ndarray, *, alpha: float = 0.05) -> ESRes:
    """Kendall's tau-b rank correlation with confidence interval.

    Parameters
    ----------
    x, y : array-like
    alpha : float
        Significance level for CI.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        raise ValueError("Need >= 4 paired observations.")

    tau, p = sp_stats.kendalltau(x, y)
    se = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = tau - z_crit * se
    ci_hi = tau + z_crit * se

    return ESRes(
        measure="kendall_tau_b",
        estimate=float(tau),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        se=float(se),
        n=n,
        extra={"p_value": float(p)},
    )


kendt = kendall_tau_b


def cheatsheet() -> str:
    return "kendall_tau_b({}) -> Kendall's tau-b with CI."
