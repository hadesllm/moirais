# moirais.fn — function file (hadesllm/moirais)
"""Restricted survival time test."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["rstst"]


def rstst(time: np.ndarray, event: np.ndarray, group: np.ndarray, cdf=None, *, tau: float | None = None) -> dict:
    """Test for equality of restricted mean survival times.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    group : array-like
        Group indicator (two distinct values).
    tau : float, optional
        Restriction time. Default: minimum of group maxima.

    Returns
    -------
    dict
        rmst_0, rmst_1, difference, z_statistic, p_value.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    groups = np.unique(group)
    if len(groups) != 2:
        raise ValueError("Exactly two groups required.")

    from .rmstd import rmstd
    result = rmstd(time, event, group, tau=tau)

    se = result["se"]
    diff = result["difference"]
    z = diff / se if se > 0 else 0.0
    pval = 2 * (1 - norm.cdf(abs(z)))

    return {
        "rmst_0": result["rmst_0"],
        "rmst_1": result["rmst_1"],
        "difference": diff,
        "z_statistic": float(z),
        "p_value": float(pval),
        "tau": result["tau"],
    }


rstst_fn = rstst


def cheatsheet() -> str:
    return "rstst(time, event, group) -> Restricted survival time test."
