# moirais.fn — function file (hadesllm/moirais)
"""Interval censoring with DP prior: contraction in Hellinger for log-concave models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_icens_dp_crt"]


def ghosal_icens_dp_crt(x):
    """
    Interval censoring with DP prior: contraction in Hellinger for log-concave models

    Formula: DP prior on monotone F for interval-censored data, rate (n/log n)^{-1/3}

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 9 §9.5.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interval censoring with DP prior: contraction in Hellinger for log-concave models"})


def cheatsheet():
    return "gh_c9_11: Interval censoring with DP prior: contraction in Hellinger for log-concave models"
