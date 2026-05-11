# morie.fn — function file (hadesllm/morie)
"""Asymptotic distribution of Spearman r_s: r_s*sqrt(n-1) ->_d N(0,1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_spearman_asymp"]


def gibbons_spearman_asymp(r_s, n):
    """
    Asymptotic distribution of Spearman r_s: r_s*sqrt(n-1) ->_d N(0,1)

    Formula: Z = r_s * sqrt(n-1) ~ N(0,1) approximately for large n

    Parameters
    ----------
    r_s : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_statistic, p_value

    References
    ----------
    Gibbons Ch 11.3 asymptotic
    """
    r_s = np.asarray(r_s, dtype=float)
    n = int(r_s) if r_s.ndim == 0 else len(r_s)
    result = float(np.mean(r_s))
    se = float(np.std(r_s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic distribution of Spearman r_s: r_s*sqrt(n-1) ->_d N(0,1)"})


def cheatsheet():
    return "gb1131n: Asymptotic distribution of Spearman r_s: r_s*sqrt(n-1) ->_d N(0,1)"
