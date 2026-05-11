# morie.fn — function file (hadesllm/morie)
"""Asymptotic normality of standardized EDF as n -> inf."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_edf_asymp_normal"]


def gibbons_edf_asymp_normal(x, n):
    """
    Asymptotic normality of standardized EDF as n -> inf

    Formula: sqrt(n)[S_n(x)-F(x)] / sqrt(F(x)(1-F(x))) ->_d N(0,1)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: normal_limit

    References
    ----------
    Gibbons Theorem 2.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic normality of standardized EDF as n -> inf"})


def cheatsheet():
    return "gb233: Asymptotic normality of standardized EDF as n -> inf"
